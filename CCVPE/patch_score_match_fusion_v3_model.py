import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import ImageFile
from mmcv.ops import MultiScaleDeformableAttention
from efficientnet_pytorch.model import EfficientNet
from model_utils.fused_image_deformable_fusion_v2 import deformable_fusion
from model_utils.position_encoding import PositionEncodingSine
from einops import rearrange
from model_utils.grd_descriptors import GroundDescriptors
from model_utils.ccvpe_modules import CCVPEModules
from dual_datasets import DatasetType

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)


class permute_channels(nn.Module):
    def __init__(self, B, C, H, W):
        super(permute_channels, self).__init__()
        self.B = B
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return torch.permute(x, (self.B, self.C, self.H, self.W))


class normalization(nn.Module):
    def __init__(self, p, dim):
        super(normalization, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
    )


class CVM_VIGOR(nn.Module):
    def __init__(
        self,
        device,
        circular_padding,
        use_adapt,
        use_concat,
        use_mlp=False,
        alpha_type=0,
        dataset_type: DatasetType = DatasetType.VIGOR,
    ):
        print("load in dataset_type ", dataset_type)
        super().__init__()
        self.device = device
        self.circular_padding = circular_padding
        self.use_adapt = use_adapt  # If using osm tiles with 50 layers
        self.use_concat = use_concat  # If using simple fusion with concat
        self.use_mlp = use_mlp

        self.adapt_concat = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.adapt_50_n = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.ground_descriptors = GroundDescriptors(
            dataset_type=dataset_type, circular_padding=circular_padding
        )

        self.grd_efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", self.circular_padding
        )

        self.grd_feature_to_descriptor1 = nn.Sequential(
            nn.Conv2d(1280, 64, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1),
        )

        self.grd_feature_to_descriptor2 = nn.Sequential(
            nn.Conv2d(1280, 32, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1),
        )

        self.grd_feature_to_descriptor3 = nn.Sequential(
            nn.Conv2d(1280, 16, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1),
        )

        self.grd_feature_to_descriptor4 = nn.Sequential(
            nn.Conv2d(1280, 8, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1),
        )

        self.grd_feature_to_descriptor5 = nn.Sequential(
            nn.Conv2d(1280, 4, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1),
        )

        self.grd_feature_to_descriptor6 = nn.Sequential(
            nn.Conv2d(1280, 2, 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(10, 1, 1),
            nn.Flatten(start_dim=1),
        )

        match dataset_type:
            case DatasetType.KITTI:
                self.rolling_number = 16
                self.shift_scale = [128, 64, 32, 16, 8, 8]
                self.embed_dims = [2048, 320, 112, 40, 24, 16]
                self.first_embed = 2048
            case DatasetType.VIGOR:
                self.rolling_number = 20
                self.shift_scale = [64, 32, 16, 8, 4, 2]
                self.embed_dims = [1280, 320, 112, 40, 24, 16]
                self.first_embed = 1280

        self.sat_efficientnet = EfficientNet.from_pretrained("efficientnet-b0", circular=False)

        self.sat_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, self.first_embed)
        )

        self.sat_normalization = normalization(2, 1)

        self.osm_efficientnet = EfficientNet.from_pretrained("efficientnet-b0", circular=False)

        self.osm_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, self.first_embed)
        )

        self.osm_normalization = normalization(2, 1)

        self.ccvpe_modules = CCVPEModules(dataset_type=dataset_type)

        self.convs = self.ccvpe_modules.get_convs()
        self.deconvs = self.ccvpe_modules.get_deconvs()
        self.convs_ori = self.ccvpe_modules.get_convs_ori()
        self.deconvs_ori = self.ccvpe_modules.get_deconvs_ori()

        self.fuse_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, self.first_embed)
        )
        self.fuse_normalization = normalization(2, 1)

        self.deformable_fusion = deformable_fusion(self.device, alpha_type)
        self.heatmap_norm = nn.LayerNorm(normalized_shape=(512, 512))

        self.fusion_volume = FusionModule(
            self.embed_dims[0], 8, 8, self.rolling_number, 8, device
        )  # embed dim, H, W, input_embed_dim, patch_size
        self.fusion5 = FusionModule(self.embed_dims[1], 16, 16, self.rolling_number, 16, device)
        self.fusion4 = FusionModule(self.embed_dims[2], 32, 32, self.rolling_number, 16, device)
        self.fusion3 = FusionModule(self.embed_dims[3], 64, 64, self.rolling_number, 16, device)
        self.fusion2 = FusionModule(self.embed_dims[4], 128, 128, self.rolling_number, 16, device)
        self.fusion1 = FusionModule(self.embed_dims[5], 256, 256, self.rolling_number, 16, device)

        self.learnable_Q = nn.Embedding(8 * 8, self.rolling_number)

        self.fusions = nn.ModuleList(
            [
                self.fusion5,
                self.fusion4,
                self.fusion3,
                self.fusion2,
                self.fusion1,
            ]
        )

    def get_input_proj_list(self, channels, embed_dims, num_levels):
        """
        Use to get uniform channels accross all levels, will preserve W and H dims.
        """
        input_proj_list = []
        for i in range(num_levels):
            in_channels = channels[i]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, embed_dims, kernel_size=1),
                    nn.GroupNorm(32, embed_dims),
                )
            )
        return nn.ModuleList(input_proj_list)

    def forward(self, grd, sat, osm):
        grd_descriptors, grd_descriptor_maps, grd_feature_volume = self.ground_descriptors(grd)

        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat)
        sat_feature_block0 = multiscale_sat[0]  # [16, 256, 256]
        sat_feature_block2 = multiscale_sat[2]  # [24, 128, 128]
        sat_feature_block4 = multiscale_sat[4]  # [40, 64, 64]
        sat_feature_block10 = multiscale_sat[10]  # [112, 32, 32]
        sat_feature_block15 = multiscale_sat[15]  # [320, 16, 16]

        osm_feature_volume, multiscale_osm = self.osm_efficientnet.extract_features_multiscale(osm)

        osm_feature_block0 = multiscale_osm[0]  # [16, 256, 256]
        osm_feature_block2 = multiscale_osm[2]  # [24, 128, 128]
        osm_feature_block4 = multiscale_osm[4]  # [40, 64, 64]
        osm_feature_block10 = multiscale_osm[10]  # [112, 32, 32]
        osm_feature_block15 = multiscale_osm[15]  # [320, 16, 16]

        batch_size = sat.size(0)  # Get batch size dynamically

        fuse_feature_blocks = [
            (sat_feature_block15, osm_feature_block15),
            (sat_feature_block10, osm_feature_block10),
            (sat_feature_block4, osm_feature_block4),
            (sat_feature_block2, osm_feature_block2),
            (sat_feature_block0, osm_feature_block0),
            None,
        ]  # None is an edge case, in the last step we don't concat

        sat_descriptor_map = self.get_descriptor(sat_feature_volume)
        osm_descriptor_map = self.get_descriptor(osm_feature_volume)

        batch_size = grd.shape[0]
        f_score = (
            self.learnable_Q.weight.reshape(8, 8, self.rolling_number)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1, 1)
        )
        fuse_descriptor_map, a = self.fusion_volume(f_score, sat_descriptor_map, osm_descriptor_map)

        x = fuse_descriptor_map
        matching_score_stacked_list = []

        fuse_features = []
        alphas = [a]
        # Perform localization decoder for each level (The whole localization decoder)
        for level, (
            grd_descriptor,
            grd_descriptor_map,
            fuse_feature_block,
        ) in enumerate(zip(grd_descriptors, grd_descriptor_maps, fuse_feature_blocks)):
            if fuse_feature_block is not None:
                x, matching_score_stacked, da_output, a = self.localization_decoder(
                    level,
                    x,
                    grd_descriptor,
                    grd_descriptor_map,
                    fuse_feature_block,
                    self.rolling_number,
                )
                alphas.append(a)
            else:
                x, matching_score_stacked, da_output = self.localization_decoder(
                    level,
                    x,
                    grd_descriptor,
                    grd_descriptor_map,
                    fuse_feature_block,
                    self.rolling_number,
                )
            fuse_features.append(da_output)
            matching_score_stacked_list.append(matching_score_stacked)

        logits_flattened = torch.flatten(x, start_dim=1)
        heatmap = torch.reshape(nn.Softmax(dim=-1)(logits_flattened), x.size())

        # Perform orientation decoder for each level
        x_ori = torch.cat(
            [
                matching_score_stacked_list[0],
                self.fuse_normalization(fuse_descriptor_map),
            ],
            dim=1,
        )

        for level, fuse_feature_block in enumerate(fuse_features):
            x_ori = self.orientation_decoder(level, x_ori, fuse_feature_block)

        x_ori = nn.functional.normalize(x_ori, p=2, dim=1)
        alphas = torch.cat(alphas, dim=1)
        return (alphas, logits_flattened, heatmap, x_ori) + tuple(matching_score_stacked_list)

    def get_descriptor(self, fuse_feature_volume):
        fuse_row_chunks = torch.stack(
            list(torch.chunk(fuse_feature_volume, 8, 2)), dim=-1
        )  # dimension 4 is the number of row chunks (splitted in height dimension)
        for i, fuse_row_chunk in enumerate(torch.unbind(fuse_row_chunks, dim=-1), 0):
            fuse_chunks = torch.stack(
                list(torch.chunk(fuse_row_chunk, 8, 3)), dim=-1
            )  # dimension 5 is the number of vertical chunks (splitted in width dimension)
            for j, fuse_chunk in enumerate(torch.unbind(fuse_chunks, dim=-1), 0):
                if j == 0:
                    fuse_descriptor_row = (
                        self.fuse_feature_to_descriptors(fuse_chunk).unsqueeze(2).unsqueeze(3)
                    )
                else:
                    fuse_descriptor_row = torch.cat(
                        (
                            fuse_descriptor_row,
                            self.fuse_feature_to_descriptors(fuse_chunk).unsqueeze(2).unsqueeze(3),
                        ),
                        3,
                    )
            if i == 0:
                fuse_descriptor_map = fuse_descriptor_row
            else:
                fuse_descriptor_map = torch.cat((fuse_descriptor_map, fuse_descriptor_row), 2)

        return fuse_descriptor_map

    def compute_matching_score(
        self, shift, x, grd_des_len, grd_descriptor_map, grd_map_norm, rolling_number=20
    ):
        """
        LMU component: rolling and matching part
        TODO: use it once instead
        """
        for i in range(rolling_number):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i * shift, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:, :grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p="fro", dim=1, keepdim=True)

            matching_score = torch.sum(
                (grd_descriptor_map * sat_descriptor_map_window), dim=1, keepdim=True
            ) / (sat_map_norm * grd_map_norm)  # cosine similarity
            if i == 0:
                matching_score_stacked = matching_score
            else:
                matching_score_stacked = torch.cat([matching_score_stacked, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)

        return matching_score_max, matching_score_stacked

    def localization_decoder(
        self, level, x, grd_descriptor, grd_descriptor_map, fuse_feature_block, rolling_number
    ):
        grd_des_len = grd_descriptor.size()[1]
        fuse_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map, p="fro", dim=1, keepdim=True)

        shift = self.shift_scale[level]
        matching_score_max, matching_score_stacked = self.compute_matching_score(
            shift, x, grd_des_len, grd_descriptor_map, grd_map_norm, rolling_number
        )
        # loc
        x = torch.cat([matching_score_max, self.fuse_normalization(x)], dim=1)

        x = self.deconvs[level](x)
        da_output = None
        if fuse_feature_block is not None:
            da_output, a = self.fusions[level](
                matching_score_stacked, fuse_feature_block[0], fuse_feature_block[1]
            )
            x = torch.cat([x, da_output], dim=1)

        x = self.convs[level](x)

        if fuse_feature_block is None:
            return x, matching_score_stacked, da_output
        else:
            return x, matching_score_stacked, da_output, a

    def orientation_decoder(self, level, x_ori, fuse_feature_block):
        x_ori = self.deconvs_ori[level](x_ori)
        if fuse_feature_block is not None:
            x_ori = torch.cat([x_ori, fuse_feature_block], dim=1)
        x_ori = self.convs_ori[level](x_ori)
        return x_ori


class FusionModule(nn.Module):
    def __init__(self, embed_dim, H, W, input_embed_dim, patch_size, device):
        super().__init__()

        self.ca_osm = CrossAttention(embed_dim, H, W, input_embed_dim, device)
        self.ca_sat = CrossAttention(embed_dim, H, W, input_embed_dim, device)
        self.conv = nn.Conv2d(in_channels=input_embed_dim, out_channels=embed_dim, kernel_size=1)
        self.upsample = nn.Upsample(size=(H, W), mode="bilinear")
        self.patch_router = PatchRouter(
            embed_dim * patch_size * patch_size, patch_size=patch_size, embed_dim=embed_dim
        )

    def forward(self, ground, sat, osm):
        ground = self.upsample(ground)

        ground = self.conv(ground)

        sat_output = self.ca_sat(ground, sat)
        osm_output = self.ca_osm(ground, osm)

        return self.patch_router(sat_output, osm_output)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, H, W, input_embed_dim, device):
        super().__init__()

        grd_row_self_ = np.linspace(0, 1, W)
        grd_col_self_ = np.linspace(0, 1, H)
        grd_row_self, grd_col_self = np.meshgrid(grd_row_self_, grd_col_self_, indexing="ij")

        self.reference_points = (
            torch.stack((torch.tensor(grd_col_self), torch.tensor(grd_row_self)), -1)
            .view(-1, 2)
            .unsqueeze(1)
            .to(torch.float)
            .to(device)
        )
        self.spatial_shape = torch.tensor(([[H, W]])).to(device).long()
        self.level_start_index = torch.tensor([0]).to(device)

        self.attention = MultiScaleDeformableAttention(
            embed_dims=embed_dim, num_levels=1, batch_first=True
        )

        self.pe = PositionEncodingSine(embed_dim)
        self.H = H
        self.W = W
        self.embed_dim = embed_dim

    def forward(self, query, value):
        query += self.pe(query)
        query_flattened = query.flatten(start_dim=2).permute(0, 2, 1)
        value_flattened = value.flatten(start_dim=2).permute(0, 2, 1)

        reference_points = self.reference_points.unsqueeze(0).repeat(query.shape[0], 1, 1, 1)
        output = self.attention(
            query=query_flattened,
            value=value_flattened,
            reference_points=reference_points,
            spatial_shapes=self.spatial_shape,
            level_start_index=self.level_start_index,
        )
        output = output.permute(0, 2, 1).view(query.shape[0], self.embed_dim, self.H, self.W)
        return output + query


class PatchRouter(nn.Module):
    def __init__(self, D, patch_size=16, hard=False, embed_dim=64):
        super().__init__()
        self.logit = nn.Linear(D, 2)
        self.patch_size = patch_size
        self.hard = hard
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, temp=0.1):
        p1 = self.norm1(p1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        p2 = self.norm2(p2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        B, C, H, W = p1.shape
        ph = pw = self.patch_size
        # --- einop patchify ---
        #   (B,C,H,W) → (B, num_patches, patch_dim)
        p1 = rearrange(p1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=ph, pw=pw)
        p2 = rearrange(p2, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=ph, pw=pw)

        B, N, D = p1.shape  # now N = (H/ph)*(W/pw), D = C*ph*pw
        # 1) score each patch
        x = 0.5 * (p1 + p2)  # [B,N,D]
        logits = self.logit(x)  # [B,N,2]
        # We rescale in the same fashion as in attention, or we will get mostly 0 and 1
        scale = D**0.5

        # 2) gumbel‐softmax into weights
        weights = F.softmax((logits / scale).flatten(0, 1), dim=-1).view(B, N, 2)  # [B,N,2]

        # 3) stack & route: [B,N,2,D] → [B,N,D]
        stacked = torch.stack([p1, p2], dim=2)
        routed = (weights.unsqueeze(-1) * stacked).sum(
            dim=2
        )  # unsqueeze -1 since we apply the same for all pixel in a given patch

        out = rearrange(
            routed,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=H // ph,
            w=W // pw,
            ph=ph,
            pw=pw,
        )

        return out, weights
