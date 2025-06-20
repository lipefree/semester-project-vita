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
    ):
        super(CVM_VIGOR, self).__init__()
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

        self.sat_efficientnet = EfficientNet.from_pretrained("efficientnet-b0", circular=False)

        self.sat_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, 1280)
        )

        self.sat_normalization = normalization(2, 1)

        self.osm_efficientnet = EfficientNet.from_pretrained("efficientnet-b0", circular=False)

        self.osm_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, 1280)
        )

        self.osm_normalization = normalization(2, 1)

        # loc
        self.deconv6 = nn.ConvTranspose2d(1281, 1024, 2, 2)
        self.conv6 = double_conv(1344, 640)

        self.deconv5 = nn.ConvTranspose2d(641, 320, 2, 2)
        self.conv5 = double_conv(432, 320)

        self.deconv4 = nn.ConvTranspose2d(321, 160, 2, 2)
        self.conv4 = double_conv(200, 160)

        self.deconv3 = nn.ConvTranspose2d(161, 80, 2, 2)
        self.conv3 = double_conv(104, 80)

        self.deconv2 = nn.ConvTranspose2d(81, 40, 2, 2)
        self.conv2 = double_conv(56, 40)

        self.deconv1 = nn.ConvTranspose2d(41, 16, 2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
        )

        self.convs = nn.ModuleList(
            [self.conv6, self.conv5, self.conv4, self.conv3, self.conv2, self.conv1]
        )
        self.deconvs = nn.ModuleList(
            [
                self.deconv6,
                self.deconv5,
                self.deconv4,
                self.deconv3,
                self.deconv2,
                self.deconv1,
            ]
        )

        # ori
        self.deconv6_ori = nn.ConvTranspose2d(1300, 1024, 2, 2)
        self.conv6_ori = double_conv(1344, 640)

        self.deconv5_ori = nn.ConvTranspose2d(640, 256, 2, 2)
        self.conv5_ori = double_conv(368, 256)

        self.deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4_ori = double_conv(168, 128)

        self.deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3_ori = double_conv(88, 64)

        self.deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2_ori = double_conv(48, 32)

        self.deconv1_ori = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv1_ori = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, stride=1, padding=1),
        )

        self.convs_ori = nn.ModuleList(
            [
                self.conv6_ori,
                self.conv5_ori,
                self.conv4_ori,
                self.conv3_ori,
                self.conv2_ori,
                self.conv1_ori,
            ]
        )
        self.deconvs_ori = nn.ModuleList(
            [
                self.deconv6_ori,
                self.deconv5_ori,
                self.deconv4_ori,
                self.deconv3_ori,
                self.deconv2_ori,
                self.deconv1_ori,
            ]
        )

        self.fuse_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, 1280)
        )
        self.fuse_normalization = normalization(2, 1)

        self.router_volume = PatchRouter(1280 * 16 * 16, embed_dim=1280)
        self.router15 = PatchRouter(320 * 16 * 16, embed_dim=320)
        self.router10 = PatchRouter(112 * 16 * 16, embed_dim=112)
        self.router4 = PatchRouter(40 * 16 * 16, embed_dim=40)
        self.router2 = PatchRouter(24 * 16 * 16, embed_dim=24)
        self.router0 = PatchRouter(16 * 16 * 16, embed_dim=16)

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

    def forward(self, grd, sat, osm, scores, temp=0.1):
        grd_feature_volume = self.grd_efficientnet.extract_features(grd)
        grd_descriptor1 = self.grd_feature_to_descriptor1(grd_feature_volume)  # length 1280
        grd_descriptor2 = self.grd_feature_to_descriptor2(grd_feature_volume)  # length 640
        grd_descriptor3 = self.grd_feature_to_descriptor3(grd_feature_volume)  # length 320
        grd_descriptor4 = self.grd_feature_to_descriptor4(grd_feature_volume)  # length 160
        grd_descriptor5 = self.grd_feature_to_descriptor5(grd_feature_volume)  # length 80
        grd_descriptor6 = self.grd_feature_to_descriptor6(grd_feature_volume)  # length 40

        grd_descriptors = [
            grd_descriptor1,
            grd_descriptor2,
            grd_descriptor3,
            grd_descriptor4,
            grd_descriptor5,
            grd_descriptor6,
        ]

        grd_descriptor_map1 = grd_descriptor1.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        grd_descriptor_map2 = grd_descriptor2.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)
        grd_descriptor_map3 = grd_descriptor3.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)
        grd_descriptor_map4 = grd_descriptor4.unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
        grd_descriptor_map5 = grd_descriptor5.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)
        grd_descriptor_map6 = grd_descriptor6.unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256)

        grd_descriptor_maps = [
            grd_descriptor_map1,
            grd_descriptor_map2,
            grd_descriptor_map3,
            grd_descriptor_map4,
            grd_descriptor_map5,
            grd_descriptor_map6,
        ]

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

        fusion_feature_block15, weights15 = self.router15(
            sat_feature_block15, osm_feature_block15, temp
        )
        fusion_feature_block10, weights10 = self.router10(
            sat_feature_block10, osm_feature_block10, temp
        )
        fusion_feature_block4, weights4 = self.router4(sat_feature_block4, osm_feature_block4, temp)
        fusion_feature_block2, weights2 = self.router2(sat_feature_block2, osm_feature_block2, temp)
        fusion_feature_block0, weights0 = self.router0(sat_feature_block0, osm_feature_block0, temp)
        fuse_feature_volume, weights_volume = self.router_volume(
            sat_feature_volume, osm_feature_volume, temp
        )
        weights = torch.cat(
            (weights_volume, weights15, weights10, weights4, weights2, weights0), dim=1
        )  # we get decision for all patch at all scales

        fuse_feature_blocks = [
            fusion_feature_block15,
            fusion_feature_block10,
            fusion_feature_block4,
            fusion_feature_block2,
            fusion_feature_block0,
            None,
        ]  # None is an edge case, in the last step we don't concat

        fuse_descriptor_map = self.get_descriptor(fuse_feature_volume)

        x = fuse_descriptor_map
        matching_score_stacked_list = []

        # Perform localization decoder for each level (The whole localization decoder)
        for level, (
            grd_descriptor,
            grd_descriptor_map,
            fuse_feature_block,
        ) in enumerate(zip(grd_descriptors, grd_descriptor_maps, fuse_feature_blocks)):
            x, matching_score_stacked = self.localization_decoder(
                level, x, grd_descriptor, grd_descriptor_map, fuse_feature_block
            )
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

        for level, fuse_feature_block in enumerate(fuse_feature_blocks):
            x_ori = self.orientation_decoder(level, x_ori, fuse_feature_block)

        x_ori = nn.functional.normalize(x_ori, p=2, dim=1)

        return (weights, logits_flattened, heatmap, x_ori) + tuple(matching_score_stacked_list)

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

    def compute_matching_score(self, shift, x, grd_des_len, grd_descriptor_map, grd_map_norm):
        """
        LMU component: rolling and matching part
        TODO: use it once instead
        """
        for i in range(20):
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
        self, level, x, grd_descriptor, grd_descriptor_map, fuse_feature_block
    ):
        grd_des_len = grd_descriptor.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map, p="fro", dim=1, keepdim=True)

        shift = int(64 / 2**level)
        matching_score_max, matching_score_stacked = self.compute_matching_score(
            shift, x, grd_des_len, grd_descriptor_map, grd_map_norm
        )

        # loc
        x = torch.cat([matching_score_max, self.fuse_normalization(x)], dim=1)

        x = self.deconvs[level](x)
        if fuse_feature_block is not None:
            x = torch.cat([x, fuse_feature_block], dim=1)
        x = self.convs[level](x)

        return x, matching_score_stacked

    def orientation_decoder(self, level, x_ori, fuse_feature_block):
        x_ori = self.deconvs_ori[level](x_ori)
        if fuse_feature_block is not None:
            x_ori = torch.cat([x_ori, fuse_feature_block], dim=1)
        x_ori = self.convs_ori[level](x_ori)
        return x_ori


class PatchRouter(nn.Module):
    def __init__(self, D, patch_size=16, hard=False, embed_dim=64):
        super().__init__()
        self.logit = nn.Linear(D, 2)
        self.patch_size = patch_size
        self.hard = hard
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

        # 2) softmax into weights
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
