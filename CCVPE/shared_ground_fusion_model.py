import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import ImageFile
from mmcv.ops import MultiScaleDeformableAttention
from efficientnet_pytorch.model import EfficientNet
from model_utils.fused_image_deformable_fusion_v2 import deformable_fusion
from model_utils.position_encoding import PositionEncodingSine

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
    def __init__(self, device, circular_padding, use_adapt, use_concat, use_mlp=False, alpha_type=0):
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

        self.sat_efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", circular=False
        )

        self.sat_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, 1280)
        )

        self.sat_normalization = normalization(2, 1)

        self.osm_efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", circular=False
        )

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

        self.deformable_fusion = deformable_fusion(self.device, alpha_type)
        self.heatmap_norm = nn.LayerNorm(normalized_shape=(512,512))

        self.single_fusion = FusionModule(embed_dim=256)
        
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

    def forward(self, grd, sat, osm, heatmap=None, timestep=0):
        
        heatmap += timestep
        heatmap = self.heatmap_norm(heatmap)
        if heatmap is not None:
            osm += heatmap
            sat += heatmap

        grd_feature_volume = self.grd_efficientnet.extract_features(grd)
        grd_descriptor1 = self.grd_feature_to_descriptor1(
            grd_feature_volume
        )  # length 1280
        grd_descriptor2 = self.grd_feature_to_descriptor2(
            grd_feature_volume
        )  # length 640
        grd_descriptor3 = self.grd_feature_to_descriptor3(
            grd_feature_volume
        )  # length 320
        grd_descriptor4 = self.grd_feature_to_descriptor4(
            grd_feature_volume
        )  # length 160
        grd_descriptor5 = self.grd_feature_to_descriptor5(
            grd_feature_volume
        )  # length 80
        grd_descriptor6 = self.grd_feature_to_descriptor6(
            grd_feature_volume
        )  # length 40

        grd_descriptors = [
            grd_descriptor1,
            grd_descriptor2,
            grd_descriptor3,
            grd_descriptor4,
            grd_descriptor5,
            grd_descriptor6,
        ]

        grd_descriptor_map1 = (
            grd_descriptor1.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        )
        grd_descriptor_map2 = (
            grd_descriptor2.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)
        )
        grd_descriptor_map3 = (
            grd_descriptor3.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)
        )
        grd_descriptor_map4 = (
            grd_descriptor4.unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
        )
        grd_descriptor_map5 = (
            grd_descriptor5.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)
        )
        grd_descriptor_map6 = (
            grd_descriptor6.unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256)
        )

        grd_descriptor_maps = [
            grd_descriptor_map1,
            grd_descriptor_map2,
            grd_descriptor_map3,
            grd_descriptor_map4,
            grd_descriptor_map5,
            grd_descriptor_map6,
        ]

        sat_feature_volume, multiscale_sat = (
            self.sat_efficientnet.extract_features_multiscale(sat)
        )
        sat_feature_block0 = multiscale_sat[0]  # [16, 256, 256]
        sat_feature_block2 = multiscale_sat[2]  # [24, 128, 128]
        sat_feature_block4 = multiscale_sat[4]  # [40, 64, 64]
        sat_feature_block10 = multiscale_sat[10]  # [112, 32, 32]
        sat_feature_block15 = multiscale_sat[15]  # [320, 16, 16]

        sat_features = [
            # multiscale_sat[0],  # [batch, C, H, W]
            multiscale_sat[2],
            multiscale_sat[4],
            multiscale_sat[10],
            # multiscale_sat[15],
            sat_feature_volume
        ]

        osm_feature_volume, multiscale_osm = (
            self.osm_efficientnet.extract_features_multiscale(osm)
        )

        osm_feature_block0 = multiscale_osm[0]  # [16, 256, 256]
        osm_feature_block2 = multiscale_osm[2]  # [24, 128, 128]
        osm_feature_block4 = multiscale_osm[4]  # [40, 64, 64]
        osm_feature_block10 = multiscale_osm[10]  # [112, 32, 32]
        osm_feature_block15 = multiscale_osm[15]  # [320, 16, 16]

        osm_features = [
            # multiscale_osm[0],
            multiscale_osm[2],
            multiscale_osm[4],
            multiscale_osm[10],
            # multiscale_osm[15],
            osm_feature_volume
        ]

        batch_size = sat.size(0)  # Get batch size dynamically

        (
            fuse_feature_block0,
            fuse_feature_block2,
            fuse_feature_block4,
            fuse_feature_block10,
            fuse_feature_block15,
            fuse_feature_volume,
            fused_image,
            alpha
        ) = self.deformable_fusion(sat_features, osm_features, 
                                   osm_feature_block0, sat_feature_block0, 
                                   batch_size, heatmap)

        fuse_feature_blocks = [
            (sat_feature_block15, osm_feature_block15),
            (sat_feature_block10, osm_feature_block10),
            (sat_feature_block4, osm_feature_block4),
            (sat_feature_block2, osm_feature_block2),
            (sat_feature_block0, osm_feature_block0),
            None,
        ]  # None is an edge case, in the last step we don't concat

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
                        self.fuse_feature_to_descriptors(fuse_chunk)
                        .unsqueeze(2)
                        .unsqueeze(3)
                    )
                else:
                    fuse_descriptor_row = torch.cat(
                        (
                            fuse_descriptor_row,
                            self.fuse_feature_to_descriptors(fuse_chunk)
                            .unsqueeze(2)
                            .unsqueeze(3),
                        ),
                        3,
                    )
            if i == 0:
                fuse_descriptor_map = fuse_descriptor_row
            else:
                fuse_descriptor_map = torch.cat(
                    (fuse_descriptor_map, fuse_descriptor_row), 2
                )

        x = fuse_descriptor_map
        matching_score_stacked_list = []

        fuse_features = []
        alphas = [alpha]
        # Perform localization decoder for each level (The whole localization decoder)
        for level, (
            grd_descriptor,
            grd_descriptor_map,
            fuse_feature_block,
        ) in enumerate(zip(grd_descriptors, grd_descriptor_maps, fuse_feature_blocks)):
            if fuse_feature_block is not None:
                x, matching_score_stacked, da_output, a = self.localization_decoder(
                    level, x, grd_descriptor, grd_descriptor_map, fuse_feature_block
                )
                alphas.append(a)
            else:
                x, matching_score_stacked, da_output = self.localization_decoder(
                    level, x, grd_descriptor, grd_descriptor_map, fuse_feature_block
                )
            fuse_features.append(da_output)
            matching_score_stacked_list.append(matching_score_stacked)
            alphas.append(a)

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


        return (alphas, fused_image, logits_flattened, heatmap, x_ori) + tuple(matching_score_stacked_list)

    def compute_matching_score(
        self, shift, x, grd_des_len, grd_descriptor_map, grd_map_norm
    ):
        """
        LMU component: rolling and matching part
        TODO: use it once instead
        """
        for i in range(20):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i * shift, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:, :grd_des_len, :, :]
            sat_map_norm = torch.norm(
                sat_descriptor_map_window, p="fro", dim=1, keepdim=True
            )

            matching_score = torch.sum(
                (grd_descriptor_map * sat_descriptor_map_window), dim=1, keepdim=True
            ) / (
                sat_map_norm * grd_map_norm
            )  # cosine similarity
            if i == 0:
                matching_score_stacked = matching_score
            else:
                matching_score_stacked = torch.cat(
                    [matching_score_stacked, matching_score], dim=1
                )
        matching_score_max, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)

        return matching_score_max, matching_score_stacked

    def localization_decoder(
        self, level, x, grd_descriptor, grd_descriptor_map, fuse_feature_block
    ):
        grd_des_len = grd_descriptor.size()[1]
        fuse_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map, p="fro", dim=1, keepdim=True)

        shift = int(64 / 2**level)
        matching_score_max, matching_score_stacked = self.compute_matching_score(
            shift, x, grd_des_len, grd_descriptor_map, grd_map_norm
        )

        # loc
        x = torch.cat([matching_score_max, self.fuse_normalization(x)], dim=1)

        x = self.deconvs[level](x) # 1024, 16, 16
        da_output = None
        if fuse_feature_block is not None:

            da_output, a = self.single_fusion(x, fuse_feature_block[0], fuse_feature_block[1], level)
            x = torch.cat([x, da_output], dim=1)

        x = self.convs[level](x)

        if fuse_feature_block is not None:
            return x, matching_score_stacked, da_output, a
        else:
            return x, matching_score_stacked, da_output

    def orientation_decoder(self, level, x_ori, fuse_feature_block):
        x_ori = self.deconvs_ori[level](x_ori)
        if fuse_feature_block is not None:
            x_ori = torch.cat([x_ori, fuse_feature_block], dim=1)
        x_ori = self.convs_ori[level](x_ori)
        return x_ori

class FusionModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.ca_osm = CrossAttention(embed_dim)
        self.ca_sat = CrossAttention(embed_dim)
        self.adaptive_alpha = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # always 256 embed
        self.osm_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=320, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=112, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=40, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=24, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=16, out_channels=embed_dim, kernel_size=1),
            ]
        )

        self.sat_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=320, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=112, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=40, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=24, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=16, out_channels=embed_dim, kernel_size=1),
            ]
        )

        self.upconvs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=embed_dim, out_channels=320, kernel_size=1),
                nn.Conv2d(in_channels=embed_dim, out_channels=112, kernel_size=1),
                nn.Conv2d(in_channels=embed_dim, out_channels=40, kernel_size=1),
                nn.Conv2d(in_channels=embed_dim, out_channels=24, kernel_size=1),
                nn.Conv2d(in_channels=embed_dim, out_channels=16, kernel_size=1),
            ]
        )
        
        self.ground_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1024, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=320, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=160, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=80, out_channels=embed_dim, kernel_size=1),
                nn.Conv2d(in_channels=40, out_channels=embed_dim, kernel_size=1),
            ]
        )
        
    def forward(self, ground, sat, osm, index):
        B, C, H, W = sat.shape

        ground = self.ground_convs[index](ground)
        sat = self.sat_convs[index](sat)
        osm = self.sat_convs[index](osm)

        ground_flattened = ground.flatten(start_dim=2).permute(0, 2, 1)

        a = self.adaptive_alpha(ground_flattened).mean(dim=1).unsqueeze(1).unsqueeze(1)
        sat_output = self.ca_sat(ground, sat)
        osm_output = self.ca_osm(ground, osm)

        fused_output = a * sat_output + (1 - a) *osm_output

        return self.upconvs[index](fused_output), a

    def get_reference_points(self, H, W):
        grd_row_self_ = np.linspace(0, 1, W) 
        grd_col_self_ = np.linspace(0, 1, H) 
        grd_row_self, grd_col_self = np.meshgrid(grd_row_self_, grd_col_self_, indexing='ij') 
        reference_points = torch.stack((torch.tensor(grd_col_self), torch.tensor(grd_row_self)), -1).view(-1,2).unsqueeze(1).to(torch.float)   
        return reference_points


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.level_start_index = torch.tensor([0], device=device)
        self.attention = MultiScaleDeformableAttention(embed_dims=embed_dim, num_levels=1, 
                                                           batch_first=True)

        self.pe = PositionEncodingSine(embed_dim)
        self.embed_dim = embed_dim
        
    def forward(self, query, value):
        B, C, H, W = value.shape
        device = value.device
        query += self.pe(query)
        query_flattened = query.flatten(start_dim=2).permute(0, 2, 1)
        value_flattened = value.flatten(start_dim=2).permute(0, 2, 1)
        
        reference_points = self.get_reference_points(H, W, device).unsqueeze(0).repeat(query.shape[0], 1, 1, 1)
        spatial_shapes = torch.tensor(([[H, W]]), device=device).long()
        output = self.attention(query=query_flattened, value=value_flattened, reference_points=reference_points, 
                                            spatial_shapes=spatial_shapes, level_start_index=self.level_start_index)

        output = output.permute(0,2,1).view(query.shape[0], self.embed_dim, H, W)
        return output + query

    def get_reference_points(self, H, W, device):
        grd_row_self_ = np.linspace(0, 1, W) 
        grd_col_self_ = np.linspace(0, 1, H) 
        grd_row_self, grd_col_self = np.meshgrid(grd_row_self_, grd_col_self_, indexing='ij') 
        reference_points = torch.stack((torch.tensor(grd_col_self, device=device), torch.tensor(grd_row_self, device=device)), -1).view(-1,2).unsqueeze(1).to(torch.float)   
        return reference_points

