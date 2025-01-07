import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import ImageFile
from mmcv.ops import MultiScaleDeformableAttention
from efficientnet_pytorch.model import EfficientNet

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
    def __init__(self, device, circular_padding, use_adapt, use_concat, use_mlp=False):
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

        self.query_dim = 256
        self.num_query = self.query_dim**2

        self.embed_dims = 128
        num_levels = 5
        self.deformable_attention_sat = MultiScaleDeformableAttention(
            embed_dims=self.embed_dims,
            num_levels=num_levels,
            batch_first=True,
        )

        self.deformable_attention_osm = MultiScaleDeformableAttention(
            embed_dims=self.embed_dims,
            num_levels=num_levels,
            batch_first=True,
        )

        self.learnable_Q = nn.parameter.Parameter(
            nn.init.kaiming_normal_(torch.zeros(1, self.num_query, self.embed_dims))
        )

        if self.query_dim != 256:
            self.conv_block0 = nn.Sequential(
                nn.Upsample(
                    size=(256, 256), mode="bilinear"
                ),  # Needed if we want to query less points due to hardware limit
                nn.Conv2d(self.embed_dims, 16, kernel_size=1),  # [batch, 16, 256, 256]
            )
        else:
            self.conv_block0 = nn.Conv2d(self.embed_dims, 16, kernel_size=1)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                16, 24, kernel_size=3, stride=2, padding=1
            ),  # [batch, 24, 128, 128]
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(
                24, 40, kernel_size=3, stride=2, padding=1
            ),  # [batch, 40, 64, 64]
        )

        self.conv_block10 = nn.Sequential(
            nn.Conv2d(
                40, 112, kernel_size=3, stride=2, padding=1
            ),  # [batch, 112, 32, 32]
        )

        self.conv_block15 = nn.Sequential(
            nn.Conv2d(
                112, 320, kernel_size=3, stride=2, padding=1
            ),  # [batch, 320, 16, 16]
        )

        self.conv_block_volume = nn.Conv2d(320, 1280, kernel_size=1)

        self.fuse_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, 1280)
        )
        self.fuse_normalization = normalization(2, 1)

        # Make all of these in modules instead
        channels = [16, 24, 40, 112, 320]

        self.input_proj_list_sat = self.get_input_proj_list(
            channels, self.embed_dims, num_levels
        )

        self.input_proj_list_osm = self.get_input_proj_list(
            channels, self.embed_dims, num_levels
        )

        # Define the MLP for fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * self.embed_dims, self.embed_dims),  # Reduce to embed_dims
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),  # Keep dimension consistent
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

        # TODO: This is really a bad way to do it, redo it later (I hope one day)
        if self.use_adapt:
            sat = self.adapt_50_n(sat)

        if self.use_concat:
            sat = self.adapt_concat(sat)

        sat_feature_volume, multiscale_sat = (
            self.sat_efficientnet.extract_features_multiscale(sat)
        )
        sat_feature_block0 = multiscale_sat[0]  # [16, 256, 256]
        sat_feature_block2 = multiscale_sat[2]  # [24, 128, 128]
        sat_feature_block4 = multiscale_sat[4]  # [40, 64, 64]
        sat_feature_block10 = multiscale_sat[10]  # [112, 32, 32]
        sat_feature_block15 = multiscale_sat[15]  # [320, 16, 16]

        sat_features = [
            multiscale_sat[0],  # [batch, C, H, W]
            multiscale_sat[2],
            multiscale_sat[4],
            multiscale_sat[10],
            multiscale_sat[15],
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
            multiscale_osm[0],
            multiscale_osm[2],
            multiscale_osm[4],
            multiscale_osm[10],
            multiscale_osm[15],
        ]

        # MS deformable attention has particular inputs
        (
            sat_flattened,
            reference_points_sat,
            sat_spatial_shapes,
            sat_level_start_index,
        ) = self.prepare_input_ms_deformable_attention(
            sat_features, self.input_proj_list_sat
        )

        (
            osm_flattened,
            reference_points_osm,
            osm_spatial_shapes,
            osm_level_start_index,
        ) = self.prepare_input_ms_deformable_attention(
            osm_features, self.input_proj_list_osm
        )

        batch_size = sat.size(0)  # Get batch size dynamically
        learnable_Q = self.learnable_Q.expand(batch_size, -1, -1)

        # Apply deformable attention for satellite and OSM
        sat_attention_output = self.deformable_attention_sat(
            query=learnable_Q,
            key=sat_flattened,
            value=sat_flattened,
            reference_points=reference_points_sat,
            spatial_shapes=sat_spatial_shapes,
            level_start_index=sat_level_start_index,
        )  # Shape: [batch, num_queries, embed_dims]

        osm_attention_output = self.deformable_attention_osm(
            query=learnable_Q,
            key=osm_flattened,
            value=osm_flattened,
            reference_points=reference_points_osm,
            spatial_shapes=osm_spatial_shapes,
            level_start_index=osm_level_start_index,
        )  # Shape: [batch, num_queries, embed_dims]

        if self.use_mlp:
            # Concatenate the flattened attention outputs along the embedding dimension
            fused_flat = torch.cat([sat_attention_output, osm_attention_output], dim=-1)  # [batch, num_queries, 2 * embed_dims]

            # Apply the MLP to the concatenated flattened features
            fused_flat_fused = self.fusion_mlp(fused_flat)  # [batch, num_queries, embed_dims]

            # Reshape back into spatial format
            fused_output = fused_flat_fused.transpose(1, 2).view(-1, self.embed_dims, self.query_dim, self.query_dim)  # [batch, embed_dims, query_dim, query_dim]
        else:
            # Reshape attention outputs to image-like format [batch, embed_dims, query_dim, query_dim]
            sat_attention_reshaped = sat_attention_output.transpose(1, 2).view(-1, self.embed_dims, self.query_dim, self.query_dim)
            osm_attention_reshaped = osm_attention_output.transpose(1, 2).view(-1, self.embed_dims, self.query_dim, self.query_dim)

            fused_output = torch.add(sat_attention_reshaped, osm_attention_reshaped)
        
        # Now we need to create the conv net to match for multi-scale
        fuse_feature_block0 = self.conv_block0(fused_output)
        fuse_feature_block2 = self.conv_block2(fuse_feature_block0)
        fuse_feature_block4 = self.conv_block4(fuse_feature_block2)
        fuse_feature_block10 = self.conv_block10(fuse_feature_block4)
        fuse_feature_block15 = self.conv_block15(fuse_feature_block10)
        fuse_feature_volume = self.conv_block_volume(fuse_feature_block15)

        fuse_feature_blocks = [
            fuse_feature_block15,
            fuse_feature_block10,
            fuse_feature_block4,
            fuse_feature_block2,
            fuse_feature_block0,
            None,
        ]  # None is an edge case, in the last step we don't concat

        # We add the skip connections from the encoder of OSM and SAT

        fuse_feature_volume = fuse_feature_volume + osm_feature_volume

        temp_block = []

        for i, block in enumerate(fuse_feature_blocks):
            if block is not None:
                temp_block.append(block + osm_features[-i - 1])
            else:
                temp_block.append(None)

        fuse_feature_blocks = temp_block

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

        return (logits_flattened, heatmap, x_ori) + tuple(matching_score_stacked_list)

    def prepare_input_ms_deformable_attention(self, features, input_proj_list):
        """
        Follow the specification of https://mmcv.readthedocs.io/en/latest/api/generated/mmcv.ops.MultiScaleDeformableAttention.html?highlight=deformable%20attention#mmcv.ops.MultiScaleDeformableAttention
        """
        spatial_shapes = torch.tensor(
            [[f.size(2), f.size(3)] for f in features], device=self.device
        )

        level_start_index = torch.cat(
            [
                torch.tensor([0], device=self.device),
                torch.cumsum(spatial_shapes[:, 0] * spatial_shapes[:, 1], dim=0)[:-1],
            ]
        )

        features_flattened = []
        for lvl, f in enumerate(features):
            # First we have to reduce the number of channels to 256
            f = input_proj_list[lvl](f)
            f = f.flatten(2).transpose(1, 2)
            features_flattened.append(f)

        features_flattened = torch.cat(features_flattened, dim=1)

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, self.query_dim, device=self.device),  # X coordinates
            torch.linspace(0, 1, self.query_dim, device=self.device),  # Y coordinates
            indexing="xy",
        )

        # Stack and reshape to form (num_query, num_levels, 2)
        reference_points = torch.stack((grid_x, grid_y), dim=-1).view(
            1, -1, 1, 2
        )  # Shape: (1, query_dim*query_dim, 1, 2)

        # Expand to batch size and num_levels
        reference_points_sat = reference_points.expand(
            features_flattened.size(0), -1, len(features), -1
        )  # Shape: (batch_size, num_query, num_levels, 2)

        return features_flattened, reference_points, spatial_shapes, level_start_index

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

    def compute_loc(
        self,
        matching_score_max_sat,
        sat_descriptor_map,
        sat_feature_block,
        conv,
        deconv,
    ):
        """
        Localization Decoder: use output of previous LMU and prepare for next LMU
        """
        x = torch.cat(
            [matching_score_max_sat, self.sat_normalization(sat_descriptor_map)], dim=1
        )
        x = deconv(x)
        x = torch.cat([x, sat_feature_block], dim=1)
        x = conv(x)

        return x

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
