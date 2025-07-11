import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import ImageFile
from mmcv.ops import MultiScaleDeformableAttention
from efficientnet_pytorch.model import EfficientNet
from model_utils.fused_image_deformable_fusion_v2 import deformable_fusion
from model_utils.ccvpe import CCVPEDecoder
from model_utils.ground_encoder import GroundEncoder
from model_utils.position_encoding import PositionEmbeddingLearned
from model_utils.light_pyramid_ms import PyramidMs

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
        self, device, circular_padding, use_adapt, use_concat, use_mlp=False, alpha_type=0
    ):
        super(CVM_VIGOR, self).__init__()
        self.device = device
        self.circular_padding = circular_padding
        self.use_adapt = use_adapt  # If using osm tiles with 50 layers
        self.use_concat = use_concat  # If using simple fusion with concat
        self.use_mlp = use_mlp

        self.sat_efficientnet = EfficientNet.from_pretrained("efficientnet-b0", circular=False)

        self.osm_efficientnet = EfficientNet.from_pretrained("efficientnet-b0", circular=False)

        self.deformable_fusion = deformable_fusion(self.device, alpha_type)
        self.heatmap_norm = nn.LayerNorm(normalized_shape=(512, 512))

        self.ccvpe = CCVPEDecoder(self.device)

        self.ccvpe_osm = CCVPEDecoder(self.device)
        self.ccvpe_sat = CCVPEDecoder(self.device)

        self.ground_encoder = GroundEncoder()
        self.cross_da1 = deformable_cross_attention(
            self.device, query_dim=128, alpha_type=alpha_type
        )

    def forward(self, grd, sat, osm, heatmap=None, timestep=0):
        heatmap += timestep
        heatmap = self.heatmap_norm(heatmap)
        if heatmap is not None:
            osm += heatmap
            sat += heatmap

        grd_descriptors, grd_descriptor_maps = self.ground_encoder(grd)

        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat)
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
            sat_feature_volume,
        ]

        osm_feature_volume, multiscale_osm = self.osm_efficientnet.extract_features_multiscale(osm)

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
            osm_feature_volume,
        ]

        batch_size = sat.size(0)  # Get batch size dynamically

        (
            fuse_feature_block0,
            fuse_feature_block2,
            fuse_feature_block4,
            fuse_feature_block10,
            fuse_feature_block15,
            fuse_feature_volume,
            sat_feature_block0,
            sat_feature_block2,
            sat_feature_block4,
            sat_feature_block10,
            sat_feature_block15,
            sat_feature_volume,
            osm_feature_block0,
            osm_feature_block2,
            osm_feature_block4,
            osm_feature_block10,
            osm_feature_block15,
            osm_feature_volume,
            fused_image,
            alpha,
        ) = self.deformable_fusion(
            sat_features, osm_features, osm_feature_block0, sat_feature_block0, batch_size, heatmap
        )

        fuse_feature_blocks = [
            fuse_feature_block15,
            fuse_feature_block10,
            fuse_feature_block4,
            fuse_feature_block2,
            fuse_feature_block0,
            None,
        ]  # None is an edge case, in the last step we don't concat
        sat_feature_blocks = [
            sat_feature_block15,
            sat_feature_block10,
            sat_feature_block4,
            sat_feature_block2,
            sat_feature_block0,
            None,
        ]  # None is an edge case, in the last step we don't concat
        osm_feature_blocks = [
            osm_feature_block15,
            osm_feature_block10,
            osm_feature_block4,
            osm_feature_block2,
            osm_feature_block0,
            None,
        ]  # None is an edge case, in the last step we don't concat

        ccvpe_output = self.ccvpe(
            grd_descriptor_maps, grd_descriptors, fuse_feature_blocks, fuse_feature_volume
        )
        ccvpe_output_sat = self.ccvpe_sat(
            grd_descriptor_maps, grd_descriptors, sat_feature_blocks, sat_feature_volume
        )
        ccvpe_output_osm = self.ccvpe_osm(
            grd_descriptor_maps, grd_descriptors, osm_feature_blocks, osm_feature_volume
        )

        return alpha, fused_image, ccvpe_output, ccvpe_output_sat, ccvpe_output_osm


class deformable_fusion(nn.Module):
    def __init__(self, device, d_model=128, query_dim=128, use_pyramid=True, alpha_type=0):
        super().__init__()

        self.device = device
        self.query_dim = query_dim
        self.num_query = self.query_dim**2
        self.embed_dims = 256
        self.use_pyramid = use_pyramid
        self.learnable_Q = nn.Embedding(self.num_query, self.embed_dims)

        self.pe_layer = PositionEmbeddingLearned(self.device, self.query_dim, self.embed_dims)

        self.cross_da1 = deformable_cross_attention(
            self.device, query_dim=128, alpha_type=alpha_type
        )

        self.pyramid1 = PyramidMs(embed_dims=self.embed_dims, query_dim=self.query_dim)
        self.output_decoder = output_decoder()
        self.output_decoder_osm = output_decoder()
        self.output_decoder_sat = output_decoder()
        self.match_input = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear"),
            nn.Conv2d(in_channels=self.embed_dims, out_channels=16, kernel_size=1),
        )

        # self.pull_and_select = nn.Conv2d(in_channels=16*3, out_channels=16, kernel_size=1)

        self.to_image = nn.Sequential(
            nn.Upsample(size=(512, 512), mode="bilinear"),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1),
        )
        self.to_osm = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(512, 512), mode="bilinear"),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1),
        )
        self.to_sat = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(512, 512), mode="bilinear"),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1),
        )

    def forward(
        self,
        osm_features,
        sat_features,
        osm_0,
        sat_0,
        batch_size,
        grd=None,
        alpha_hint=None,
        osm_sat_separated=False,
        heatmap=None,
    ):
        pos = self.pe_layer(batch_size)
        pos = pos.view(batch_size, self.num_query, self.embed_dims)

        learnable_Q = self.learnable_Q.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        if heatmap is not None:
            learnable_Q += heatmap

        fused_output, alpha, sat_output, osm_output = self.cross_da1(
            learnable_Q + pos,
            sat_features,
            osm_features,
            batch_size,
            alpha_hint=alpha_hint,
        )

        fused_output = fused_output.transpose(1, 2).view(
            -1, self.embed_dims, self.query_dim, self.query_dim
        )

        p128, p64, p32, p16 = self.pyramid1(fused_output)

        # Adding a skip connection
        # fused_output = self.match_input(fused_output) + (1 - alpha)*osm_0 + alpha*sat_0
        fused_output = self.match_input(fused_output)

        return (
            *self.output_decoder(p128, p64, p32, p16),
            *self.output(sat_output, self.output_decoder_sat),
            *self.output(osm_output, self.output_decoder_osm),
            self.to_image(fused_output + osm_0 + sat_0),
            alpha,
        )

    def output(self, output, decoder):
        output = output.transpose(1, 2).view(-1, self.embed_dims, self.query_dim, self.query_dim)

        p128, p64, p32, p16 = self.pyramid1(output)

        return decoder(p128, p64, p32, p16)


class output_decoder(nn.Module):
    def __init__(self, embed_dims=256):
        super(output_decoder, self).__init__()

        self.conv_volume = nn.Sequential(nn.Conv2d(embed_dims, 1280, kernel_size=1))
        self.conv16 = nn.Sequential(nn.Conv2d(embed_dims, 320, kernel_size=1))
        self.conv32 = nn.Sequential(nn.Conv2d(embed_dims, 112, kernel_size=1))
        self.conv64 = nn.Sequential(nn.Conv2d(embed_dims, 40, kernel_size=1))
        self.conv128 = nn.Sequential(nn.Conv2d(embed_dims, 24, kernel_size=1))
        self.conv256 = nn.Sequential(
            nn.Upsample(size=(256, 256), mode="bilinear"),
            nn.Conv2d(embed_dims, 16, kernel_size=1),
        )

    def forward(self, p128, p64, p32, p16):
        r0 = self.conv256(p128)
        r2 = self.conv128(p128)
        r4 = self.conv64(p64)
        r10 = self.conv32(p32)
        r15 = self.conv16(p16)
        rvolume = self.conv_volume(p16)
        return r0, r2, r4, r10, r15, rvolume


class deformable_cross_attention(nn.Module):
    def __init__(self, device, query_dim=128, dropout=0.1, alpha_type=0):
        super().__init__()

        self.device = device

        self.query_dim = query_dim

        self.num_query = self.query_dim**2

        self.embed_dims = 256
        num_levels = 4

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

        # self.pe_layer = PositionEmbeddingLearned(self.device, self.query_dim, self.embed_dims)

        self.fuse_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, 1280)
        )
        self.fuse_normalization = normalization(2, 1)

        # Make all of these in modules instead
        channels = [24, 40, 112, 1280]

        self.input_proj_list_sat = self.get_input_proj_list(channels, self.embed_dims, num_levels)

        self.input_proj_list_osm = self.get_input_proj_list(channels, self.embed_dims, num_levels)

        hidden_dim = 128
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.embed_dims, hidden_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, self.embed_dims)
        self.learnable_alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        self.adaptive_alpha = nn.Sequential(
            nn.Linear(self.embed_dims, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.alpha_type = alpha_type

        self.norm = nn.LayerNorm(self.embed_dims)
        self.norm_sat = nn.LayerNorm(self.embed_dims)
        self.norm_osm = nn.LayerNorm(self.embed_dims)

    def forward(self, Q, sat_features, osm_features, batch_size, alpha_hint=None):
        # MS deformable attention has particular inputs
        (
            sat_flattened,
            reference_points_sat,
            sat_spatial_shapes,
            sat_level_start_index,
        ) = self.prepare_input_ms_deformable_attention(sat_features, self.input_proj_list_sat)

        (
            osm_flattened,
            reference_points_osm,
            osm_spatial_shapes,
            osm_level_start_index,
        ) = self.prepare_input_ms_deformable_attention(osm_features, self.input_proj_list_osm)

        # Apply deformable attention for satellite and OSM
        sat_attention_output = self.deformable_attention_sat(
            query=Q,
            key=sat_flattened,
            value=sat_flattened,
            reference_points=reference_points_sat,
            spatial_shapes=sat_spatial_shapes,
            level_start_index=sat_level_start_index,
        )  # Shape: [batch, num_queries, embed_dims]

        osm_attention_output = self.deformable_attention_osm(
            query=Q,
            key=osm_flattened,
            value=osm_flattened,
            reference_points=reference_points_osm,
            spatial_shapes=osm_spatial_shapes,
            level_start_index=osm_level_start_index,
        )  # Shape: [batch, num_queries, embed_dims]

        if self.alpha_type == 0:
            alpha = torch.tensor([0.5]).to(self.device)
        elif self.alpha_type == 1:
            alpha = torch.sigmoid(self.learnable_alpha)
        else:
            alpha = self.adaptive_alpha(
                torch.cat((sat_attention_output, osm_attention_output), dim=1)
            )
            alpha = alpha.mean(dim=1).unsqueeze(1)

        fused_output = torch.add(alpha * sat_attention_output, (1 - alpha) * osm_attention_output)

        fused_output = self.norm(fused_output + Q)
        sat_attention_output = self.norm_sat(sat_attention_output + Q)
        osm_attention_output = self.norm_osm(osm_attention_output + Q)

        return fused_output, alpha.unsqueeze(-1), sat_attention_output, osm_attention_output

    def add_skip(self, fused_features, other_features):
        out = []
        for fused_feature, other_feature in zip(fused_features, other_features):
            out.append(fused_feature + other_feature)
        return out

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
