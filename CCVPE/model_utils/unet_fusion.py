import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.ops import MultiScaleDeformableAttention
from model_utils.position_encoding import PositionEmbeddingLearned
from model_utils.pyramid_ms import PyramidMs


class unet_fusion(nn.Module):

    def __init__(self, device, d_model=128):
        super(unet_fusion, self).__init__()
        
        self.device = device
        self.attention_1 = nn.Sequential(
            nn.BatchNorm2d(640),
            nn.ReLU(),
            nn.Conv2d(640, 320, kernel_size=1),
            nn.Sigmoid()
        )

        self.attention_2 = nn.Sequential(
            nn.BatchNorm2d(112*2),
            nn.ReLU(),
            nn.Conv2d(112*2, 112, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_3 = nn.Sequential(
            nn.BatchNorm2d(40*2),
            nn.ReLU(),
            nn.Conv2d(40*2, 40, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_4 = nn.Sequential(
            nn.BatchNorm2d(24*2),
            nn.ReLU(),
            nn.Conv2d(24*2, 24, kernel_size=1),
            nn.Sigmoid()
        )
        self.attention_5 = nn.Sequential(
            nn.BatchNorm2d(16*2),
            nn.ReLU(),
            nn.Conv2d(16*2, 16, kernel_size=1),
            nn.Sigmoid()
        )

        self.decoder1 = decoderBlock(32, 32, 112, 320, 112) #h, w, c_cur, c_prev, c_cur
        self.decoder2 = decoderBlock(64, 64, 40, 112, 40)
        self.decoder3 = decoderBlock(128, 128, 24, 40, 24)
        self.decoder4 = decoderBlock(256, 256, 16, 24, 16)

        self.volume_encoder = nn.Conv2d(320, 1280, kernel_size=1)

    def forward(self, osm_features, sat_features, batch_size):

        attention_map_1 = self.attention_1(torch.concat((osm_features[4], sat_features[4]), dim=1))
        fuse_feature_block15 = attention_map_1*osm_features[4] + (1 - attention_map_1)*sat_features[4]

        attention_map_2 = self.attention_2(torch.concat((osm_features[3], sat_features[3]), dim=1))
        fuse_feature_block10 = attention_map_2*osm_features[3] + (1 - attention_map_2)*sat_features[3]

        fuse_feature_block10 = self.decoder1(fuse_feature_block15, fuse_feature_block10)

        attention_map_3 = self.attention_3(torch.concat((osm_features[2], sat_features[2]), dim=1))
        fuse_feature_block4 = attention_map_3*osm_features[2] + (1 - attention_map_3)*sat_features[2]
        fuse_feature_block4 = self.decoder2(fuse_feature_block10, fuse_feature_block4)

        attention_map_4 = self.attention_4(torch.concat((osm_features[1], sat_features[1]), dim=1))
        fuse_feature_block2 = attention_map_4*osm_features[1] + (1 - attention_map_4)*sat_features[1]
        fuse_feature_block2 = self.decoder3(fuse_feature_block4, fuse_feature_block2)

        attention_map_5 = self.attention_5(torch.concat((osm_features[0], sat_features[0]), dim=1))
        fuse_feature_block0 = attention_map_5*osm_features[0] + (1 - attention_map_5)*sat_features[0]
        fuse_feature_block0 = self.decoder4(fuse_feature_block2, fuse_feature_block0)

        fuse_feature_volume = self.volume_encoder(fuse_feature_block15)
        
        return (
            fuse_feature_block0,
            fuse_feature_block2,
            fuse_feature_block4,
            fuse_feature_block10,
            fuse_feature_block15,
            fuse_feature_volume)


    def add_skip(self, fused_features, other_features):
        out = []

        for fused_feature, other_feature in zip(fused_features, other_features):
            out.append(fused_feature + other_feature)
        return out

class decoderBlock(nn.Module):
    def __init__(self, h, w, input_channels, skip_channels, output_channels):
        super(decoderBlock, self).__init__()
        self.h = h
        self.w = w
        self.upsample = nn.Upsample(size=(h, w), mode="bilinear")
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(input_channels + skip_channels),
            nn.ReLU(),
            nn.Conv2d(input_channels + skip_channels, output_channels, kernel_size=3, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        )

        self.residual_conv = nn.Conv2d(input_channels + skip_channels, output_channels, kernel_size=1)

    def forward(self, f1, f2):
        # f1 h/2 w/2
        f1 = self.upsample(f1)
        f3_concat = torch.cat((f1, f2), dim=1)
        f3 = self.conv1(f3_concat)
        f3 = self.conv2(f3)

        return f3 + self.residual_conv(f3_concat)


class normalization(nn.Module):
    def __init__(self, p, dim):
        super(normalization, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)

class deformable_cross_attention(nn.Module):
    def __init__(self, device, query_dim=256, dropout=0.1):
        super(deformable_cross_attention, self).__init__()

        self.device = device

        self.query_dim = query_dim

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

        self.pe_layer = PositionEmbeddingLearned(self.device, self.query_dim, self.embed_dims)

        hidden_dim = 512
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.embed_dims, hidden_dim)
        self.activation1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, self.embed_dims)
        self.norm = nn.LayerNorm(self.embed_dims)

        self.pyramid_ms = PyramidMs(self.embed_dims, self.query_dim)

        self.learnable_alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def forward(self, sat_features, osm_features, batch_size):

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

        # positional embedings
        pos = self.pe_layer(batch_size)
        pos = pos.view(batch_size, self.num_query, self.embed_dims)

        learnable_Q = self.learnable_Q.expand(batch_size, -1, -1)

        # Apply deformable attention for satellite and OSM
        sat_attention_output = self.deformable_attention_sat(
            query=learnable_Q + pos,
            key=sat_flattened,
            value=sat_flattened,
            reference_points=reference_points_sat,
            spatial_shapes=sat_spatial_shapes,
            level_start_index=sat_level_start_index,
        )  # Shape: [batch, num_queries, embed_dims]

        osm_attention_output = self.deformable_attention_osm(
            query=learnable_Q + pos,
            key=osm_flattened,
            value=osm_flattened,
            reference_points=reference_points_osm,
            spatial_shapes=osm_spatial_shapes,
            level_start_index=osm_level_start_index,
        )  # Shape: [batch, num_queries, embed_dims]

        # Reshape attention outputs to image-like format [batch, embed_dims, query_dim, query_dim]

        fused_output = torch.add(self.learnable_alpha * sat_attention_output,(1 - self.learnable_alpha) * osm_attention_output)
        fused_output = fused_output + self.norm(self.linear2(self.dropout1(self.activation1(self.linear1(fused_output)))))
        fused_output = fused_output.transpose(1, 2).view(-1, self.embed_dims, self.query_dim, self.query_dim)

        return self.pyramid_ms(fused_output)

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


