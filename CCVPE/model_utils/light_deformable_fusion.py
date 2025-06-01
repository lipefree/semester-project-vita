import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.ops import MultiScaleDeformableAttention
from model_utils.position_encoding import PositionEmbeddingLearned
from model_utils.light_pyramid_ms import PyramidMs


class deformable_fusion(nn.Module):

    def __init__(self, device, d_model=128, query_dim=128, use_pyramid=True):
        super().__init__()

        self.device = device
        self.query_dim = query_dim
        self.num_query = self.query_dim**2
        self.embed_dims = 256
        self.use_pyramid = use_pyramid
        self.learnable_Q = nn.Embedding(self.num_query, self.embed_dims)

        self.pe_layer = PositionEmbeddingLearned(
            self.device, self.query_dim, self.embed_dims
        )

        self.cross_da1 = deformable_cross_attention(self.device, query_dim=128)
        # self.cross_da2 = deformable_cross_attention(self.device, query_dim=128)
        # self.cross_da3 = deformable_cross_attention(self.device, query_dim=128)
        # self.cross_da4 = deformable_cross_attention(self.device, query_dim=128)

        # self.self_da1 = deformable_self_attention(self.device, self.query_dim)
        # self.self_da2 = deformable_self_attention(self.device, self.query_dim)

        # self.cross_grd = deformable_cross_grd_attention(self.device, query_dim=128)
        self.pyramid1 = PyramidMs(embed_dims=self.embed_dims, query_dim=self.query_dim)
        self.output_decoder = output_decoder()

    def forward(
        self, osm_features, sat_features, batch_size, grd=None, alpha_hint=None
    ):
        pos = self.pe_layer(batch_size)
        pos = pos.view(batch_size, self.num_query, self.embed_dims)

        learnable_Q = self.learnable_Q.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        fused_output = self.cross_da1(
            learnable_Q + pos,
            sat_features,
            osm_features,
            batch_size,
            alpha_hint=alpha_hint,
        )

        # fused_output = self.self_da2(fused_output)

        if grd is not None:
            fused_output = self.cross_grd(fused_output, grd, batch_size)

        # fused_output = self.cross_da2(fused_output, sat_features, osm_features, batch_size)
        # fused_output = self.self_da1(fused_output)

        if self.use_pyramid:
            fused_output = fused_output.transpose(1, 2).view(
                -1, self.embed_dims, self.query_dim, self.query_dim
            )
            p128, p64, p32, p16 = self.pyramid1(fused_output)

            return self.output_decoder(p128, p64, p32, p16)
        else:
            fused_output = fused_output.permute(0, 2, 1)
            return fused_output

    def add_skip(self, fused_features, other_features):
        out = []

        for fused_feature, other_feature in zip(fused_features, other_features):
            out.append(fused_feature + other_feature)
        return out


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


class normalization(nn.Module):
    def __init__(self, p, dim):
        super(normalization, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)


class deformable_cross_attention(nn.Module):
    def __init__(self, device, query_dim=128, dropout=0.1):
        super(deformable_cross_attention, self).__init__()

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

        self.input_proj_list_sat = self.get_input_proj_list(
            channels, self.embed_dims, num_levels
        )

        self.input_proj_list_osm = self.get_input_proj_list(
            channels, self.embed_dims, num_levels
        )

        hidden_dim = 512
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.embed_dims, hidden_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, self.embed_dims)
        self.norm = nn.LayerNorm(self.embed_dims)
        self.learnable_alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, Q, sat_features, osm_features, batch_size, alpha_hint=None):

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

        if alpha_hint is not None:
            alpha = torch.sigmoid(torch.tensor(alpha_hint, device=self.device))
        else:
            alpha = torch.sigmoid(self.learnable_alpha)

        fused_output = torch.add(
            alpha * sat_attention_output, (1 - alpha) * osm_attention_output
        )
        fused_output = fused_output + Q
        fused_output = fused_output + self.norm(
            self.linear2(self.dropout1(self.activation1(self.linear1(fused_output))))
        )

        return fused_output

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


class deformable_cross_grd_attention(nn.Module):
    def __init__(self, device, query_dim=128, dropout=0.1):
        super(deformable_cross_grd_attention, self).__init__()

        self.device = device

        self.query_dim = query_dim

        self.num_query = self.query_dim**2

        self.embed_dims = 256
        num_levels = 4

        self.deformable_attention_grd = MultiScaleDeformableAttention(
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
        channels = [24, 40, 112, 1280]

        self.input_proj_list_grd = self.get_input_proj_list(
            channels, self.embed_dims, num_levels
        )

        hidden_dim = 512
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.embed_dims, hidden_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, self.embed_dims)
        self.norm = nn.LayerNorm(self.embed_dims)

        self.pyramid_ms = PyramidMs(self.embed_dims, self.query_dim)

        self.learnable_alpha = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, Q, grd, batch_size):

        (
            grd_flattened,
            reference_points_grd,
            grd_spatial_shapes,
            grd_level_start_index,
        ) = self.prepare_input_ms_deformable_attention(grd, self.input_proj_list_grd)

        grd_attention_output = self.deformable_attention_grd(
            query=Q,
            key=grd_flattened,
            value=grd_flattened,
            reference_points=reference_points_grd,
            spatial_shapes=grd_spatial_shapes,
            level_start_index=grd_level_start_index,
        )  # Shape: [batch, num_queries, embed_dims]

        # Reshape attention outputs to image-like format [batch, embed_dims, query_dim, query_dim]
        fused_output = grd_attention_output + Q
        fused_output = fused_output + self.norm(
            self.linear2(self.dropout1(self.activation1(self.linear1(fused_output))))
        )

        return fused_output

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


class deformable_self_attention(nn.Module):
    def __init__(self, device, query_dim=128, dropout=0.1):
        super(deformable_self_attention, self).__init__()
        self.device = device
        self.query_dim = query_dim
        self.num_query = self.query_dim**2
        self.embed_dims = 256
        num_levels = 1

        self.deformable_attention = MultiScaleDeformableAttention(
            embed_dims=self.embed_dims,
            num_levels=num_levels,
            batch_first=True,
        )

        hidden_dim = 512
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.embed_dims, hidden_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, self.embed_dims)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, q):
        # MS deformable attention has particular inputs
        (
            q,
            reference_points_q,
            q_spatial_shapes,
            q_level_start_index,
        ) = self.prepare_input_ms_deformable_attention(q)

        q_attention_output = self.deformable_attention(
            query=q,
            key=q,
            value=q,
            reference_points=reference_points_q,
            spatial_shapes=q_spatial_shapes,
            level_start_index=q_level_start_index,
        )  # Shape: [batch, num_queries, embed_dims]

        q_attention_output = q_attention_output + q
        q_attention_output = q_attention_output + self.norm(
            self.linear2(
                self.dropout1(self.activation1(self.linear1(q_attention_output)))
            )
        )
        return q_attention_output

    def prepare_input_ms_deformable_attention(self, features):
        """
        Follow the specification of https://mmcv.readthedocs.io/en/latest/api/generated/mmcv.ops.MultiScaleDeformableAttention.html?highlight=deformable%20attention#mmcv.ops.MultiScaleDeformableAttention
        """
        spatial_shapes = torch.tensor(
            [[self.query_dim, self.query_dim]], device=self.device
        )

        level_start_index = torch.tensor([0], device=self.device)
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(0, 1, self.query_dim, device=self.device),  # X coordinates
            torch.linspace(0, 1, self.query_dim, device=self.device),  # Y coordinates
            indexing="xy",
        )

        # Stack and reshape to form (num_query, num_levels, 2)
        reference_points = torch.stack((grid_x, grid_y), dim=-1).view(
            1, -1, 1, 2
        )  # Shape: (1, query_dim*query_dim, 1, 2)

        return features, reference_points, spatial_shapes, level_start_index
