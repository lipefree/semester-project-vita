import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import ImageFile
from mmcv.ops import MultiScaleDeformableAttention
from efficientnet_pytorch.model import EfficientNet
import os
from models import CVM_VIGOR as CVM
from model_utils.light_deformable_fusion import (
    deformable_fusion,
    deformable_cross_attention,
)
from model_utils.position_encoding import PositionEmbeddingLearned#, PositionEmbeddingSine


ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)


class CVM_VIGOR(nn.Module):
    def __init__(self, device, circular_padding, use_adapt, use_concat, use_mlp=False):
        super(CVM_VIGOR, self).__init__()
        self.device = device
        self.circular_padding = circular_padding

        self.grd_efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", self.circular_padding
        )

        self.sat_efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", circular=False
        )

        self.sat_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, 1280)
        )

        self.osm_efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", circular=False
        )

        self.osm_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, 1280)
        )

        self.fuse_feature_to_descriptors = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(1280 * 2 * 2, 1280)
        )
        
        self.query_dim = 128
        self.num_query = self.query_dim**2
        self.embed_dims = 256

        self.deformable_fusion = deformable_fusion(self.device, use_pyramid=False)

        self.learnable_Q = nn.Embedding(self.num_query, self.embed_dims)

        # self.pe_layer = PositionEmbeddingSine(self.embed_dims // 2, normalize=True)

        self.cross_da1 = deformable_cross_attention(self.device, query_dim=128)

        input_dim = 256
        # hidden_dim = 640
        hidden_dim = 128
        self.global_pool = nn.AdaptiveAvgPool1d((1))  # Reduce to (B, 256, 1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, grd, sat, osm):

        grd_feature_volume, multiscale_grd = (
            self.grd_efficientnet.extract_features_multiscale(grd)
        )

        grd_features = [
            multiscale_grd[2],
            multiscale_grd[4],
            multiscale_grd[10],
            # multiscale_grd[15],
            grd_feature_volume,
        ]

        sat_feature_volume, multiscale_sat = (
            self.sat_efficientnet.extract_features_multiscale(sat)
        )

        sat_features = [
            # multiscale_sat[0],  # [batch, C, H, W]
            multiscale_sat[2],
            multiscale_sat[4],
            multiscale_sat[10],
            # multiscale_sat[15],
            sat_feature_volume,
        ]

        osm_feature_volume, multiscale_osm = (
            self.osm_efficientnet.extract_features_multiscale(osm)
        )

        osm_features = [
            # multiscale_osm[0], # 16, 256, 256
            multiscale_osm[2],  # 25, 128, 128
            multiscale_osm[4],  # 40, 64, 64
            multiscale_osm[10],  # 112, 32, 32
            # multiscale_osm[15],  # 320, 16, 16
            osm_feature_volume, # 1280, 16, 16
        ]

        batch_size = sat.size(0)  # Get batch size dynamically
        learnable_Q = self.learnable_Q.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # pos = self.pe_layer(learnable_Q)

        # print('pos size ', pos.size())
        # pos = pos.view(batch_size, self.num_query, self.embed_dims)

        fused_output  = self.cross_da1(learnable_Q, sat_features, osm_features, batch_size)

        # Classification
        fused_output_perm = fused_output.permute(0, 2, 1)

        fused_output_perm = self.deformable_fusion(
            osm_features, sat_features, batch_size,
            # grd_features
        )

        avg_pooling = self.global_pool(fused_output_perm)

        # Flatten to (B, 320)
        avg_pooling = avg_pooling.view(avg_pooling.size(0), -1)
        chosen = self.mlp(avg_pooling)

        return chosen
