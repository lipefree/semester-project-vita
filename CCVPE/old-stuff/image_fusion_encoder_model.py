import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from PIL import ImageFile
from mmcv.ops import MultiScaleDeformableAttention
from efficientnet_pytorch.model import EfficientNet
from model_utils.fused_image_deformable_fusion_v2 import deformable_fusion

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)


class CVM_VIGOR(nn.Module):
    def __init__(self, device, image_only=True):
        super(CVM_VIGOR, self).__init__()
        self.device = device

        self.sat_efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", circular=False
        )

        self.osm_efficientnet = EfficientNet.from_pretrained(
            "efficientnet-b0", circular=False
        )

        self.deformable_fusion = deformable_fusion(self.device)

        self.image_only = image_only

    def forward(self, grd, sat, osm):

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
            alpha,
            recons_osm,
            recons_sat
        ) = self.deformable_fusion(sat_features, osm_features, osm_feature_block0, sat_feature_block0, batch_size, osm_sat_separated=True)

        if self.image_only:
            return fused_image
        else:
            return (
                fuse_feature_block0,
                fuse_feature_block2,
                fuse_feature_block4,
                fuse_feature_block10,
                fuse_feature_block15,
                fuse_feature_volume,
                fused_image,
                recons_osm,
                recons_sat
            ) 
