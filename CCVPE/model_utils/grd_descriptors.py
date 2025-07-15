import torch
import torch.nn as nn
from efficientnet_pytorch.model import EfficientNet
from dual_datasets import DatasetType


class permute_channels(nn.Module):
    def __init__(self, B, C, H, W):
        super().__init__()
        self.B = B
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return torch.permute(x, (self.B, self.C, self.H, self.W))


class GroundDescriptors(nn.Module):
    def __init__(self, dataset_type: DatasetType, circular_padding):
        super().__init__()
        self.grd_efficientnet = EfficientNet.from_pretrained("efficientnet-b0", circular_padding)

        dimensions: list[int]
        proj_dim: int
        match dataset_type:
            case DatasetType.VIGOR:
                dimensions = [64, 32, 16, 8, 4, 2]
                proj_dim = 10
            case DatasetType.KITTI:
                dimensions = [16, 8, 4, 2, 1, 1]
                proj_dim = 8
            case _:
                raise Exception("dataset_type must be of one of DatasetType")

        self.grd_feature_to_descriptor1 = nn.Sequential(
            nn.Conv2d(1280, dimensions[0], 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(proj_dim, 1, 1),
            nn.Flatten(start_dim=1),
        )

        self.grd_feature_to_descriptor2 = nn.Sequential(
            nn.Conv2d(1280, dimensions[1], 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(proj_dim, 1, 1),
            nn.Flatten(start_dim=1),
        )

        self.grd_feature_to_descriptor3 = nn.Sequential(
            nn.Conv2d(1280, dimensions[2], 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(proj_dim, 1, 1),
            nn.Flatten(start_dim=1),
        )
        self.grd_feature_to_descriptor4 = nn.Sequential(
            nn.Conv2d(1280, dimensions[3], 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(proj_dim, 1, 1),
            nn.Flatten(start_dim=1),
        )
        self.grd_feature_to_descriptor5 = nn.Sequential(
            nn.Conv2d(1280, dimensions[4], 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(proj_dim, 1, 1),
            nn.Flatten(start_dim=1),
        )
        self.grd_feature_to_descriptor6 = nn.Sequential(
            nn.Conv2d(1280, dimensions[5], 1),
            permute_channels(0, 2, 3, 1),
            nn.Conv2d(proj_dim, 1, 1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, grd):
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

        return grd_descriptors, grd_descriptor_maps, grd_feature_volume
