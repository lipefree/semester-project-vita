import torch.nn as nn
from dual_datasets import DatasetType


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
    )


class CCVPEModules(nn.Module):
    """
    provides the localization and orientation convolutions and deconvolutions
    for CCVPE
    """

    def __init__(self, dataset_type: DatasetType):
        super().__init__()
        match dataset_type:
            case DatasetType.VIGOR:
                deconv6 = nn.ConvTranspose2d(1281, 1024, 2, 2)
                conv6 = double_conv(1344, 640)

                deconv5 = nn.ConvTranspose2d(641, 320, 2, 2)
                conv5 = double_conv(432, 320)

                deconv4 = nn.ConvTranspose2d(321, 160, 2, 2)
                conv4 = double_conv(200, 160)

                deconv3 = nn.ConvTranspose2d(161, 80, 2, 2)
                conv3 = double_conv(104, 80)

                deconv2 = nn.ConvTranspose2d(81, 40, 2, 2)
                conv2 = double_conv(56, 40)

                deconv1 = nn.ConvTranspose2d(41, 16, 2, 2)
                conv1 = nn.Sequential(
                    nn.Conv2d(16, 16, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 1, 3, stride=1, padding=1),
                )

                # ori
                deconv6_ori = nn.ConvTranspose2d(1300, 1024, 2, 2)
                conv6_ori = double_conv(1344, 640)

                deconv5_ori = nn.ConvTranspose2d(640, 256, 2, 2)
                conv5_ori = double_conv(368, 256)

                deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
                conv4_ori = double_conv(168, 128)

                deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
                conv3_ori = double_conv(88, 64)

                deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
                conv2_ori = double_conv(48, 32)

                deconv1_ori = nn.ConvTranspose2d(32, 16, 2, 2)
                conv1_ori = nn.Sequential(
                    nn.Conv2d(16, 16, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 2, 3, stride=1, padding=1),
                )

            case DatasetType.KITTI:
                # loc
                deconv6 = nn.ConvTranspose2d(2048 + 1, 1024, 2, 2)
                conv6 = double_conv(1344, 512)

                deconv5 = nn.ConvTranspose2d(512 + 1, 256, 2, 2)
                conv5 = double_conv(368, 256)

                deconv4 = nn.ConvTranspose2d(256 + 1, 128, 2, 2)
                conv4 = double_conv(168, 128)

                deconv3 = nn.ConvTranspose2d(128 + 1, 64, 2, 2)
                conv3 = double_conv(88, 128)

                deconv2 = nn.ConvTranspose2d(128 + 1, 32, 2, 2)
                conv2 = double_conv(48, 32)

                deconv1 = nn.ConvTranspose2d(32 + 1, 16, 2, 2)
                conv1 = nn.Sequential(
                    nn.Conv2d(16, 16, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 1, 3, stride=1, padding=1),
                )

                # ori
                deconv6_ori = nn.ConvTranspose2d(2048 + 16, 1024, 2, 2)
                conv6_ori = double_conv(1344, 512)

                deconv5_ori = nn.ConvTranspose2d(512, 256, 2, 2)
                conv5_ori = double_conv(368, 256)

                deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
                conv4_ori = double_conv(168, 128)

                deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
                conv3_ori = double_conv(88, 64)

                deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
                conv2_ori = double_conv(48, 32)

                deconv1_ori = nn.ConvTranspose2d(32, 16, 2, 2)
                conv1_ori = nn.Sequential(
                    nn.Conv2d(16, 16, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 2, 3, stride=1, padding=1),
                )
            case _:
                raise Exception("dataset_type must be of one of DatasetType")

        self.convs = nn.ModuleList([conv6, conv5, conv4, conv3, conv2, conv1])
        self.deconvs = nn.ModuleList(
            [
                deconv6,
                deconv5,
                deconv4,
                deconv3,
                deconv2,
                deconv1,
            ]
        )

        self.convs_ori = nn.ModuleList(
            [
                conv6_ori,
                conv5_ori,
                conv4_ori,
                conv3_ori,
                conv2_ori,
                conv1_ori,
            ]
        )
        self.deconvs_ori = nn.ModuleList(
            [
                deconv6_ori,
                deconv5_ori,
                deconv4_ori,
                deconv3_ori,
                deconv2_ori,
                deconv1_ori,
            ]
        )

    def get_convs(self):
        return self.convs

    def get_deconvs(self):
        return self.deconvs

    def get_convs_ori(self):
        return self.convs_ori

    def get_deconvs_ori(self):
        return self.deconvs_ori
