import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial


class CCVPEDecoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        # loc
        self.deconv6 = nn.ConvTranspose2d(1281, 1024, 2, 2)
        self.conv6 = self.double_conv(1344, 640)

        self.deconv5 = nn.ConvTranspose2d(641, 320, 2, 2)
        self.conv5 = self.double_conv(432, 320)

        self.deconv4 = nn.ConvTranspose2d(321, 160, 2, 2)
        self.conv4 = self.double_conv(200, 160)

        self.deconv3 = nn.ConvTranspose2d(161, 80, 2, 2)
        self.conv3 = self.double_conv(104, 80)

        self.deconv2 = nn.ConvTranspose2d(81, 40, 2, 2)
        self.conv2 = self.double_conv(56, 40)

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
        self.conv6_ori = self.double_conv(1344, 640)

        self.deconv5_ori = nn.ConvTranspose2d(640, 256, 2, 2)
        self.conv5_ori = self.double_conv(368, 256)

        self.deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4_ori = self.double_conv(168, 128)

        self.deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3_ori = self.double_conv(88, 64)

        self.deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2_ori = self.double_conv(48, 32)

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
        self.fuse_normalization = partial(F.normalize, p=2, dim=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, grd_descriptor_maps, grd_descriptors, map_features, map_volume):

        fuse_feature_blocks = map_features # it's called fuse_feature_blocks because using OSM + sat is better
        fuse_feature_volume = map_volume
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


