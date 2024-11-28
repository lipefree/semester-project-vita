from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import ImageFile
from torch.utils.tensorboard import SummaryWriter
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)
from efficientnet_pytorch.model import EfficientNet

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
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )

class CVM_VIGOR(nn.Module):
    def __init__(self, device, circular_padding, use_adapt, use_concat):
        super(CVM_VIGOR, self).__init__()
        self.device = device
        self.circular_padding = circular_padding
        self.use_adapt = use_adapt # If using osm tiles with 50 layers
        self.use_concat = use_concat # If using simple fusion with concat

        self.adapt_concat = nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.adapt_50_n = nn.Sequential(
                nn.Conv2d(in_channels=50, out_channels=3, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.grd_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', self.circular_padding)

        self.grd_feature_to_descriptor1 = nn.Sequential(
                                    nn.Conv2d(1280, 64, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor2 = nn.Sequential(
                                    nn.Conv2d(1280, 32, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor3 = nn.Sequential(
                                    nn.Conv2d(1280, 16, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor4 = nn.Sequential(
                                    nn.Conv2d(1280, 8, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor5 = nn.Sequential(
                                    nn.Conv2d(1280, 4, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.grd_feature_to_descriptor6 = nn.Sequential(
                                    nn.Conv2d(1280, 2, 1),
                                    permute_channels(0, 2, 3, 1),
                                    nn.Conv2d(10, 1, 1),
                                    nn.Flatten(start_dim=1)
                                    )
        
        self.sat_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', circular=False)
        
        
        self.sat_feature_to_descriptors = nn.Sequential(nn.Flatten(start_dim=1),
                                                        nn.Linear(1280*2*2, 1280)
                                                       )
        
        self.sat_normalization = normalization(2, 1)

        self.osm_efficientnet = EfficientNet.from_pretrained('efficientnet-b0', circular=False)
        
        
        self.osm_feature_to_descriptors = nn.Sequential(nn.Flatten(start_dim=1),
                                                        nn.Linear(1280*2*2, 1280)
                                                       )
        
        self.osm_normalization = normalization(2, 1)

        # loc
        self.deconv6 = nn.ConvTranspose2d(1281*2, 1024, 2, 2)
        self.conv6 = double_conv(1344*2, 640)
                                    
        self.deconv5 = nn.ConvTranspose2d(641*2, 320, 2, 2)
        self.conv5 = double_conv(432*2, 320)
        
        self.deconv4 = nn.ConvTranspose2d(321*2, 160, 2, 2)
        self.conv4 = double_conv(200*2, 160)
        
        self.deconv3 = nn.ConvTranspose2d(161*2, 80, 2, 2)
        self.conv3 = double_conv(104*2, 80)
        
        self.deconv2 = nn.ConvTranspose2d(81*2, 40, 2, 2)
        self.conv2 = double_conv(56*2, 40)
        
        self.deconv1 = nn.ConvTranspose2d(41*2, 16, 2, 2)
        self.conv1 = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 1, 3, stride=1, padding=1))
        
        # ori
        self.deconv6_ori = nn.ConvTranspose2d(1300*2, 1024, 2, 2)
        self.conv6_ori = double_conv(1344*2, 640)
                                    
        self.deconv5_ori = nn.ConvTranspose2d(640, 256, 2, 2)
        self.conv5_ori = double_conv(368*2, 256)
        
        self.deconv4_ori = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4_ori = double_conv(168*2, 128)
        
        self.deconv3_ori = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3_ori = double_conv(88*2, 64)
        
        self.deconv2_ori = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2_ori = double_conv(48*2, 32)
        
        self.deconv1_ori = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv1_ori = nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 2, 3, stride=1, padding=1))
        
    def forward(self, grd, sat, osm):
        grd_feature_volume = self.grd_efficientnet.extract_features(grd)
        grd_descriptor1 = self.grd_feature_to_descriptor1(grd_feature_volume) # length 1280
        grd_descriptor2 = self.grd_feature_to_descriptor2(grd_feature_volume) # length 640
        grd_descriptor3 = self.grd_feature_to_descriptor3(grd_feature_volume) # length 320
        grd_descriptor4 = self.grd_feature_to_descriptor4(grd_feature_volume) # length 160
        grd_descriptor5 = self.grd_feature_to_descriptor5(grd_feature_volume) # length 80
        grd_descriptor6 = self.grd_feature_to_descriptor6(grd_feature_volume) # length 40
        
        grd_descriptor_map1 = grd_descriptor1.unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 8)
        grd_descriptor_map2 = grd_descriptor2.unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16)
        grd_descriptor_map3 = grd_descriptor3.unsqueeze(2).unsqueeze(3).repeat(1, 1, 32, 32)
        grd_descriptor_map4 = grd_descriptor4.unsqueeze(2).unsqueeze(3).repeat(1, 1, 64, 64)
        grd_descriptor_map5 = grd_descriptor5.unsqueeze(2).unsqueeze(3).repeat(1, 1, 128, 128)
        grd_descriptor_map6 = grd_descriptor6.unsqueeze(2).unsqueeze(3).repeat(1, 1, 256, 256)

        sat_feature_volume, multiscale_sat = self.sat_efficientnet.extract_features_multiscale(sat)
        sat_feature_block0 = multiscale_sat[0] # [16, 256, 256]
        sat_feature_block2 = multiscale_sat[2] #[24, 128, 128]
        sat_feature_block4 = multiscale_sat[4] # [40, 64, 64]
        sat_feature_block10 = multiscale_sat[10] # [112, 32, 32]
        sat_feature_block15 = multiscale_sat[15] # [320, 16, 16]
        
        osm_feature_volume, multiscale_osm = self.osm_efficientnet.extract_features_multiscale(osm)
        osm_feature_block0 = multiscale_osm[0] # [16, 256, 256]
        osm_feature_block2 = multiscale_osm[2] #[24, 128, 128]
        osm_feature_block4 = multiscale_osm[4] # [40, 64, 64]
        osm_feature_block10 = multiscale_osm[10] # [112, 32, 32]
        osm_feature_block15 = multiscale_osm[15] # [320, 16, 16]

        sat_row_chunks = torch.stack(list(torch.chunk(sat_feature_volume, 8, 2)), dim=-1) # dimension 4 is the number of row chunks (splitted in height dimension)
        for i, sat_row_chunk in enumerate(torch.unbind(sat_row_chunks, dim=-1), 0):
            sat_chunks = torch.stack(list(torch.chunk(sat_row_chunk, 8, 3)), dim=-1) # dimension 5 is the number of vertical chunks (splitted in width dimension)
            for j, sat_chunk in enumerate(torch.unbind(sat_chunks, dim=-1), 0):
                if j == 0:
                    sat_descriptor_row = self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)
                else:
                    sat_descriptor_row = torch.cat((sat_descriptor_row, self.sat_feature_to_descriptors(sat_chunk).unsqueeze(2).unsqueeze(3)), 3)
            if i == 0:
                sat_descriptor_map = sat_descriptor_row
            else:
                sat_descriptor_map = torch.cat((sat_descriptor_map, sat_descriptor_row), 2)

        
        osm_row_chunks = torch.stack(list(torch.chunk(osm_feature_volume, 8, 2)), dim=-1) # dimension 4 is the number of row chunks (splitted in height dimension)
        for i, osm_row_chunk in enumerate(torch.unbind(osm_row_chunks, dim=-1), 0):
            osm_chunks = torch.stack(list(torch.chunk(osm_row_chunk, 8, 3)), dim=-1) # dimension 5 is the number of vertical chunks (splitted in width dimension)
            for j, osm_chunk in enumerate(torch.unbind(osm_chunks, dim=-1), 0):
                if j == 0:
                    osm_descriptor_row = self.osm_feature_to_descriptors(osm_chunk).unsqueeze(2).unsqueeze(3)
                else:
                    osm_descriptor_row = torch.cat((osm_descriptor_row, self.osm_feature_to_descriptors(osm_chunk).unsqueeze(2).unsqueeze(3)), 3)
            if i == 0:
                osm_descriptor_map = osm_descriptor_row
            else:
                osm_descriptor_map = torch.cat((osm_descriptor_map, osm_descriptor_row), 2)

        # matching bottleneck
        grd_des_len = grd_descriptor1.size()[1]
        sat_des_len = sat_descriptor_map.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map1, p='fro', dim=1, keepdim=True)
        
        osm_des_len = osm_descriptor_map.size()[1]

        matching_score_max_sat, matching_score_stacked_sat = self.compute_matching_score(64, sat_descriptor_map, grd_des_len, grd_descriptor_map1, grd_map_norm)
        matching_score_max_osm, matching_score_stacked_osm = self.compute_matching_score(64, osm_descriptor_map, grd_des_len, grd_descriptor_map1, grd_map_norm)
    
        # loc
        x = self.compute_loc(matching_score_max_sat,
                             matching_score_max_osm,
                             sat_descriptor_map,
                             osm_descriptor_map,
                             sat_feature_block15,
                             osm_feature_block15,
                             self.conv6,
                             self.deconv6
                         )

        # matching 16*16
        grd_des_len = grd_descriptor2.size()[1] # 640
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map2, p='fro', dim=1, keepdim=True)
        osm_des_len = x.size()[1]

        matching_score_max_sat, matching_score_stacked_sat2 = self.compute_matching_score(32, x, grd_des_len, grd_descriptor_map2, grd_map_norm)
        matching_score_max_osm, matching_score_stacked_osm2 = self.compute_matching_score(32, x, grd_des_len, grd_descriptor_map2, grd_map_norm)
        
        x = self.compute_loc(matching_score_max_sat,
                             matching_score_max_osm,
                             x, # TODO: discuss this
                             x,
                             sat_feature_block10,
                             osm_feature_block10,
                             self.conv5,
                             self.deconv5
                         )

        # matching 32*32
        grd_des_len = grd_descriptor3.size()[1] # 320
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map3, p='fro', dim=1, keepdim=True)
        
        matching_score_max_sat, matching_score_stacked_sat3 = self.compute_matching_score(16, x, grd_des_len, grd_descriptor_map3, grd_map_norm)
        matching_score_max_osm, matching_score_stacked_osm3 = self.compute_matching_score(16, x, grd_des_len, grd_descriptor_map3, grd_map_norm)
        
        x = self.compute_loc(matching_score_max_sat,
                             matching_score_max_osm,
                             x, # TODO: discuss this
                             x,
                             sat_feature_block4,
                             osm_feature_block4,
                             self.conv4,
                             self.deconv4)

        # matching 64*64
        grd_des_len = grd_descriptor4.size()[1] # 160
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map4, p='fro', dim=1, keepdim=True)

        matching_score_max_sat, matching_score_stacked_sat4 = self.compute_matching_score(8, x, grd_des_len, grd_descriptor_map4, grd_map_norm)
        matching_score_max_osm, matching_score_stacked_osm4 = self.compute_matching_score(8, x, grd_des_len, grd_descriptor_map4, grd_map_norm)
        
        x = self.compute_loc(matching_score_max_sat,
                             matching_score_max_osm,
                             x, # TODO: discuss this
                             x,
                             sat_feature_block2,
                             osm_feature_block2,
                             self.conv3,
                             self.deconv3
                         )

        # matching 128*128
        grd_des_len = grd_descriptor5.size()[1] # 80
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map5, p='fro', dim=1, keepdim=True)

        matching_score_max_sat, matching_score_stacked_sat5 = self.compute_matching_score(4, x, grd_des_len, grd_descriptor_map5, grd_map_norm)
        matching_score_max_osm, matching_score_stacked_osm5 = self.compute_matching_score(4, x, grd_des_len, grd_descriptor_map5, grd_map_norm)
        
        x = self.compute_loc(matching_score_max_sat,
                             matching_score_max_osm,
                             x, # TODO: discuss this
                             x,
                             sat_feature_block0,
                             osm_feature_block0,
                             self.conv2,
                             self.deconv2
                         )


        # matching 256*256
        grd_des_len = grd_descriptor6.size()[1] # 40
        sat_des_len = x.size()[1]
        grd_map_norm = torch.norm(grd_descriptor_map6, p='fro', dim=1, keepdim=True)

        matching_score_max_sat, matching_score_stacked_sat6 = self.compute_matching_score(2, x, grd_des_len, grd_descriptor_map6, grd_map_norm)
        matching_score_max_osm, matching_score_stacked_osm6 = self.compute_matching_score(2, x, grd_des_len, grd_descriptor_map6, grd_map_norm)
        
        x = torch.cat([matching_score_max_sat, matching_score_max_osm, self.sat_normalization(x), self.osm_normalization(x)], dim=1)
        x = self.deconv1(x)
        x = self.conv1(x)
        
        logits_flattened = torch.flatten(x, start_dim=1)
        heatmap = torch.reshape(nn.Softmax(dim=-1)(logits_flattened), x.size())
        
        # ori
        x_ori = torch.cat([matching_score_stacked_sat,
                          matching_score_stacked_osm,
                          self.sat_normalization(sat_descriptor_map),
                          self.osm_normalization(osm_descriptor_map)], dim=1)

        x_ori = self.deconv6_ori(x_ori)
        
        x_ori = torch.cat([x_ori, sat_feature_block15, x_ori, osm_feature_block15], dim=1)
        
        x_ori = self.conv6_ori(x_ori)
        x_ori = self.deconv5_ori(x_ori)
        
        x_ori = torch.cat([x_ori, sat_feature_block10, x_ori, osm_feature_block10], dim=1)
        
        x_ori = self.conv5_ori(x_ori)
        x_ori = self.deconv4_ori(x_ori)
        
        x_ori = torch.cat([x_ori, sat_feature_block4, x_ori, osm_feature_block4], dim=1)
        
        x_ori = self.conv4_ori(x_ori)
        x_ori = self.deconv3_ori(x_ori)
        
        x_ori = torch.cat([x_ori, sat_feature_block2, x_ori, osm_feature_block2], dim=1)
        
        x_ori = self.conv3_ori(x_ori)
        x_ori = self.deconv2_ori(x_ori)
        
        x_ori = torch.cat([x_ori, sat_feature_block0, x_ori, osm_feature_block0], dim=1)
        
        x_ori = self.conv2_ori(x_ori)
        x_ori = self.deconv1_ori(x_ori)
        x_ori = self.conv1_ori(x_ori)
        x_ori = nn.functional.normalize(x_ori, p=2, dim=1)

        # TODO: discuss how to deal with matching score stacked and contrastive learning loss
        # for the moment we simply average them
        matching_score_stacked = (matching_score_stacked_sat + matching_score_stacked_osm)/2
        matching_score_stacked2 = (matching_score_stacked_sat2 + matching_score_stacked_osm2)/2
        matching_score_stacked3 = (matching_score_stacked_sat3 + matching_score_stacked_osm3)/2
        matching_score_stacked4 = (matching_score_stacked_sat4 + matching_score_stacked_osm4)/2
        matching_score_stacked5 = (matching_score_stacked_sat5 + matching_score_stacked_osm5)/2
        matching_score_stacked6 = (matching_score_stacked_sat6 + matching_score_stacked_osm6)/2
        
        return logits_flattened, heatmap, x_ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6

    def compute_matching_score(self, shift, x, grd_des_len, grd_descriptor_map, grd_map_norm):
        '''
            LMU component: rolling and matching part
            TODO: use it once instead
        '''
        for i in range(20):
            sat_descriptor_map_rolled = torch.roll(x, shifts=-i*shift, dims=1)
            sat_descriptor_map_window = sat_descriptor_map_rolled[:,:grd_des_len, :, :]
            sat_map_norm = torch.norm(sat_descriptor_map_window, p='fro', dim=1, keepdim=True)

            matching_score = torch.sum((grd_descriptor_map*sat_descriptor_map_window), dim=1, keepdim=True) / (sat_map_norm * grd_map_norm) # cosine similarity
            if i == 0:
                matching_score_stacked = matching_score
            else:
                matching_score_stacked = torch.cat([matching_score_stacked, matching_score], dim=1)
        matching_score_max, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)

        return matching_score_max, matching_score_stacked

    def compute_loc(self, 
                    matching_score_max_sat, 
                    matching_score_max_osm, 
                    sat_descriptor_map, 
                    osm_descriptor_map, 
                    sat_feature_block, 
                    osm_feature_block,
                    conv,
                    deconv
                ):
        '''
          Localization Decoder: use output of previous LMU and prepare for next LMU
        '''
        x = torch.cat([matching_score_max_sat, matching_score_max_osm, self.sat_normalization(sat_descriptor_map), self.osm_normalization(osm_descriptor_map)], dim=1)
        x = deconv(x)
        x = torch.cat([x, sat_feature_block, x, osm_feature_block], dim=1)
        x = conv(x)

        return x
