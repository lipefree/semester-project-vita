import torch
import torch.nn as nn


class PyramidMs(nn.Module):
    def __init__(self, embed_dims=256, query_dim=128):
        super(PyramidMs, self).__init__()

        self.embed_dims = embed_dims
        self.query_dim = query_dim

        # 256, 64, 64
        self.pyramid_conv0 = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=2, stride=2),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1),
            nn.LayerNorm([self.embed_dims, 64, 64])
        )

        # 256, 32, 32
        self.pyramid_conv1 = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=2, stride=2),
            nn.LayerNorm([self.embed_dims, 64, 64]),
            nn.GELU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=2, stride=2),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1),
            nn.LayerNorm([self.embed_dims, 32, 32])
        )

        # 256, 16, 16
        self.pyramid_conv2 = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=2, stride=2),
            nn.LayerNorm([self.embed_dims, 64, 64]),
            nn.GELU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=2, stride=2),
            nn.LayerNorm([self.embed_dims, 32, 32]),
            nn.GELU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=2, stride=2),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1),
            nn.LayerNorm([self.embed_dims, 16, 16])
        )


    def forward(self, features):
        # features : 256, 128, 128

        return features, self.pyramid_conv0(features), self.pyramid_conv1(features), self.pyramid_conv2(features)

class InterPyramidMs(nn.Module):
    def __init__(self, embed_dims=128, query_dim=256):
        super(PyramidMs, self).__init__()

        self.embed_dims = embed_dims
        self.query_dim = query_dim

        self.upsample = nn.Upsample(size=(256,256), mode="bilinear")

        # First block from 256,256,256 to 16,256,256
        self.pyramid_conv0 = nn.Sequential(
            nn.Conv2d(self.embed_dims, 16, kernel_size=1),
        )

        # Second block from 256,256,256 to 24,128,128
        self.pyramid_conv1 = nn.Sequential(
            nn.Conv2d(self.embed_dims, 24, kernel_size=1),
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1), # 128x128
        )

        # Third block to 40, 64, 64
        self.pyramid_conv2 = nn.Sequential(
            nn.Conv2d(self.embed_dims, 40, kernel_size=1),
            nn.Conv2d(40, 40, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(40, 40, kernel_size=3, stride=2, padding=1), # 64x64
        )

        # Fourth block to 112, 32, 32
        self.pyramid_conv3 = nn.Sequential(
            nn.Conv2d(self.embed_dims, 112, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(112, 112, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(112, 112, kernel_size=3, stride=2, padding=1), 
            nn.Conv2d(112, 112, kernel_size=3, stride=2, padding=1), # 32x32
        )

        # Fith block to 320, 16, 16
        self.pyramid_conv4 = nn.Sequential(
            nn.Conv2d(self.embed_dims, 320, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1), # 16x16
        )

        self.conv_block_volume = nn.Conv2d(320, 1280, kernel_size=1)
 
    def forward(self, features):
        # features : 256, 128, 128

        if(self.query_dim != 256):
            features = self.upsample(features)
            
        # Now we need to create the conv net to match for multi-scale
        pyramid_feature_block0 = self.pyramid_conv0(features)
        pyramid_feature_block2 = self.pyramid_conv1(features)
        pyramid_feature_block4 = self.pyramid_conv2(features)
        pyramid_feature_block10 = self.pyramid_conv3(features)
        pyramid_feature_block15 = self.pyramid_conv4(features)
        pyramid_feature_volume = self.conv_block_volume(pyramid_feature_block15)

        return pyramid_feature_block0, pyramid_feature_block2, pyramid_feature_block4, pyramid_feature_block10, pyramid_feature_block15, pyramid_feature_volume 


class MultiMAEPyramidMs(nn.Module):
    def __init__(self, embed_dims=256, query_dim=128):
        super(MultiMAEPyramidMs, self).__init__()

        self.embed_dims = embed_dims
        self.query_dim = query_dim

        # 256, 128, 128
        self.pyramid_conv3 = nn.Sequential(
            nn.Upsample(size=(128,128), mode='bilinear'),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1),
            nn.LayerNorm([self.embed_dims, 128, 128])
        )

        # 256, 64, 64
        self.pyramid_conv0 = nn.Sequential(
            nn.Upsample(size=(64,64), mode='bilinear'),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1),
            nn.LayerNorm([self.embed_dims, 64, 64])
        )

        # 256, 32, 32
        self.pyramid_conv1 = nn.Sequential(
            nn.Upsample(size=(32,32), mode='bilinear'),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1),
            nn.LayerNorm([self.embed_dims, 32, 32])
        )

        # 256, 16, 16
        self.pyramid_conv2 = nn.Sequential(
            nn.Upsample(size=(16,16), mode='bilinear'),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1),
            nn.LayerNorm([self.embed_dims, 16, 16])
        )


    def forward(self, features):
        # features : [bs, 129, 768]
        bs, num_tokens, dims = features.size()

        # drop extra token
        features = features[:, 1:, :]

        # 128 = 8 x 16
        features = features.transpose(1, 2).reshape(bs, dims, 8, 16) # shape: bs, 768, 8, 16

        return self.pyramid_conv3(features), self.pyramid_conv0(features), self.pyramid_conv1(features), self.pyramid_conv2(features)
