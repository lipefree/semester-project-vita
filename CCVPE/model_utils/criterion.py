# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MaskedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param label_smoothing: Amount of smoothing in the loss (default is 0.0)
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, label_smoothing : float = 0.0):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.label_smoothing = label_smoothing

    def forward(self, input, target, mask=None, osm=False):

        loss = F.cross_entropy(input, target, reduction='none', label_smoothing=self.label_smoothing)

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            if osm:
                print('osm tile ', target)

            H, W = input.shape[-2:]
            nh, nw = H // self.scale_factor, W // self.scale_factor
            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss


class MaskedMSELoss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix
        self.sobelconv = Sobelxy()

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p)
        return imgs

    def forward(self, input, target, mask=None, osm=False):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.mse_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)
                
            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            
            if osm: # [B, C, H, W]
                mode_value, _ = torch.mode(target.mean(dim=1).flatten())
                background_mean_value = 1.98 # We don't want the model to focus on reconstructing the background
                tolerance = 0.01
                osm_mask = target.mean(dim=1) # [B, H W]
                osm_mask = (osm_mask >= (background_mean_value - tolerance)) & (osm_mask <= (background_mean_value + tolerance))
                loss = loss.masked_fill(osm_mask, 0.0)

            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean() # Account for zero masks
        else:
            loss = loss.mean() # If this is ever nan, we want it to stop training

        return loss


class MaskedL1Loss(nn.Module):
    """L1 loss with masking
    :param patch_size: Patch size
    :param stride: Stride of task / modality
    :param norm_pix: Normalized pixel loss
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p)
        return imgs

    def forward(self, input, target, mask=None):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.l1_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss
    
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        
    def rgb_to_grayscale(self, images):
        """
        Converts a batch of RGB images to grayscale.
        """
        weights = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype)
        weights = weights.view(1, 3, 1, 1)
        grayscale = (images * weights).sum(dim=1, keepdim=True)
    
        return grayscale

    def forward(self,x):
        x = self.rgb_to_grayscale(x)
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
