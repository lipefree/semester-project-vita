import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
# os.environ["MKL_NUM_THREADS"] = "4"
# os.environ["NUMEXPR_NUM_THREADS"] = "4"
# os.environ["OMP_NUM_THREADS"] = "4"

from torchvision import transforms
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from training_utils import *


def run_infer(idx, dataset, model, device):
    grd, sat, osm, gt, _, orientation, city, _ = dataset.__getitem__(idx)

    grd_feed = grd.unsqueeze(0)
    sat_feed = sat.unsqueeze(0)
    osm_feed = osm.unsqueeze(0)

    grd_feed = grd_feed.to(device)
    sat_feed = sat_feed.to(device)
    osm_feed = osm_feed.to(device)

    invTrans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )

    grd = invTrans(grd)
    sat = sat

    (
        logits_flattened,
        heatmap,
        ori,
        matching_score_stacked,
        matching_score_stacked2,
        matching_score_stacked3,
        matching_score_stacked4,
        matching_score_stacked5,
        matching_score_stacked6,
    ) = model(grd_feed, sat_feed, osm_feed)
    matching_score_max1, _ = torch.max(matching_score_stacked, dim=1, keepdim=True)
    matching_score_max2, _ = torch.max(matching_score_stacked2, dim=1, keepdim=True)
    matching_score_max3, _ = torch.max(matching_score_stacked3, dim=1, keepdim=True)
    matching_score_max4, _ = torch.max(matching_score_stacked4, dim=1, keepdim=True)
    matching_score_max5, _ = torch.max(matching_score_stacked5, dim=1, keepdim=True)
    matching_score_max6, _ = torch.max(matching_score_stacked6, dim=1, keepdim=True)

    # grd = grd.cpu().detach().numpy()
    # sat = sat.cpu().detach().numpy()
    gt = gt.permute(1, 2, 0)
    gt = gt.cpu().detach().numpy()
    loc_gt = np.unravel_index(gt.argmax(), gt.shape)

    orientation = orientation.permute(1, 2, 0).cpu().detach().numpy()

    heatmap = torch.squeeze(heatmap, dim=0).permute(1, 2, 0)
    heatmap = heatmap.cpu().detach().numpy()
    loc_pred = np.unravel_index(heatmap.argmax(), heatmap.shape)
    ori = torch.squeeze(ori, dim=0).permute(1, 2, 0)
    ori = ori.cpu().detach().numpy()

    cos_pred_dense = ori[:, :, 0]
    sin_pred_dense = ori[:, :, 1]
    cos_pred, sin_pred = ori[loc_pred[0], loc_pred[1], :]

    cos_gt, sin_gt = orientation[loc_gt[0], loc_gt[1], :]
    a_acos_gt = math.acos(cos_gt)
    if sin_gt < 0:
        angle_gt = math.degrees(-a_acos_gt) % 360
    else:
        angle_gt = math.degrees(a_acos_gt)

    return (
        city,
        sat,
        grd,
        heatmap,
        loc_gt,
        loc_pred,
        sin_pred_dense,
        cos_pred_dense,
        sin_pred,
        cos_pred,
    )


def show_image(
    sat,
    grd,
    heatmap,
    loc_gt,
    loc_pred,
    sin_pred_dense,
    cos_pred_dense,
    sin_pred,
    cos_pred,
):
    plt.figure(figsize=(8, 12))
    plt.imshow(grd.permute(1, 2, 0))
    plt.axvline(grd.size()[2] / 2, color="g")
    plt.axis("off")
    plt.savefig(
        "figures/" + area + "_" + str(idx) + "_grd_" + ".png",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(sat.permute(1, 2, 0))
    plt.imshow(heatmap, norm=LogNorm(vmax=np.max(heatmap)), alpha=0.6, cmap="Reds")
    plt.scatter(
        loc_gt[1],
        loc_gt[0],
        s=300,
        marker="^",
        facecolor="g",
        label="GT",
        edgecolors="white",
    )
    plt.scatter(
        loc_pred[1],
        loc_pred[0],
        s=300,
        marker="*",
        facecolor="gold",
        label="Ours",
        edgecolors="white",
    )
    xx, yy = np.meshgrid(np.linspace(0, 512, 512), np.linspace(0, 512, 512))
    cos_angle = ori[:, :, 0]
    sin_angle = ori[:, :, 1]
    plt.quiver(
        xx[::40, ::40],
        yy[::40, ::40],
        -sin_pred_dense[::40, ::40],
        cos_pred_dense[::40, ::40],
        linewidths=0.2,
        scale=14,
        width=0.01,
    )  # plot the predicted rotation angle + 90 degrees
    plt.quiver(
        loc_pred[1],
        loc_pred[0],
        -sin_pred,
        cos_pred,
        color="gold",
        linewidths=0.2,
        scale=10,
        width=0.015,
    )
    plt.quiver(
        loc_gt[1],
        loc_gt[0],
        -np.sin(angle_gt / 180 * np.pi),
        np.cos(angle_gt / 180 * np.pi),
        color="g",
        linewidths=0.2,
        scale=10,
        width=0.015,
    )
    plt.axis("off")
    plt.legend(loc=2, framealpha=0.8, labelcolor="black", prop={"size": 15})
