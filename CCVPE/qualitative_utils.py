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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm
from training_utils import get_location, get_meter_distance, get_orientation_distance
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

def get_meter_distance(loc_gt, loc_pred, city: str, batch_idx) -> float:
    """
    distance in meters between groundtruth location (x,y) and predicted one
    """
    pixel_distance = np.sqrt(
        (loc_gt[0] - loc_pred[0]) ** 2 + (loc_gt[1] - loc_pred[1]) ** 2
    )
    meter_distance = -1
    if city == "NewYork":
        meter_distance = pixel_distance * 0.113248 / 512 * 640
    elif city == "Seattle":
        meter_distance = pixel_distance * 0.100817 / 512 * 640
    elif city == "SanFrancisco":
        meter_distance = pixel_distance * 0.118141 / 512 * 640
    elif city == "Chicago":
        meter_distance = pixel_distance * 0.111262 / 512 * 640
    if meter_distance == -1:
        print("problem finding the corresponding city")
        meter_distance = math.inf

    return meter_distance


def get_images(idx, dataset):
    grd, sat, osm, gt, _, orientation, city, _ = dataset.__getitem__(idx)
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    grd = invTrans(grd)
    sat = invTrans(sat)
    osm = invTrans(osm)
    return grd, sat, osm, gt


def show_ground(grd):
    plt.figure(figsize=(8,12))
    plt.imshow(  grd.permute(1, 2, 0)  )
    plt.axvline(grd.size()[2]/2, color='g')
    plt.axis('off')


def get_heatmap_array(experiment_name, base_test_result_path):
    heatmaps_file_path = f'{base_test_result_path}{experiment_name}/heatmaps.npz'
    return np.load(heatmaps_file_path)

def get_distance_array(experiment_name, best_test_result_path):
    distance_array_path = f'{best_test_result_path}{experiment_name}/distance_test.npy'
    return np.load(distance_array_path)

def get_heatmap(heatmap_array, idx):
    key = f"heatmap_{idx}"
    return heatmap_array[key].squeeze(0)


def show_image(sat, grd, heatmap, gt):
    gt = gt.permute(1, 2, 0)
    loc_pred = np.unravel_index(heatmap.argmax(), heatmap.shape)
    loc_gt = np.unravel_index(gt.argmax(), gt.shape)
    plt.figure(figsize=(6,6))
    plt.imshow(  sat.permute(1, 2, 0)  )
    plt.imshow(heatmap,  norm=LogNorm(vmax=np.max(heatmap)), alpha=0.4, cmap='Reds')
    plt.scatter(loc_gt[1], loc_gt[0], s=300, marker='^', facecolor='g', label='GT', edgecolors='white')
    plt.scatter(loc_pred[1], loc_pred[0], s=300, marker='*', facecolor='gold', label='Ours', edgecolors='white')
    xx,yy = np.meshgrid(np.linspace(0,512,512),np.linspace(0,512,512))
    plt.axis('off')
    plt.legend(loc=2, framealpha=0.8, labelcolor='black', prop={'size': 15})

def show_image_subplot(sat, grd, heatmap, gt, n, m, axs, title):
    gt = gt.permute(1, 2, 0)
    loc_pred = np.unravel_index(heatmap.argmax(), heatmap.shape)
    loc_gt = np.unravel_index(gt.argmax(), gt.shape)
    axs[n, m].imshow(  sat.permute(1, 2, 0)  )
    axs[n, m].imshow(heatmap,  norm=LogNorm(vmax=np.max(heatmap)), alpha=0.4, cmap='Reds')
    axs[n, m].scatter(loc_gt[1], loc_gt[0], s=300, marker='^', facecolor='g', label='GT', edgecolors='white')
    axs[n, m].scatter(loc_pred[1], loc_pred[0], s=300, marker='*', facecolor='gold', label='Ours', edgecolors='white')
    xx,yy = np.meshgrid(np.linspace(0,512,512),np.linspace(0,512,512))
    axs[n, m].axis('off')
    axs[n, m].legend(loc=2, framealpha=0.8, labelcolor='black', prop={'size': 15})
    axs[n, m].title.set_text(title)
    
class QualitativeUtils:

    def __init__(self, model, model_name, epoch, dataset=None):

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model_path = "/work/vita/qngo/models/VIGOR/"

        select_epoch = str(epoch)
        model_path = os.path.join(base_model_path, model_name, select_epoch, "model.pt")

        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()

        self.model = model
        torch.cuda.empty_cache()

        self.dataset = dataset
        self.batch_size = 16

    def run_infer_batch(self, dataset):
        self.dataset = dataset
        test_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        distance = []
        distance_in_meters = []
        longitudinal_error_in_meters = []
        lateral_error_in_meters = []
        orientation_error = []
        orientation_pred = []
        probability = []
        probability_at_gt = []

        for i, data in enumerate(tqdm(test_dataloader), 0):
            grd, sat, osm, gt, gt_with_ori, gt_orientation, city, orientation_angle = data
            grd = grd.to(self.device)
            sat = sat.to(self.device)
            osm = osm.to(self.device)
            orientation_angle = orientation_angle.to(self.device)

            FoV = 360
            grd_width = int(grd.size()[3] * FoV / 360)
            grd_FoV = grd[:, :, :, :grd_width]

            gt_with_ori = gt_with_ori.to(self.device)

            gt_flattened = torch.flatten(gt, start_dim=1)
            gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

            gt_bottleneck = nn.MaxPool2d(64, stride=64)(gt_with_ori)

            with torch.no_grad():
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
                ) = self.model(grd_FoV, sat, osm)

            gt = gt.cpu().detach().numpy()
            gt_with_ori = gt_with_ori.cpu().detach().numpy()
            gt_orientation = gt_orientation.cpu().detach().numpy()
            orientation_angle = orientation_angle.cpu().detach().numpy()
            heatmap = heatmap.cpu().detach().numpy()
            ori = ori.cpu().detach().numpy()
            for batch_idx in range(gt.shape[0]):
                if city[batch_idx] == "None":
                    pass
                else:
                    current_gt = gt[batch_idx, :, :, :]
                    loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
                    current_pred = heatmap[batch_idx, :, :, :]
                    loc_pred = np.unravel_index(
                        current_pred.argmax(), current_pred.shape
                    )
                    pixel_distance = np.sqrt(
                        (loc_gt[1] - loc_pred[1]) ** 2 + (loc_gt[2] - loc_pred[2]) ** 2
                    )
                    distance.append(pixel_distance)
                    if city[batch_idx] == "NewYork":
                        meter_distance = pixel_distance * 0.113248 / 512 * 640
                    elif city[batch_idx] == "Seattle":
                        meter_distance = pixel_distance * 0.100817 / 512 * 640
                    elif city[batch_idx] == "SanFrancisco":
                        meter_distance = pixel_distance * 0.118141 / 512 * 640
                    elif city[batch_idx] == "Chicago":
                        meter_distance = pixel_distance * 0.111262 / 512 * 640
                    distance_in_meters.append(meter_distance)

                    cos_pred, sin_pred = ori[batch_idx, :, loc_pred[1], loc_pred[2]]
                    if np.abs(cos_pred) <= 1 and np.abs(sin_pred) <= 1:
                        a_acos_pred = math.acos(cos_pred)
                        if sin_pred < 0:
                            angle_pred = math.degrees(-a_acos_pred) % 360
                        else:
                            angle_pred = math.degrees(a_acos_pred)
                        cos_gt, sin_gt = gt_orientation[
                            batch_idx, :, loc_gt[1], loc_gt[2]
                        ]
                        a_acos_gt = math.acos(cos_gt)
                        if sin_gt < 0:
                            angle_gt = math.degrees(-a_acos_gt) % 360
                        else:
                            angle_gt = math.degrees(a_acos_gt)

                        orientation_error.append(
                            np.min(
                                [
                                    np.abs(angle_gt - angle_pred),
                                    360 - np.abs(angle_gt - angle_pred),
                                ]
                            )
                        )

                    probability_at_gt.append(
                        heatmap[batch_idx, 0, loc_gt[1], loc_gt[2]]
                    )

            if i % 20 == 0:
                print(np.mean(distance_in_meters))

        return np.array(distance_in_meters)

    def run_infer(self, idx):
        grd, sat, osm, gt, _, orientation, city, _ = self.dataset.__getitem__(idx)

        grd_feed = grd.unsqueeze(0)
        sat_feed = sat.unsqueeze(0)
        osm_feed = osm.unsqueeze(0)

        grd_feed = grd_feed.to(self.device)
        sat_feed = sat_feed.to(self.device)
        osm_feed = osm_feed.to(self.device)

        invTrans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

        grd = invTrans(grd)
        sat = invTrans(sat)
        osm = invTrans(osm)

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
        ) = self.model(grd_feed, sat_feed, osm_feed)
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
            osm,
            grd,
            heatmap,
            loc_gt,
            loc_pred,
            sin_pred_dense,
            cos_pred_dense,
            sin_pred,
            cos_pred,
        )
