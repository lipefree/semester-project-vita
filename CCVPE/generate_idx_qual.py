import os
import argparse
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from models import CVM_VIGOR as CVM
from datasets import VIGORDataset
import PIL.Image
from PIL import ImageFile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from training_utils import *
from qualitative_utils import *
from tqdm import tqdm

area = "samearea"
ori_noise = 180.0
ori_noise = 18 * (
    ori_noise // 18
)  # round the closest multiple of 18 degrees within prior
pos_only = True
use_osm = True
use_adapt = False
training = False
use_concat = False

dataset_root = "/scratch/izar/qngo/VIGOR"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)
np.random.seed(0)

transform_grd = transforms.Compose(
    [
        transforms.Resize([320, 640]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_sat = transforms.Compose(
    [
        # resize
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

zero_ori = np.zeros(52605)

vigor = VIGORDataset(
    dataset_root,
    split=area,
    train=training,
    pos_only=pos_only,
    transform=(transform_grd, transform_sat),
    ori_noise=ori_noise,
    use_osm_tiles=use_osm,
    use_50_n_osm_tiles=use_adapt,
    use_concat=use_concat,
    random_orientation=zero_ori
)

base_model_path = "/scratch/izar/qngo/models/VIGOR/"
os.listdir(base_model_path)

selected_model = "samearea_HFoV360_samearea_lr_1e-04normalized_osm_rendered_tile"
select_epoch = str(9)

test_model_path = os.path.join(
    base_model_path, selected_model, select_epoch, "model.pt"
)
torch.cuda.empty_cache()
CVM_model = CVM(device, ori_noise, use_adapt=use_adapt, use_concat=False)
CVM_model.load_state_dict(torch.load(test_model_path))
CVM_model.to(device)
CVM_model.eval()

distance_array = np.zeros(len(vigor))

for i in tqdm(range(len(vigor))):
    idx = i
    (
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
    ) = run_infer(idx, vigor, CVM_model, device)
    get_meter_distance(loc_pred, loc_gt, city, 0)
    distance_array[i] = get_meter_distance(loc_pred, loc_gt, city, 0)

# show_image(sat, grd, heatmap, loc_gt, loc_pred, sin_pred_dense, cos_pred_dense, sin_pred, cos_pred)
# plt.savefig('figures/'+area+'_'+str(idx)+'_noise_in_orientation_'+str(ori_noise)+'.png', bbox_inches='tight', pad_inches=0)
# print('Images are written to figures/')
save_qual = os.path.join("/scratch/izar/qngo", "qualitative", selected_model, "distance_test_new.npy")

if not os.path.exists(os.path.join("/scratch/izar/qngo", "qualitative", selected_model)):
    os.mkdir(os.path.join("/scratch/izar/qngo", "qualitative", selected_model))

with open(save_qual, "wb") as f:
    np.save(f, distance_array)
