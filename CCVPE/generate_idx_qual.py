import os
import argparse
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from multiple_deformable_attention_model import CVM_VIGOR as CVM
from dual_datasets import VIGORDataset
import PIL.Image
from PIL import ImageFile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from training_utils import get_location, get_orientation_distance
from qualitative_utils import QualitativeUtils, get_meter_distance
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

dataset_root = "/work/vita/qngo/VIGOR"

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

base_model_path = "/work/vita/qngo/models/VIGOR/"
os.listdir(base_model_path)

selected_model = "multiple2_lighter_deformable_attention"
selected_epoch = 5
print(f'compute mean distance on test for {selected_model}')

torch.cuda.empty_cache()
CVM_model = CVM(device, ori_noise, use_adapt=use_adapt, use_concat=False)

qualitative_model = QualitativeUtils(CVM_model, selected_model, selected_epoch)

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
    random_orientation=zero_ori,
    use_osm_rendered=True
)

distance_array = np.zeros(len(vigor))

# indexes = np.arange(len(vigor))
# np.random.shuffle(indexes) # shuffling in debug will quickly approximate the mean

distance_array = qualitative_model.run_infer_batch(vigor)

# for i in tqdm(indexes):
#     idx = i
#     (
#         city,
#         sat,
#         osm,
#         grd,
#         heatmap,
#         loc_gt,
#         loc_pred,
#         sin_pred_dense,
#         cos_pred_dense,
#         sin_pred,
#         cos_pred,
#     ) = qualitative_model.run_infer(idx)
#     distance_array[i] = get_meter_distance(loc_pred, loc_gt, city, 0)
#     if i % 100 == 0:
#         print(distance_array[distance_array > 0].mean()) # use for debug

# show_image(sat, grd, heatmap, loc_gt, loc_pred, sin_pred_dense, cos_pred_dense, sin_pred, cos_pred)
# plt.savefig('figures/'+area+'_'+str(idx)+'_noise_in_orientation_'+str(ori_noise)+'.png', bbox_inches='tight', pad_inches=0)
# print('Images are written to figures/')
save_qual = os.path.join("/work/vita/qngo", "qualitative", selected_model, "distance_test_new.npy")

if not os.path.exists(os.path.join("/work/vita/qngo", "qualitative", selected_model)):
    os.mkdir(os.path.join("/work/vita/qngo", "qualitative", selected_model))

with open(save_qual, "wb") as f:
    np.save(f, distance_array)

# mean = np.mean(distance_array)
# median = np.median(distance_array)

# save_qual_res = os.path.join("/work/vita/qngo", "qualitative", selected_model, "test_quantitative_results.txt")
# with open(save_qual_res, 'w') as f:
#     f.write(f'mean in meter : {mean}, median in meter : {median}')
