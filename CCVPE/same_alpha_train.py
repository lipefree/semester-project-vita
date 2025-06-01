import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ["MKL_NUM_THREADS"] = "4" 
# os.environ["NUMEXPR_NUM_THREADS"] = "4" 
# os.environ["OMP_NUM_THREADS"] = "4" 

import argparse
from re import U
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import math
from dual_datasets import VIGORDataset
from losses import infoNCELoss, cross_entropy_loss, orientation_loss, loss_ccvpe
from fused_image_deformable_attention_model import CVM_VIGOR as CVM
from models import CVM_VIGOR_ori_prior as CVM_with_ori_prior
from vigor_osm_handler import prepare_osm_data
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from training_utils import get_meter_distance, get_orientation_distance, get_location
from PIL import Image
from torchvision.utils import save_image
from model_utils.fused_image_loss import Fusionloss
import random

torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--area', type=str, help='samearea or crossarea', default='samearea')
parser.add_argument('--training', choices=('True','False'), default='True')
parser.add_argument('--pos_only', choices=('True','False'), default='True')
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-4)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=8)
parser.add_argument('--weight_ori', type=float, help='weight on orientation loss', default=1e1)
parser.add_argument('--weight_infoNCE', type=float, help='weight on infoNCE loss', default=1e4)
parser.add_argument('-f', '--FoV', type=int, help='field of view', default=360)
parser.add_argument('--ori_noise', type=float, help='noise in orientation prior, 180 means unknown orientation', default=180.)
parser.add_argument('--osm', choices=('True', 'False'), default='True')
parser.add_argument('--osm_rendered', choices=('True', 'False'), default='True')
parser.add_argument('--osm_50n', choices=('True', 'False'), default='False')
parser.add_argument('--osm_concat', choices=('True', 'False'), default='False')
parser.add_argument('--use_fusion_loss', choices=('True', 'False'), default='False')
dataset_root='/work/vita/qngo/VIGOR'

args = vars(parser.parse_args())
area = args['area']
learning_rate = args['learning_rate']
batch_size = args['batch_size']
weight_ori = args['weight_ori']
weight_infoNCE = args['weight_infoNCE']
training = args['training'] == 'True'
pos_only = args['pos_only'] == 'True'
FoV = args['FoV']
pos_only = args['pos_only']
label = area + '_HFoV' + str(FoV) + "_" + area + "_lr_" + format(learning_rate, '.0e')
ori_noise = args['ori_noise']
ori_noise = 18 * (ori_noise // 18) # round the closest multiple of 18 degrees within prior 
use_osm = args['osm'] == 'True'
use_adapt = args['osm_50n'] == 'True' # 50 dim representation
use_osm_rendered = args['osm_rendered'] == 'True' # Use rendered tiles, NOTE: 50n and rendered are not compatible
use_concat = args['osm_concat'] == 'True' # concat osm tiles and sat images into 6 channels
use_fusion_loss = args['use_fusion_loss'] == 'True' # concat osm tiles and sat images into 6 channels

label = 'fused_image_ccvpe_alpha1'
use_fusion_loss = True
number_pass = 1

if os.path.exists(os.path.join('runs', label)) and "debug" not in label:
    raise Exception(f"name already taken {label}")

print(f'model name {label}')
writer = SummaryWriter(log_dir=os.path.join('runs', label))

if use_osm:
    prepare_osm_data(dataset_root)

if FoV == 360:
    circular_padding = True # apply circular padding along the horizontal direction in the ground feature extractor
else:
    circular_padding = False

transform_grd = transforms.Compose([
    transforms.Resize([320, 640]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_sat = transforms.Compose([
    # resize
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if training is False and ori_noise==180: # load pre-defined random orientation for testing
    if area == 'samearea':
        with open('samearea_orientation_test.npy', 'rb') as f:
            random_orientation = np.load(f)
    elif area == 'crossarea':
        with open('crossarea_orientation_test.npy', 'rb') as f:
            random_orientation = np.load(f)

vigor = VIGORDataset(dataset_root, 
                     split=area, 
                     train=training, 
                     pos_only=pos_only, 
                     transform=(transform_grd, transform_sat), 
                     ori_noise=ori_noise, 
                     use_osm_tiles=use_osm, 
                     use_50_n_osm_tiles=use_adapt, 
                     use_osm_rendered=True,
                     use_concat=use_concat)

if training is True:
    dataset_length = int(vigor.__len__())
    index_list = np.arange(vigor.__len__())
    np.random.shuffle(index_list)
    train_indices = index_list[0: int(len(index_list)*0.8)]
    val_indices = index_list[int(len(index_list)*0.8):]
    training_set = Subset(vigor, train_indices)
    val_set = Subset(vigor, val_indices)
    train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
else:
    test_dataloader = DataLoader(vigor, batch_size=batch_size, shuffle=False)

if training:
    torch.cuda.empty_cache()
    CVM_model = CVM(device, circular_padding, use_adapt=use_adapt, use_concat=use_concat, alpha_type=1)
    
    CVM_model.to(device)
    for param in CVM_model.parameters():
        param.requires_grad = True

    params = [p for p in CVM_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))

    global_step = 0
    # with torch.autograd.set_detect_anomaly(True):
    print('start training')
    for epoch in range(15):  # loop over the dataset multiple times
        running_loss = 0.0
        CVM_model.train()
        fusion_loss = Fusionloss()
        for i, data in enumerate(train_dataloader, 0):
            grd, sat, osm, gt, gt_with_ori, gt_orientation, city, _ = data
            grd = grd.to(device)
            sat = sat.to(device)
            osm = osm.to(device)
            gt = gt.to(device)
            gt_with_ori = gt_with_ori.to(device)
            gt_orientation = gt_orientation.to(device)

            gt_flattened = torch.flatten(gt, start_dim=1)
            gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

            # zero the parameter gradients

            heatmap = torch.zeros_like(gt)

            for timestep in range(number_pass):
                optimizer.zero_grad()
                sat_c = sat
                osm_c = osm

                sat_c = sat_c.detach()
                osm_c = osm_c.detach()
                heatmap = heatmap.detach()
                output = CVM_model(grd, sat_c, osm_c, heatmap, timestep)

                (
                    alpha,
                    fused_image,
                    logits_flattened,
                    heatmap,
                    ori,
                    matching_score_stacked,
                    matching_score_stacked2,
                    matching_score_stacked3,
                    matching_score_stacked4,
                    matching_score_stacked5,
                    matching_score_stacked6,
                ) = output

                output = output[1:]

                loss, loss_ce, loss_infonce, loss_ori = loss_ccvpe(
                    output, gt, gt_orientation, gt_with_ori, weight_infoNCE, weight_ori
                )

                if use_fusion_loss:
                    loss_fusion, loss_in, ssim_loss, loss_grad = fusion_loss(
                        image_vis=osm, image_ir=sat, generate_img=fused_image, i=None, labels=None
                    )

                    # loss += loss_fusion
                    writer.add_scalar("Train/loss_fusion", loss_fusion, global_step)

                writer.add_scalar("Train/loss_ce", loss_ce, global_step)
                writer.add_scalar("Train/loss_infonce", loss_infonce, global_step)
                writer.add_scalar("Train/loss_ori", loss_ori, global_step)
                writer.add_scalar("Loss/train", loss, global_step)

                loss.backward()
                optimizer.step()

            global_step += 1
            # print statistics
            running_loss += loss.item()

            if global_step % 200 == 0:    # print every 200 mini-batches
                print('record mini-batch')
                gt = gt.cpu().detach().numpy()
                gt_with_ori = gt_with_ori.cpu().detach().numpy()
                gt_orientation = gt_orientation.cpu().detach().numpy()
                heatmap = heatmap.cpu().detach().numpy()
                ori = ori.cpu().detach().numpy()

                distance = []
                orientation_error = []
                for batch_idx in range(len(city)):
                    loc_pred = get_location(heatmap[batch_idx, :, :, :])
                    loc_gt = get_location(gt[batch_idx, :, :, :])
                    meter_distance = get_meter_distance(
                        loc_gt, loc_pred, city[batch_idx], batch_idx
                    )
                    distance.append(meter_distance)

                    orientation_distance = get_orientation_distance(
                        gt_orientation, ori, loc_gt, loc_pred, batch_idx
                    )

                    if orientation_distance is not None:
                        orientation_error.append(orientation_distance)

                writer.add_scalar("Train/mean_distance", np.mean(distance), global_step)
                writer.add_scalar(
                    "Train/median_distance", np.median(distance), global_step
                )
                writer.add_scalar(
                    "Train/mean_orientation_error",
                    np.mean(orientation_error),
                    global_step,
                )
                writer.add_scalar(
                    "Train/median_orientation_error",
                    np.median(orientation_error),
                    global_step,
                )
    
                print(f'[{epoch}, {i + 1:5d}] loss: {np.mean(running_loss):.3f}')
                running_loss = 0.0
                writer.flush()

            if global_step % 5000 == 1:

                osm = osm.cpu().detach()
                sat = sat.cpu().detach()
                fused_image = fused_image.cpu().detach()

                invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

                sat = invTrans(sat)
                osm = invTrans(osm)

                for idx in range(len(city)):
                    base_dir = 'fused_images/'
                    label = label
                    im_path = os.path.join(base_dir, label, 'train', str(global_step))
                    os.makedirs(im_path, exist_ok=True)
                    save_image(osm[idx], os.path.join(im_path, f"osm_{idx}.png"))
                    save_image(sat[idx], os.path.join(im_path, f"sat_{idx}.png"))
                    save_image(fused_image[idx], os.path.join(im_path, f"fused_image_{idx}.png"))

        scratch_path = '/work/vita/qngo'
        model_name = 'models/VIGOR/'+label+'/' + str(epoch) + '/'
        model_dir = os.path.join(scratch_path, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(CVM_model.cpu().state_dict(), model_dir+'model.pt') # saving model
        CVM_model.cuda() # moving model to GPU for further training
        CVM_model.eval()

        # validation
        distance = []
        orientation_error = []
        running_loss_validation = []
        running_loss_ce_validation = []
        running_loss_infonce_validation = []
        running_loss_ori_validation = []
        running_loss_image_validation = []
        with torch.no_grad(): # fix this
            for i, data in enumerate(val_dataloader, 0):
                grd, sat, osm, gt, gt_with_ori, gt_orientation, city, _ = data
                grd = grd.to(device)
                sat = sat.to(device)
                osm = osm.to(device)
                gt = gt.to(device)
                gt_with_ori = gt_with_ori.to(device)
                gt_orientation = gt_orientation.to(device)

                grd_width = int(grd.size()[3] * FoV / 360)
                grd_FoV = grd[:, :, :, :grd_width]
                heatmap = torch.zeros_like(gt)

                for timestep in range(number_pass):
                    output = CVM_model(grd, sat, osm, heatmap, timestep)

                    (
                        alpha,
                        fused_image,
                        logits_flattened,
                        heatmap,
                        ori,
                        matching_score_stacked,
                        matching_score_stacked2,
                        matching_score_stacked3,
                        matching_score_stacked4,
                        matching_score_stacked5,
                        matching_score_stacked6,
                    ) = output

                output = output[1:]

                loss_validation, loss_ce, loss_infonce, loss_ori = loss_ccvpe(
                    output, gt, gt_orientation, gt_with_ori, weight_infoNCE, weight_ori
                )

                if use_fusion_loss:
                    loss_fusion, loss_in, ssim_loss, loss_grad = fusion_loss(
                        image_vis=osm, image_ir=sat, generate_img=fused_image, i=None, labels=None
                    )

                    # loss_validation += loss_fusion

                running_loss_validation.append(loss_validation.item())
                running_loss_ce_validation.append(loss_ce.item())
                running_loss_infonce_validation.append(loss_infonce.item())
                running_loss_ori_validation.append(loss_ori.item())
                running_loss_image_validation.append(loss_ori.item())

                gt = gt.cpu().detach().numpy()
                gt_with_ori = gt_with_ori.cpu().detach().numpy()
                gt_orientation = gt_orientation.cpu().detach().numpy()
                heatmap = heatmap.cpu().detach().numpy()
                ori = ori.cpu().detach().numpy()
                for batch_idx in range(gt.shape[0]):
                    loc_pred = get_location(heatmap[batch_idx, :, :, :])
                    loc_gt = get_location(gt[batch_idx, :, :, :])
                    meter_distance = get_meter_distance(
                        loc_gt, loc_pred, city[batch_idx], batch_idx
                    )
                    distance.append(meter_distance)

                    orientation_distance = get_orientation_distance(
                        gt_orientation, ori, loc_gt, loc_pred, batch_idx
                    )

                    if orientation_distance is not None:
                        orientation_error.append(orientation_distance)

                if i == 0:
                    osm = osm.cpu().detach()
                    sat = sat.cpu().detach()
                    fused_image = fused_image.cpu().detach()

                    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])

                    sat = invTrans(sat)
                    osm = invTrans(osm)

                    for idx in range(len(city)):
                        base_dir = 'fused_images/'
                        label = label
                        im_path = os.path.join(base_dir, label, 'validation', str(epoch))
                        os.makedirs(im_path, exist_ok=True)
                        save_image(osm[idx], os.path.join(im_path, f"osm_{idx}.png"))
                        save_image(sat[idx], os.path.join(im_path, f"sat_{idx}.png"))
                        save_image(fused_image[idx], os.path.join(im_path, f"fused_image_{idx}.png"))


            writer.add_scalar("Loss/Validation", np.mean(running_loss_validation), epoch)
            writer.add_scalar("Validation/loss_ce", np.mean(running_loss_ce_validation), epoch)
            writer.add_scalar("Validation/loss_infonce", np.mean(running_loss_infonce_validation), epoch)
            writer.add_scalar("Validation/loss_ori", np.mean(running_loss_ori_validation), epoch)
            mean_distance_error = np.mean(distance)
            writer.add_scalar("Validation/mean_distance", mean_distance_error, epoch)
        print(
            "epoch: ",
            epoch,
            "FoV" + str(FoV) + "_mean distance error on validation set: ",
            mean_distance_error,
        )
        file = "results/" + label + "_mean_distance_error.txt"
        with open(file, "ab") as f:
            np.savetxt(
                f,
                [mean_distance_error],
                fmt="%4f",
                header="FoV"
                + str(FoV)
                + "_validation_set_mean_distance_error_in_meters:",
                comments=str(epoch) + "_",
            )

        median_distance_error = np.median(distance)
        writer.add_scalar("Validation/median_distance", median_distance_error, epoch)
        print(
            "epoch: ",
            epoch,
            "FoV" + str(FoV) + "_median distance error on validation set: ",
            median_distance_error,
        )
        file = "results/" + label + "_median_distance_error.txt"
        with open(file, "ab") as f:
            np.savetxt(
                f,
                [median_distance_error],
                fmt="%4f",
                header="FoV"
                + str(FoV)
                + "_validation_set_median_distance_error_in_meters:",
                comments=str(epoch) + "_",
            )

        mean_orientation_error = np.mean(orientation_error)
        # print('epoch: ', epoch, 'FoV'+str(FoV)+ '_mean orientation error on validation set: ', mean_orientation_error)
        writer.add_scalar(
            "Validation/mean_orientation_error", mean_orientation_error, epoch
        )
        file = "results/" + label + "_mean_orientation_error.txt"
        with open(file, "ab") as f:
            np.savetxt(
                f,
                [mean_orientation_error],
                fmt="%4f",
                header="FoV" + str(FoV) + "_validation_set_mean_orientatione_error:",
                comments=str(epoch) + "_",
            )

        median_orientation_error = np.median(orientation_error)
        writer.add_scalar(
            "Validation/median_orientation_error", median_orientation_error, epoch
        )

        # print('epoch: ', epoch, 'FoV'+str(FoV)+ '_median orientation error on validation set: ', median_orientation_error)
        file = "results/" + label + "_median_orientation_error.txt"
        with open(file, "ab") as f:
            np.savetxt(
                f,
                [median_orientation_error],
                fmt="%4f",
                header="FoV" + str(FoV) + "_validation_set_median_orientation_error:",
                comments=str(epoch) + "_",
            )

    print("Finished Training")
    writer.flush()

else:
    torch.cuda.empty_cache()
    CVM_model = CVM_with_ori_prior(device, ori_noise, circular_padding)
    test_model_path = 'models/VIGOR/samearea/model.pt'
    print('load model from: ' + test_model_path)

    CVM_model.load_state_dict(torch.load(test_model_path))
    CVM_model.to(device)
    CVM_model.eval()

    distance = []
    distance_in_meters = []
    longitudinal_error_in_meters = []
    lateral_error_in_meters = []
    orientation_error = []
    orientation_pred = []
    probability = []
    probability_at_gt = []

    for i, data in enumerate(test_dataloader, 0):
        print(i)
        grd, sat, gt, gt_with_ori, gt_orientation, city, orientation_angle = data
        grd = grd.to(device)
        sat = sat.to(device)
        orientation_angle = orientation_angle.to(device)

        grd_width = int(grd.size()[3] * FoV / 360)
        grd_FoV = grd[:, :, :, :grd_width]

        gt_with_ori = gt_with_ori.to(device)

        gt_flattened = torch.flatten(gt, start_dim=1)
        gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

        gt_bottleneck = nn.MaxPool2d(64, stride=64)(gt_with_ori)

        logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = CVM_model(grd_FoV, sat)

        gt = gt.cpu().detach().numpy() 
        gt_with_ori = gt_with_ori.cpu().detach().numpy() 
        gt_orientation = gt_orientation.cpu().detach().numpy() 
        orientation_angle = orientation_angle.cpu().detach().numpy() 
        heatmap = heatmap.cpu().detach().numpy()
        ori = ori.cpu().detach().numpy()
        for batch_idx in range(gt.shape[0]):
            if city[batch_idx] == 'None':
                pass
            else:
                current_gt = gt[batch_idx, :, :, :]
                loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
                current_pred = heatmap[batch_idx, :, :, :]
                loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
                pixel_distance = np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2)
                distance.append(pixel_distance) 
                if city[batch_idx] == 'NewYork':
                    meter_distance = pixel_distance * 0.113248 / 512 * 640
                elif city[batch_idx] == 'Seattle':
                     meter_distance = pixel_distance * 0.100817 / 512 * 640
                elif city[batch_idx] == 'SanFrancisco':
                    meter_distance = pixel_distance * 0.118141 / 512 * 640
                elif city[batch_idx] == 'Chicago':
                    meter_distance = pixel_distance * 0.111262 / 512 * 640
                distance_in_meters.append(meter_distance) 

                cos_pred, sin_pred = ori[batch_idx, :, loc_pred[1], loc_pred[2]]
                if np.abs(cos_pred) <= 1 and np.abs(sin_pred) <=1:
                    a_acos_pred = math.acos(cos_pred)
                    if sin_pred < 0:
                        angle_pred = math.degrees(-a_acos_pred) % 360
                    else: 
                        angle_pred = math.degrees(a_acos_pred)
                    cos_gt, sin_gt = gt_orientation[batch_idx, :, loc_gt[1], loc_gt[2]]
                    a_acos_gt = math.acos(cos_gt)
                    if sin_gt < 0:
                        angle_gt = math.degrees(-a_acos_gt) % 360
                    else: 
                        angle_gt = math.degrees(a_acos_gt)

                    orientation_error.append(np.min([np.abs(angle_gt-angle_pred), 360-np.abs(angle_gt-angle_pred)]))     

                probability_at_gt.append(heatmap[batch_idx, 0, loc_gt[1], loc_gt[2]])


    print('mean localization error (m): ', np.mean(distance_in_meters))   
    print('median localization error (m): ', np.median(distance_in_meters))
    
    print('---------------------------------------')
    print('mean orientation error (degrees): ', np.mean(orientation_error))
    print('median orientation error (degrees): ', np.median(orientation_error))   
    
    print('---------------------------------------')
    print('mean probability at gt', np.mean(probability_at_gt))   
    print('median probability at gt', np.median(probability_at_gt)) 
    
