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
from dist_dual_datasets import VIGORDataset
from losses import infoNCELoss, cross_entropy_loss, orientation_loss, loss_ccvpe, loss_router, get_distances
from learn_new_router_model import CVM_VIGOR as CVM
from models import CVM_VIGOR_ori_prior as CVM_with_ori_prior
from vigor_osm_handler import prepare_osm_data
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from training_utils import get_meter_distance, get_orientation_distance, get_location
import sklearn.metrics as metrics

torch.manual_seed(17)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"The device is: {}".format(device)

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

# label = 'router_overfit_debug2'
label = 'score_stacked_debug'
learning_rate = 1e-4
batch_size = 4
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
    # np.random.shuffle(index_list)

    print(f'data set lenght {dataset_length}')
    # val_indices = index_list[int(len(index_list)*0.9):]
    val_indices = index_list[:1]
    if (subset := False):
    # we can also load sub set according to a max dist : dist_sub_1.npy  dist_sub_2.npy  dist_sub_3.npy  dist_sub_5.npy  dist_sub_9.npy dist_sub_30cm.npy dist_sub_50cm.npy dist_sub_15cm.npy
        dists_path = '/work/vita/qngo/dists'
        dist_file = "dist_sub_9.npy"
        dist_indices = np.load(os.path.join(dists_path, dist_file))
        np.random.shuffle(dist_indices)
        train_indices = dist_indices
        # train_indices = index_list[0: int(len(index_list)*0.8)]
        # train_indices = index_list[0:3000]
        print(f'size before clashing {len(train_indices)}')

        len_before = len(train_indices)
        to_del = []
        for idx, val in enumerate(train_indices):
            if val in val_indices:
                to_del.append(idx)

        train_indices = np.delete(train_indices, to_del)
        print(f'size after clashing {len(train_indices)}')
        print(f'difference is {len_before - len(train_indices)}')

    else:
        train_indices = index_list[:1]

    # val_indices = index_list[20:25]
    training_set = Subset(vigor, train_indices)
    val_set = Subset(vigor, val_indices)
    train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
else:
    test_dataloader = DataLoader(vigor, batch_size=batch_size, shuffle=False)

if training:
    torch.cuda.empty_cache()

    model_path = '/work/vita/qngo/models/VIGOR/router_loss1_dist_over_1v2/10/model.pt'
    CVM_model = CVM(device, circular_padding, use_adapt=use_adapt, use_concat=use_concat)
    # CVM_model.load_state_dict(torch.load(model_path))
    CVM_model.to(device)
    for param in CVM_model.parameters():
        param.requires_grad = True

    params = [p for p in CVM_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(params, lr=1e-2)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, pct_start=0.1, steps_per_epoch=len(train_dataloader), epochs=10)

    global_step = 0
    global_osm_selected = 0
    global_sat_selected = 0 
    global_osm_true = 0
    global_sat_true = 0 

    update_step = 0
    nbr_sample = 0

    nbr_correct_sample = 0
    sat_mean = []
    osm_mean = []

    window_label = []
    window_pred = []

    window_sat_selected = []

    distance = []
    window_size = 2000
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        CVM_model.train()
        print('start training')
        for i, data in enumerate(train_dataloader, 0):
            sat_dist, osm_dist, grd, sat, osm, gt, gt_with_ori, gt_orientation, city, _ = data
            sat_dist = sat_dist.to(device)
            osm_dist = osm_dist.to(device)
            grd = grd.to(device)
            sat = sat.to(device)
            osm = osm.to(device)
            gt = gt.to(device)
            gt_with_ori = gt_with_ori.to(device)
            gt_orientation = gt_orientation.to(device)

            gt_flattened = torch.flatten(gt, start_dim=1)
            gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

            # Sometimes sat_dist or osm_dist were not well init and are 0, we should skip them (it's around 1% of the data)

            # mask = (sat_dist == 0 or osm_dist == 0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits, output_osm, output_sat = CVM_model(grd, sat, osm)

            (   osm_logits_flattened,
                osm_heatmap,
                osm_ori,
                osm_matching_score_stacked,
                osm_matching_score_stacked2,
                osm_matching_score_stacked3,
                osm_matching_score_stacked4,
                osm_matching_score_stacked5,
                osm_matching_score_stacked6,
            ) = output_osm

               

            (   sat_logits_flattened,
                sat_heatmap,
                sat_ori,
                sat_matching_score_stacked,
                sat_matching_score_stacked2,
                sat_matching_score_stacked3,
                sat_matching_score_stacked4,
                sat_matching_score_stacked5,
                sat_matching_score_stacked6,
            ) = output_sat

            gt_choices, sat_dist, osm_dist = get_distances(gt, osm_heatmap, sat_heatmap, city)

            chosen = logits.argmax(dim=1)

            bs = logits.size()[0]
            gt_choices = (sat_dist > osm_dist).long() # This means : 0 when sat is better, smaller dist is better (closer to gt)

            sat_mean.append(torch.mean(sat_dist).cpu().detach().numpy())
            osm_mean.append(torch.mean(osm_dist).cpu().detach().numpy())

            t = -10
            # t = 30 - 28*(global_step/60000)
            # sat:0.6, osm:0.4 unbalanced dataset
            weight = torch.tensor([1.0, 1.0]).to(device)
            
            loss1, loss2 = loss_router(
                weight, logits, gt_choices, sat_dist, osm_dist, t
            )

            # lambd = 1/20000
            lambd = 1
            loss = torch.mean(loss1 * lambd * loss2)
            

            # print(f'computed sat dist {sat_dist}')
            # print(f'computed osm dist {osm_dist}')

            nbr_sample += bs

            gt_cpu = gt_choices.cpu().detach().numpy()
            chosen_cpu = chosen.cpu().detach().numpy()

            chosen_cpu = np.array([chosen_cpu > 0.5]).astype(int)

            chosen_t = chosen_cpu.flatten()

            labels = gt_choices.cpu().detach().numpy().flatten()
            window_pred.extend(chosen_t)
            window_label.extend(labels)


            # print(f'chosen_cpu {chosen_cpu}')
            # print(f'gt_cpu {gt_cpu}')
            
            curr_acc = np.sum(gt_cpu == chosen_cpu)

            nbr_correct_sample += curr_acc

            # print(f'acc {curr_acc}')
            writer.add_scalar("train/acc",nbr_correct_sample/nbr_sample, global_step)

            sat_this_round = np.sum([chosen.cpu().detach().numpy() < 0.5])
            global_sat_selected += sat_this_round
            global_osm_selected += bs - sat_this_round

            window_sat_selected.extend(np.array([chosen.cpu().detach().numpy() < 0.5]).astype(int).flatten())
            
            osm_this_round_true = np.sum([gt_choices.cpu().detach().numpy()])
            global_sat_true += osm_this_round_true
            global_osm_true += bs - osm_this_round_true

            writer.add_scalar("train/loss1", loss1.cpu().detach().numpy().mean(), global_step)
            writer.add_scalar("train/loss2", loss2.cpu().detach().numpy().mean(), global_step)
            writer.add_scalar("train/loss", loss.cpu().detach().numpy().mean(), global_step)
            writer.add_scalar("Loss/train", loss.cpu().detach().numpy().mean(), global_step)
            writer.add_scalar("train/dist_diff", (sat_dist - osm_dist).cpu().detach().numpy().max(), global_step)
            writer.add_scalar("train/dist_osm", torch.abs(osm_dist).cpu().detach().numpy().max(), global_step)
            writer.add_scalar("train/dist_sat", torch.abs(sat_dist).cpu().detach().numpy().max(), global_step)
            writer.add_scalar("train/dist_sat_mean", np.mean(sat_mean), global_step)
            writer.add_scalar("train/dist_osm_mean", np.mean(osm_mean), global_step)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            global_step += 1
            # print statistics
            running_loss += loss.item()

            if i % 50 == 0:    # print every 200 mini-batches
                # print(f'logits {logits}')
                writer.add_scalar("train_ratio/sat", global_sat_selected/(global_osm_selected + global_sat_selected), global_step)
                writer.add_scalar("train_ratio/osm", global_osm_selected/(global_osm_selected + global_sat_selected), global_step)
                writer.add_scalar("train_ratio/osm_true", global_sat_true/(global_osm_true + global_sat_true), global_step)
                writer.add_scalar("train_ratio/sat_true", global_osm_true/(global_osm_true + global_sat_true), global_step)

                # sliding window
                if len(window_label) > window_size:
                    to_cut = len(window_label) - window_size
                    window_label = window_label[to_cut:]
                    window_pred = window_pred[to_cut:]
                    window_sat_selected = window_sat_selected[to_cut:]
                    distance = distance[to_cut:]

                w_labels = np.array(window_label).flatten()
                w_preds = np.array(window_pred).flatten()

                writer.add_scalar("Train/f1_score", metrics.f1_score(w_labels,w_preds), global_step)
                writer.add_scalar("Train/acc", metrics.accuracy_score(w_labels,w_preds), global_step)
                writer.add_scalar("Train/recall", metrics.recall_score(w_labels,w_preds), global_step)
                writer.add_scalar("Train/precision", metrics.precision_score(w_labels,w_preds), global_step)
                writer.add_scalar("Train/sat_ratio_window", np.sum(window_sat_selected)/len(window_sat_selected), global_step)
                writer.add_scalar("Train/osm_ratio_window", 1 - np.sum(window_sat_selected)/len(window_sat_selected), global_step)

                # First we need to infer what to use at a batch level
                chosen = chosen.cpu().detach().numpy()

                heatmap = []
                ori = []

                gt = gt.cpu().detach().numpy()
                gt_with_ori = gt_with_ori.cpu().detach().numpy()
                gt_orientation = gt_orientation.cpu().detach().numpy()
                heatmap = np.array(heatmap)
                ori = np.array(ori)
                sat_dist = sat_dist.cpu().detach().numpy()
                osm_dist = osm_dist.cpu().detach().numpy()

                orientation_error = []
                for batch_idx in range(len(city)):
                #     loc_pred = get_location(heatmap[batch_idx, :, :, :])
                #     loc_gt = get_location(gt[batch_idx, :, :, :])
                #     meter_distance = get_meter_distance(
                #         loc_gt, loc_pred, city[batch_idx], batch_idx
                #     )
                    if chosen[batch_idx] > 0.5:
                        meter_distance = osm_dist[batch_idx]
                    else:
                        meter_distance = sat_dist[batch_idx]
                    distance.append(meter_distance)

                #     orientation_distance = get_orientation_distance(
                #         gt_orientation, ori, loc_gt, loc_pred, batch_idx
                #     )

                #     if orientation_distance is not None:
                #         orientation_error.append(orientation_distance)

                writer.add_scalar("Train/mean_distance", np.mean(distance), global_step)
                writer.add_scalar(
                    "Train/median_distance", np.median(distance), global_step
                )
                # writer.add_scalar(
                #     "Train/mean_orientation_error",
                #     np.mean(orientation_error),
                #     global_step,
                # )
                # writer.add_scalar(
                #     "Train/median_orientation_error",
                #     np.median(orientation_error),
                #     global_step,
                # )
    
                # print(f'[{epoch}, {i + 1:5d}] loss: {np.mean(running_loss):.3f}')
                running_loss = 0.0
                writer.flush()
            
        scratch_path = '/work/vita/qngo'
        model_name = 'models/VIGOR/'+label+'/' + str(epoch) + '/'
        model_dir = os.path.join(scratch_path, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(CVM_model.cpu().state_dict(), model_dir+'model.pt') # saving model
        CVM_model.cuda() # moving model to GPU for further training
        CVM_model.eval()

        # validation
        val_distance = []
        orientation_error = []
        running_loss_validation = []

        current_osm_choice = 0
        current_sat_choice = 0

        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                sat_dist, osm_dist, grd, sat, osm, gt, gt_with_ori, gt_orientation, city, _ = data
                sat_dist = sat_dist.to(device)
                osm_dist = osm_dist.to(device)
                grd = grd.to(device)
                sat = sat.to(device)
                osm = osm.to(device)
                gt = gt.to(device)
                gt_with_ori = gt_with_ori.to(device)
                gt_orientation = gt_orientation.to(device)

                grd_width = int(grd.size()[3] * FoV / 360)
                grd_FoV = grd[:, :, :, :grd_width]

                logits, output_osm, output_sat = CVM_model(grd, sat, osm)

                (   osm_logits_flattened,
                    osm_heatmap,
                    osm_ori,
                    osm_matching_score_stacked,
                    osm_matching_score_stacked2,
                    osm_matching_score_stacked3,
                    osm_matching_score_stacked4,
                    osm_matching_score_stacked5,
                    osm_matching_score_stacked6,
                ) = output_osm

                (   sat_logits_flattened,
                    sat_heatmap,
                    sat_ori,
                    sat_matching_score_stacked,
                    sat_matching_score_stacked2,
                    sat_matching_score_stacked3,
                    sat_matching_score_stacked4,
                    sat_matching_score_stacked5,
                    sat_matching_score_stacked6,
                ) = output_sat

                gt_choices, sat_dist, osm_dist = get_distances(gt, osm_heatmap, sat_heatmap, city)

                chosen = logits.argmax(dim=1)

                t = -1

                gt_choices = (sat_dist > osm_dist).long() # This means : 0 when sat is better, smaller dist is better (closer to gt)

                weight = torch.tensor([0.8, 1.0]).to(device)
                loss_validation_1, loss_validation_2 = loss_router(
                    weight, logits, gt_choices, sat_dist, osm_dist, t
                )

                loss_validation = torch.sum(loss_validation_1)

                running_loss_validation.append(loss_validation.item())
                sat_this_round = chosen.cpu().detach().numpy().sum()
                current_sat_choice += sat_this_round
                current_osm_choice += batch_size - sat_this_round

                # First we need to infer what to use at a batch level
                chosen = chosen.cpu().detach().numpy()

                heatmap = []
                ori = []

                chosen_t = np.array([chosen > 0.5]).astype(int).flatten()

                labels = gt_choices.cpu().detach().numpy().flatten()
                all_preds.extend(chosen_t)
                all_labels.extend(labels)

                heatmap = np.array(heatmap)
                ori = np.array(ori)

                gt = gt.cpu().detach().numpy()
                gt_with_ori = gt_with_ori.cpu().detach().numpy()
                gt_orientation = gt_orientation.cpu().detach().numpy()

                sat_dist = sat_dist.cpu().detach().numpy()
                osm_dist = osm_dist.cpu().detach().numpy()
                
                for batch_idx in range(gt.shape[0]):
                #     loc_pred = get_location(heatmap[batch_idx, :, :, :])
                #     loc_gt = get_location(gt[batch_idx, :, :, :])
                #     meter_distance = get_meter_distance(
                #         loc_gt, loc_pred, city[batch_idx], batch_idx
                    if chosen[batch_idx] > 0.5:
                        meter_distance = osm_dist[batch_idx]
                    else:
                        meter_distance = sat_dist[batch_idx]
                    val_distance.append(meter_distance)

                #     distance.append(meter_distance)

                #     orientation_distance = get_orientation_distance(
                #         gt_orientation, ori, loc_gt, loc_pred, batch_idx
                #     )

                #     if orientation_distance is not None:
                #         orientation_error.append(orientation_distance)

            labels = np.array(all_labels).flatten()
            preds = np.array(all_preds).flatten()

            writer.add_scalar("Validation/f1_score", metrics.f1_score(labels, preds), epoch)
            writer.add_scalar("Validation/acc", metrics.accuracy_score(labels, preds), epoch)
            writer.add_scalar("Validation/recall", metrics.recall_score(labels, preds), epoch)
            writer.add_scalar("Validation/precision", metrics.precision_score(labels, preds), epoch)

            writer.add_scalar("Loss/Validation", np.mean(running_loss_validation), epoch)
            mean_distance_error = np.mean(val_distance)
            writer.add_scalar("Validation/mean_distance", mean_distance_error, epoch)
            writer.add_scalar("Validation/ratio_osm", current_sat_choice/(current_osm_choice + current_sat_choice), epoch)
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

        median_distance_error = np.median(val_distance)
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
    
