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
from model_utils.multimae_encoder import MultiMaeEncoder
from models import CVM_VIGOR_ori_prior as CVM_with_ori_prior
from vigor_osm_handler import prepare_osm_data
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from training_utils import get_meter_distance, get_orientation_distance, get_location
from torchvision.utils import save_image
from model_utils.fused_image_loss import Fusionloss
from model_utils.criterion import MaskedMSELoss, MaskedL1Loss
from mae_image_utils import get_pred_with_input, get_masked_image

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

label = 'multimae_encoder_new-save_from-epoch-28'
use_fusion_loss = True

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
    # train_indices = index_list[0: 3]
    val_indices = index_list[int(len(index_list)*0.8):]
    # val_indices = index_list[4:5]
    training_set = Subset(vigor, train_indices)
    val_set = Subset(vigor, val_indices)
    train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
else:
    test_dataloader = DataLoader(vigor, batch_size=batch_size, shuffle=False)

if training:
    torch.cuda.empty_cache()
    model = MultiMaeEncoder()
    
    model.to(device)
    for param in model.parameters():
        param.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, pct_start=0.05, steps_per_epoch=len(train_dataloader), epochs=400)

    global_step = 0
    # with torch.autograd.set_detect_anomaly(True):

    epoch = 28
    model.load_multimaae_weights(f'/work/vita/qngo/models/VIGOR/multimae_encoder_new-save/{epoch}/model.pt')

    for epoch in range(400):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        fusion_loss = Fusionloss()
        patch_size = 32
        masked_mse_loss = MaskedMSELoss(patch_size=patch_size)
        masked_l1_loss = MaskedL1Loss(patch_size=patch_size)
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
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(grd, sat, osm)

            (
                preds,
                masks,
                encoder_tokens
            ) = output

            recons_osm = preds['osm']
            recons_sat = preds['sat']

            if use_fusion_loss:
                loss_osm = masked_mse_loss(recons_osm.float(), osm, mask=masks['osm'])
                loss_sat = masked_mse_loss(recons_sat.float(), sat, mask=masks['sat'])

                loss = loss_osm
                loss += loss_sat
                writer.add_scalar("Train/loss_osm_recons", loss_osm, global_step)
                writer.add_scalar("Train/loss_sat_recons", loss_sat, global_step)


            writer.add_scalar("Loss/train", loss, global_step)

            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            # print statistics
            running_loss += loss.item()
            
            if global_step % 8000 == 1:
                masked_osm_image = get_masked_image(
                    osm, 
                    masks['osm'],
                    image_size=512,
                    patch_size=patch_size,
                    mask_value=0.0
                ).detach().cpu()
            
                masked_sat_image = get_masked_image(
                    sat, 
                    masks['sat'],
                    image_size=512,
                    patch_size=patch_size,
                    mask_value=0.0
                ).detach().cpu()

                # print('masked osm image size ', masked_osm_image.size())

                pred_osm = get_pred_with_input(
                    osm, 
                    recons_osm, 
                    masks['osm'],
                    patch_size=patch_size,
                    image_size=512
                ).detach().cpu()
            
                pred_sat = get_pred_with_input(
                    sat, 
                    recons_sat, 
                    masks['sat'],
                    patch_size=patch_size,
                    image_size=512
                ).detach().cpu()
                
                osm = osm.cpu().detach()
                sat = sat.cpu().detach()
                recons_osm = recons_osm.cpu().detach()
                recons_sat = recons_sat.cpu().detach()

                invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

                sat = invTrans(sat)
                osm = invTrans(osm)
                recons_osm = invTrans(recons_osm)
                recons_sat = invTrans(recons_sat)
                masked_osm_image = invTrans(masked_osm_image)
                masked_sat_image = invTrans(masked_sat_image)
                pred_sat = invTrans(pred_sat)
                pred_osm = invTrans(pred_osm)

                print('save images')

                for idx in range(len(city)):
                    base_dir = 'fused_images/'
                    label = label
                    im_path = os.path.join(base_dir, label, 'train', str(global_step))
                    os.makedirs(im_path, exist_ok=True)
                    save_image(osm[idx], os.path.join(im_path, f"osm_{idx}.png"))
                    save_image(sat[idx], os.path.join(im_path, f"sat_{idx}.png"))
                    save_image(recons_osm[idx], os.path.join(im_path, f"recons_osm_{idx}.png"))
                    save_image(recons_sat[idx], os.path.join(im_path, f"recons_sat_{idx}.png"))
                    save_image(masked_osm_image[idx], os.path.join(im_path, f"masked_osm_{idx}.png"))
                    save_image(pred_osm[idx], os.path.join(im_path, f"pred_osm_{idx}.png"))
                    save_image(masked_sat_image[idx], os.path.join(im_path, f"masked_sat_{idx}.png"))
                    save_image(pred_sat[idx], os.path.join(im_path, f"pred_sat_{idx}.png"))

                writer.flush()


        # scratch_path = '/work/vita/qngo'
        # model_name = 'models/VIGOR/'+label+'/' + str(epoch) + '/'
        # model_dir = os.path.join(scratch_path, model_name)
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)
        # torch.save(model.cpu().state_dict(), model_dir+'model.pt') # saving model
        model.save_multimae_weights(label, epoch)
        model.cuda() # moving model to GPU for further training
        model.eval()

        # validation
        running_loss_validation = []
        with torch.no_grad(): # fix this
            print('validation')
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
                output = model(grd, sat, osm)
                (
                    preds,
                    masks,
                    encoder_tokens
                ) = output

                recons_osm = preds['osm']
                recons_sat = preds['sat']

                if use_fusion_loss:
                    # loss_fusion, loss_in, ssim_loss, loss_grad = fusion_loss(
                    #     image_vis=osm, image_ir=sat, generate_img=fused_image, i=None, labels=None
                    # )

                    loss_osm = masked_l1_loss(recons_osm, osm, mask=masks['osm'])
                    loss_sat = masked_mse_loss(recons_sat, sat, mask=masks['sat'])

                    loss_validation = loss_osm
                    loss_validation += loss_sat

                running_loss_validation.append(loss_validation.item())

            writer.add_scalar("Loss/Validation", np.mean(running_loss_validation), epoch)
