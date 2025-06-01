from multimae_model import CVM_VIGOR as CVM
from losses import loss_ccvpe
from model_utils.fused_image_loss import Fusionloss
from training_utils import get_location, get_meter_distance, get_orientation_distance
import torch
import numpy as np
import os
from torchvision.utils import save_image
from torchvision import transforms
from model_utils.criterion import MaskedMSELoss, MaskedL1Loss
from wrappers.wrapper import Wrapper

class MultiMAEWrapper(Wrapper):
    def __init__(self, experiment_name, device, weight_infoNCE=1e4, weight_ori=1e1, 
                 use_reconstruction_loss=False, patch_size=32, use_sat_features=True):
        self.model = CVM(device, circular_padding=True, 
                         use_adapt=False, 
                         use_concat=False, 
                         use_mlp=False, use_sat_features=use_sat_features).to(device)
        self.weight_infoNCE = weight_infoNCE
        self.weight_ori = weight_ori
        self.running_loss = 0
        self.experiment_name = experiment_name
        self.masked_mse_loss = MaskedMSELoss(patch_size=patch_size)
        self.masked_l1_loss = MaskedL1Loss(patch_size=patch_size)
        self.use_reconstruction_loss = use_reconstruction_loss

    def train_step(self, data, global_step, writer):
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        output, losses, _ = self.infer(data)

        (
            recons_osm,
            recons_sat,
            masks,
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

        self.running_loss += losses[0]

        if global_step % 1 == 0:
            self.log_loss('Train', global_step, writer, *losses)

        if global_step % 1 == 0:
            self.log_metric(data, output, global_step, writer)

        if global_step % 5000 == 0:
            self.log_images(osm, sat, recons_osm, recons_sat, city, global_step)

        return losses[0]

    def infer(self, data):
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        heatmap = torch.zeros_like(gt)
        heatmap = heatmap.detach()
        output = self.model(grd, sat, osm)

        (
            recons_osm,
            recons_sat,
            masks,
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


        losses = self.compute_loss(output, gt, gt_orientation, gt_with_ori, osm, sat)

        return output, losses, heatmap

    def set_model_to_train(self):
        self.model.train()

    def set_model_to_eval(self):
        self.model.eval()

    def validation_step(self, data, validation_state):
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        output, losses, heatmap = self.infer(data)
        (
            recons_osm,
            recons_sat,
            masks,
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

        losses = self.compute_loss(output, gt, gt_orientation, gt_with_ori, osm, sat)

        validation_state['loss'].append(losses[0].item())
        validation_state['loss_ce'].append(losses[1].item())
        validation_state['loss_infonce'].append(losses[2].item())
        validation_state['loss_ori'].append(losses[3].item())
        validation_state['loss_sat'].append(losses[4].item())
        validation_state['loss_osm'].append(losses[5].item())

        distance, orientation_error = self.get_distances_ori(output, gt, gt_with_ori, gt_orientation, city, heatmap, ori)
        validation_state['distance'].extend(distance)
        validation_state['orientation_error'].extend(orientation_error)

        return validation_state

    def validation_log(self, validation_state, epoch, writer):
        loss = np.mean(validation_state['loss'])
        loss_ce = np.mean(validation_state['loss_ce'])
        loss_infonce = np.mean(validation_state['loss_infonce'])
        loss_ori = np.mean(validation_state['loss_ori'])
        loss_osm = np.mean(validation_state['loss_osm'])
        loss_sat = np.mean(validation_state['loss_sat'])

        self.log_loss('Validation', epoch,
                        writer, loss, loss_ce, loss_infonce,
                        loss_ori, loss_sat, loss_osm)

        writer.add_scalar("Validation/mean_distance", 
                          np.mean(validation_state['distance']), 
                          epoch)
        writer.add_scalar("Validation/median_distance", 
                          np.median(validation_state['distance']), 
                          epoch)
        writer.add_scalar("Validation/mean_orientation_error", 
            np.mean(validation_state['orientation_error']), 
            epoch)

        writer.add_scalar("Validation/median_orientation_error", 
                          np.median(validation_state['orientation_error']), 
                          epoch)

        writer.flush()

    def init_validation_state(self):
        return {
            'loss': [],
            'loss_ce': [],
            'loss_infonce': [],
            'loss_ori': [],
            'distance' : [],
            'orientation_error': [],
            'loss_sat': [],
            'loss_osm': []
        }

    def compute_loss(self, output, gt, gt_orientation, gt_with_ori, osm, sat):
        (   recons_osm,
            recons_sat,
            masks,
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

        loss, loss_ce, loss_infonce, loss_ori = loss_ccvpe(
            output[3:], gt, gt_orientation, gt_with_ori, self.weight_infoNCE, self.weight_ori
        )

        loss_osm = self.masked_mse_loss(recons_osm.float(), osm, mask=None)
        loss_sat = self.masked_mse_loss(recons_sat.float(), sat, mask=None)

        if self.use_reconstruction_loss:
            loss += (loss_osm + loss_sat)

        return loss, loss_ce, loss_infonce, loss_ori, loss_sat, loss_osm

    def log_loss(self, stage_name, global_step, writer, 
                 loss, loss_ce, loss_infonce, loss_ori,
                 loss_sat, loss_osm):
        writer.add_scalar(f"{stage_name}/loss_ce", loss_ce, global_step)
        writer.add_scalar(f"{stage_name}/loss_infonce", loss_infonce, global_step)
        writer.add_scalar(f"{stage_name}/loss_ori", loss_ori, global_step)
        writer.add_scalar(f"{stage_name}/loss_sat_recons", loss_sat, global_step)
        writer.add_scalar(f"{stage_name}/loss_osm_recons", loss_osm, global_step)
        writer.add_scalar(f"Loss/{stage_name}", loss, global_step)

    def log_metric(self, data, output, global_step, writer):
        print('log metric')
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        (   recons_osm,
            recons_sat,
            masks,
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

        self.running_loss = 0.0
        writer.flush()

    def log_images(self, osm, sat, recons_osm, recons_sat, city, global_step):
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

        for idx in range(len(city)):
            base_dir = 'fused_images/'
            label = self.experiment_name
            im_path = os.path.join(base_dir, label, 'train', str(global_step))
            os.makedirs(im_path, exist_ok=True)
            save_image(osm[idx], os.path.join(im_path, f"osm_{idx}.png"))
            save_image(sat[idx], os.path.join(im_path, f"sat_{idx}.png"))
            save_image(recons_osm[idx], os.path.join(im_path, f"recons_osm_{idx}.png"))
            save_image(recons_sat[idx], os.path.join(im_path, f"recons_sat_{idx}.png"))

    def get_model(self):
        return self.model

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
