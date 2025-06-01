from ccvpe_3_fusion_model import CVM_VIGOR as CVM
from losses import loss_ccvpe
from model_utils.fused_image_loss import Fusionloss
from training_utils import get_location, get_meter_distance, get_orientation_distance
import torch
import numpy as np
import os
from torchvision.utils import save_image
from torchvision import transforms
from wrappers.wrapper import Wrapper

class DAF3Wrapper(Wrapper):
    def __init__(self, experiment_name, device, weight_infoNCE=1e4, weight_ori=1e1, alpha_type=0):
        self.model = CVM(device, circular_padding=True, 
                         use_adapt=False, 
                         use_concat=False, 
                         use_mlp=False, 
                         alpha_type=alpha_type).to(device)
        self.weight_infoNCE = weight_infoNCE
        self.weight_ori = weight_ori
        self.fusion_loss = Fusionloss()
        self.experiment_name = experiment_name

    def train_step(self, data, global_step, writer):
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        output, losses, _ = self.infer(data)

        (
            alpha,
            fused_image,
            fusion_output,
            sat_output,
            osm_output
        ) = output
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
        ) = fusion_output

        if global_step % 10 == 0:
            self.log_loss('Train', global_step, writer, *losses)

        if global_step % 100 == 0:
            self.log_metric(data, fusion_output, global_step, writer, prefix="")
            self.log_metric(data, sat_output, global_step, writer, prefix='sat/')
            self.log_metric(data, osm_output, global_step, writer, prefix='osm/')

        if global_step % 8000 == 0:
            self.log_images(osm, sat, fused_image, city, global_step)

        return losses[0] + losses[5] + losses[9]

    def infer(self, data):
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        heatmap = torch.zeros_like(gt)
        heatmap = heatmap.detach()
        output = self.model(grd, sat, osm, heatmap, timestep=0)

        (
            alpha,
            fused_image,
            fusion_output,
            sat_output,
            osm_output
        ) = output
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
        ) = fusion_output

        losses = self.compute_loss(alpha, fused_image, fusion_output, 
                                   sat_output, osm_output,
                                gt, gt_orientation, gt_with_ori, 
                                osm, sat)

        return output, losses, heatmap

    def set_model_to_train(self):
        self.model.train()

    def set_model_to_eval(self):
        self.model.eval()

    def validation_step(self, data, validation_state):
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        output, losses, heatmap = self.infer(data)
        (
            alpha,
            fused_image,
            fusion_output,
            sat_output,
            osm_output
        ) = output

        losses = self.compute_loss(alpha, fused_image, fusion_output, 
                                   sat_output, osm_output,
                                gt, gt_orientation, gt_with_ori, 
                                osm, sat)

        validation_state['loss'].append(losses[0].item())
        validation_state['loss_ce'].append(losses[1].item())
        validation_state['loss_infonce'].append(losses[2].item())
        validation_state['loss_ori'].append(losses[3].item())
        validation_state['loss_image'].append(losses[4].item())
        
        validation_state['loss_sat'].append(losses[5].item())
        validation_state['loss_ce_sat'].append(losses[6].item())
        validation_state['loss_infonce_sat'].append(losses[7].item())
        validation_state['loss_ori_sat'].append(losses[8].item())

        validation_state['loss_osm'].append(losses[9].item())
        validation_state['loss_ce_osm'].append(losses[10].item())
        validation_state['loss_infonce_osm'].append(losses[11].item())
        validation_state['loss_ori_osm'].append(losses[12].item())

        distance, orientation_error = self.get_distances_ori(fusion_output, gt, gt_with_ori, gt_orientation, city)
        validation_state['distance'].extend(distance)
        validation_state['orientation_error'].extend(orientation_error)
        
        distance_sat, orientation_error_sat = self.get_distances_ori(sat_output, gt, gt_with_ori, gt_orientation, city)
        validation_state['distance_sat'].extend(distance_sat)
        validation_state['orientation_error_sat'].extend(orientation_error_sat)
        
        distance_osm, orientation_error_osm = self.get_distances_ori(osm_output, gt, gt_with_ori, gt_orientation, city)
        validation_state['distance_osm'].extend(distance_osm)
        validation_state['orientation_error_osm'].extend(orientation_error_osm)

        return validation_state

    def validation_log(self, validation_state, epoch, writer):
        loss = np.mean(validation_state['loss'])
        loss_ce = np.mean(validation_state['loss_ce'])
        loss_infonce = np.mean(validation_state['loss_infonce'])
        loss_ori = np.mean(validation_state['loss_ori'])
        loss_fusion = np.mean(validation_state['loss_image'])
        loss_sat = np.mean(validation_state['loss_sat'])
        loss_ce_sat = np.mean(validation_state['loss_ce_sat'])
        loss_infonce_sat = np.mean(validation_state['loss_infonce_sat'])
        loss_ori_sat = np.mean(validation_state['loss_ori_sat'])
        loss_osm = np.mean(validation_state['loss_osm'])
        loss_ce_osm = np.mean(validation_state['loss_ce_osm'])
        loss_infonce_osm = np.mean(validation_state['loss_infonce_osm'])
        loss_ori_osm = np.mean(validation_state['loss_ori_osm'])

        losses = (
            loss, loss_ce, loss_infonce, loss_ori, loss_fusion,
            loss_sat, loss_ce_sat, loss_infonce_sat, loss_ori_sat,
            loss_osm, loss_ce_osm, loss_infonce_osm, loss_ori_osm
        )
        self.log_loss('Validation', epoch, writer, *losses)


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
        
        writer.add_scalar("sat/Validation/mean_distance", 
                          np.mean(validation_state['distance_sat']), 
                          epoch)
        writer.add_scalar("sat/Validation/median_distance", 
                          np.median(validation_state['distance_sat']), 
                          epoch)
        writer.add_scalar("sat/Validation/mean_orientation_error", 
            np.mean(validation_state['orientation_error_sat']), 
            epoch)
        writer.add_scalar("sat/Validation/median_orientation_error", 
                          np.median(validation_state['orientation_error_sat']), 
                          epoch)
        
        writer.add_scalar("osm/Validation/mean_distance", 
                          np.mean(validation_state['distance_osm']), 
                          epoch)
        writer.add_scalar("osm/Validation/median_distance", 
                          np.median(validation_state['distance_osm']), 
                          epoch)
        writer.add_scalar("osm/Validation/mean_orientation_error", 
            np.mean(validation_state['orientation_error_osm']), 
            epoch)

        writer.add_scalar("osm/Validation/median_orientation_error", 
                          np.median(validation_state['orientation_error_osm']), 
                          epoch)

        writer.flush()

    def get_distances_ori(self, output, gt, gt_with_ori, gt_orientation, city):
        distances = []
        orientation_error = []
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
        ) = output

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
            distances.append(meter_distance)

            orientation_distance = get_orientation_distance(
                gt_orientation, ori, loc_gt, loc_pred, batch_idx
            )

            if orientation_distance is not None:
                orientation_error.append(orientation_distance)

        return distances, orientation_error

    def init_validation_state(self):
        return {
            'loss': [],
            'loss_ce': [],
            'loss_infonce': [],
            'loss_ori': [],
            'loss_image': [],
            'loss_sat': [],
            'loss_ce_sat': [],
            'loss_infonce_sat': [],
            'loss_ori_sat': [],
            'loss_image_sat': [],
            'loss_osm': [],
            'loss_ce_osm': [],
            'loss_infonce_osm': [],
            'loss_ori_osm': [],
            'loss_image_osm': [],
            'distance' : [],
            'orientation_error': [],
            'distance_sat' : [],
            'orientation_error_sat': [],
            'distance_osm' : [],
            'orientation_error_osm': []
        }

    def compute_loss(self, alpha, fused_image, fused_output, sat_output, osm_output, gt, gt_orientation, gt_with_ori, osm, sat):

        loss, loss_ce, loss_infonce, loss_ori = loss_ccvpe(
            fused_output, gt, gt_orientation, gt_with_ori, self.weight_infoNCE, self.weight_ori
        )
        loss_osm, loss_ce_osm, loss_infonce_osm, loss_ori_osm = loss_ccvpe(
            osm_output, gt, gt_orientation, gt_with_ori, self.weight_infoNCE, self.weight_ori
        )
        loss_sat, loss_ce_sat, loss_infonce_sat, loss_ori_sat = loss_ccvpe(
            sat_output, gt, gt_orientation, gt_with_ori, self.weight_infoNCE, self.weight_ori
        )

        loss_fusion, loss_in, ssim_loss, loss_grad = self.fusion_loss(
            image_vis=osm, image_ir=sat, generate_img=fused_image, i=None, labels=None
        )

        return (loss, loss_ce, loss_infonce, loss_ori, loss_fusion,
            loss_sat, loss_ce_sat, loss_infonce_sat, loss_ori_sat,
            loss_osm, loss_ce_osm, loss_infonce_osm, loss_ori_osm)

    def log_loss(self, stage_name, global_step, writer, 
                loss, loss_ce, loss_infonce, loss_ori, loss_fusion,
            loss_sat, loss_ce_sat, loss_infonce_sat, loss_ori_sat,
            loss_osm, loss_ce_osm, loss_infonce_osm, loss_ori_osm ):
        
        writer.add_scalar(f"{stage_name}/loss_ce", loss_ce, global_step)
        writer.add_scalar(f"{stage_name}/loss_infonce", loss_infonce, global_step)
        writer.add_scalar(f"{stage_name}/loss_ori", loss_ori, global_step)
        writer.add_scalar(f"{stage_name}/loss_fusion", loss_fusion, global_step)
        writer.add_scalar(f"{stage_name}/loss_ce", loss_ce, global_step)
        
        writer.add_scalar(f"sat/{stage_name}/loss_infonce", loss_infonce_sat, global_step)
        writer.add_scalar(f"sat/{stage_name}/loss_ori", loss_ori_sat, global_step)
        writer.add_scalar(f"sat/{stage_name}/loss_ce", loss_ce_sat, global_step)
        
        writer.add_scalar(f"osm/{stage_name}/loss_infonce", loss_infonce_osm, global_step)
        writer.add_scalar(f"osm/{stage_name}/loss_ori", loss_ori_osm, global_step)
        writer.add_scalar(f"osm/{stage_name}/loss_ce", loss_ce_osm, global_step)
        writer.add_scalar(f"Loss/{stage_name}", loss, global_step)

    def log_metric(self, data, output, global_step, writer, prefix):
        print('log metric')
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
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

        writer.add_scalar(f"{prefix}Train/mean_distance", np.mean(distance), global_step)
        writer.add_scalar(
            f"{prefix}Train/median_distance", np.median(distance), global_step
        )
        writer.add_scalar(
            f"{prefix}Train/mean_orientation_error",
            np.mean(orientation_error),
            global_step,
        )
        writer.add_scalar(
            f"{prefix}Train/median_orientation_error",
            np.median(orientation_error),
            global_step,
        )

        writer.flush()

    def log_images(self, osm, sat, fused_image, city, global_step):
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
            label = self.experiment_name
            im_path = os.path.join(base_dir, label, 'train', str(global_step))
            os.makedirs(im_path, exist_ok=True)
            save_image(osm[idx], os.path.join(im_path, f"osm_{idx}.png"))
            save_image(sat[idx], os.path.join(im_path, f"sat_{idx}.png"))
            save_image(fused_image[idx], os.path.join(im_path, f"fused_image_{idx}.png"))

    def get_model(self):
        return self.model

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

