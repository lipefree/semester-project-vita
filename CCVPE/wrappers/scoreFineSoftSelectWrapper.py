from score_fine_patch_score_match_fusion_model import CVM_VIGOR as CVM
from losses import loss_ccvpe
from model_utils.fused_image_loss import Fusionloss
from training_utils import get_location, get_meter_distance, get_orientation_distance
import torch
import numpy as np
import os
from torchvision.utils import save_image
from torchvision import transforms
import random


class ScoreFinePatchDAFWrapper:
    def __init__(
        self,
        experiment_name,
        device,
        weight_infoNCE=1e4,
        weight_ori=1e1,
        use_fusion_loss=False,
        only_sat=False,
        random=False,
    ):
        self.model = CVM(
            device,
            circular_padding=True,
            use_adapt=False,
            use_concat=False,
            use_mlp=False,
        ).to(device)
        self.weight_infoNCE = weight_infoNCE
        self.weight_ori = weight_ori
        self.fusion_loss = Fusionloss()
        self.running_loss = 0
        self.experiment_name = experiment_name

        self.use_fusion_loss = use_fusion_loss
        self.only_sat = only_sat
        self.random = random

    def train_step(self, data, global_step, writer):
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        output, losses, _ = self.infer(data, training=True)

        (
            alpha,
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

        if global_step % 10 == 0:
            self.log_loss("Train", global_step, writer, *losses)

        if global_step % 200 == 0:
            self.log_metric(data, output, global_step, writer)

        return losses[0]

    def infer(self, data, training=False):
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data

        if self.only_sat:
            osm = torch.zeros_like(osm)
        elif self.random and training:
            osm_z = torch.zeros_like(osm)
            sat_z = torch.zeros_like(sat)
            choices = [[sat, osm], [sat_z, osm], [sat, osm_z]]
            choice = random.choice(choices)
            sat, osm = choice[0], choice[1]

        output = self.model(grd, sat, osm)

        (
            alpha,
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

        losses = self.compute_loss(output[1:], gt, gt_orientation, gt_with_ori, osm, sat)
        return output, losses, heatmap

    def set_model_to_train(self):
        self.model.train()

    def set_model_to_eval(self):
        self.model.eval()

    def validation_step(self, data, validation_state):
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        output, losses, heatmap = self.infer(data)

        losses = self.compute_loss(output[1:], gt, gt_orientation, gt_with_ori, osm, sat)

        validation_state["loss"].append(losses[0].item())
        validation_state["loss_ce"].append(losses[1].item())
        validation_state["loss_infonce"].append(losses[2].item())
        validation_state["loss_ori"].append(losses[3].item())

        distance, orientation_error = self.get_distances_ori(
            output, gt, gt_with_ori, gt_orientation, city
        )
        validation_state["distance"].extend(distance)
        validation_state["orientation_error"].extend(orientation_error)

        return validation_state

    def validation_log(self, validation_state, epoch, writer):
        loss = np.mean(validation_state["loss"])
        loss_ce = np.mean(validation_state["loss_ce"])
        loss_infonce = np.mean(validation_state["loss_infonce"])
        loss_ori = np.mean(validation_state["loss_ori"])
        loss_fusion = np.mean(validation_state["loss_image"])

        self.log_loss("Validation", epoch, writer, loss, loss_ce, loss_infonce, loss_ori)

        writer.add_scalar("Validation/mean_distance", np.mean(validation_state["distance"]), epoch)
        writer.add_scalar(
            "Validation/median_distance", np.median(validation_state["distance"]), epoch
        )
        writer.add_scalar(
            "Validation/mean_orientation_error",
            np.mean(validation_state["orientation_error"]),
            epoch,
        )

        writer.add_scalar(
            "Validation/median_orientation_error",
            np.median(validation_state["orientation_error"]),
            epoch,
        )

        writer.flush()

    def get_distances_ori(self, output, gt, gt_with_ori, gt_orientation, city):
        distances = []
        orientation_error = []
        (
            alpha,
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
            meter_distance = get_meter_distance(loc_gt, loc_pred, city[batch_idx], batch_idx)
            distances.append(meter_distance)

            orientation_distance = get_orientation_distance(
                gt_orientation, ori, loc_gt, loc_pred, batch_idx
            )

            if orientation_distance is not None:
                orientation_error.append(orientation_distance)

        return distances, orientation_error

    def init_validation_state(self):
        return {
            "loss": [],
            "loss_ce": [],
            "loss_infonce": [],
            "loss_ori": [],
            "loss_image": [],
            "distance": [],
            "orientation_error": [],
        }

    def compute_loss(self, output, gt, gt_orientation, gt_with_ori, osm, sat):
        (
            # alpha,
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
            output,
            gt,
            gt_orientation,
            gt_with_ori,
            self.weight_infoNCE,
            self.weight_ori,
        )

        return loss, loss_ce, loss_infonce, loss_ori

    def log_loss(self, stage_name, global_step, writer, loss, loss_ce, loss_infonce, loss_ori):
        writer.add_scalar(f"{stage_name}/loss_ce", loss_ce, global_step)
        writer.add_scalar(f"{stage_name}/loss_infonce", loss_infonce, global_step)
        writer.add_scalar(f"{stage_name}/loss_ori", loss_ori, global_step)
        writer.add_scalar(f"Loss/{stage_name}", loss, global_step)

    def log_metric(self, data, output, global_step, writer):
        print("log metric")
        grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data
        (
            weights,
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
            meter_distance = get_meter_distance(loc_gt, loc_pred, city[batch_idx], batch_idx)
            distance.append(meter_distance)

            orientation_distance = get_orientation_distance(
                gt_orientation, ori, loc_gt, loc_pred, batch_idx
            )

            if orientation_distance is not None:
                orientation_error.append(orientation_distance)

        t1, t2 = weights.unbind(dim=-1)
        mean_choice = (t1 - t2).mean().cpu().detach().numpy()
        writer.add_scalar("Train/mean_distance", np.mean(distance), global_step)
        writer.add_scalar("Train/median_distance", np.median(distance), global_step)
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

        writer.add_scalar("Train/mean_choice", mean_choice, global_step)

        self.running_loss = 0.0
        writer.flush()

    def get_model(self):
        return self.model

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
