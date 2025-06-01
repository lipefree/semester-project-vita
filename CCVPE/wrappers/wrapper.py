from training_utils import get_location, get_meter_distance, get_orientation_distance
import torch


class Wrapper():
    def __init__(self, experiment_name, device):
        self.experiment_name = experiment_name
        self.device = device

    def set_model_to_train(self):
        self.model.train()

    def set_model_to_eval(self):
        self.model.eval()

    def get_model(self):
        return self.model

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def infer(self, data):
        pass

    def train_step(self, data, global_step, writer):
        pass

    def validation_step(self, data, validation_step):
        pass

    def get_distances_ori(self, output, gt, gt_with_ori, gt_orientation, city, heatmap, ori):
        distances = []
        orientation_error = []

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
