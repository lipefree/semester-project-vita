import torch
import os
from torch.utils.data import DataLoader, Subset
from wrappers.DAFWrapper import DAFWrapper
from dual_datasets import VIGORDataset
from utils import get_data_transforms, process_data
from tqdm import tqdm
import numpy as np
from registry import get_registry


base_path = '/work/vita/qngo/test_results'
# base_path = '/work/vita/qngo/test/test_results'

def main():
    experiment_names = ['fused_image_ccvpe_alpha2', 'fused_image_ccvpe_alpha1', 'fused_image_ccvpe_alpha0']

    dataset_root='/work/vita/qngo/VIGOR'
    batch_size = 32
    fov = 360
    ori_noise = 18 * (180 // 18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    circular_padding = True # apply circular padding along the horizontal direction in the ground feature extractor
    area = 'samearea'
    training = False # test dataset
    pos_only = True
    transform_grd, transform_sat = get_data_transforms()

    vigor = VIGORDataset(dataset_root, 
                     split=area, 
                     train=training, 
                     pos_only=pos_only, 
                     transform=(transform_grd, transform_sat),
                     use_osm_tiles=True)

    for experiment_name in experiment_names:
        print(f'model name {experiment_name}')
        if not os.path.exists(os.path.join(base_path, experiment_name)):
            os.mkdir(os.path.join(base_path, experiment_name))

        model_wrapper = get_registry(experiment_name)(experiment_name, device)

        base_model_path = "/work/vita/qngo/models/VIGOR/"
        load_model(model_wrapper, base_model_path, experiment_name, epoch=5)
        distances = run_test_dataset(experiment_name,
                         dataset=vigor, 
                         model_wrapper=model_wrapper, 
                         batch_size=batch_size,
                         device=device)

        save_distances(experiment_name, distances)


def run_test_dataset(experiment_name, dataset, model_wrapper, batch_size, device):
    model_wrapper.set_model_to_eval()
    # dataset = Subset(dataset, [i for i in range(201)])
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print('data loader len ', len(test_dataloader))
    current_idx = 0
    limit_heatmaps_size = len(dataset)
    with torch.no_grad():
        distance_in_meters = []
        
        # heatmaps = np.zeros((limit_heatmaps_size, 512, 512))
        items_in_h = 0
        for i, data in enumerate(tqdm(test_dataloader), 0):
            processed_data = process_data(data, device)
            _, loss, heatmap = model_wrapper.infer(processed_data)
            distances = compute_distances(processed_data, heatmap)

            # offload from gpu
            distances_cpu = distances.cpu().detach().tolist()
            distance_in_meters.extend(distances_cpu)

            # save heatmaps
        #     heatmap = heatmap.cpu().detach().numpy()
        #     current_idx, heatmaps, items_in_h = save_heatmap(experiment_name, current_idx, heatmap, heatmaps, limit_heatmaps_size, items_in_h)

        # save_heatmap(experiment_name, current_idx, heatmap, heatmaps, limit_heatmaps_size, items_in_h, flush=True)
    return distance_in_meters


def save_distances(experiment_name, distances):
    save_qual = os.path.join(base_path, experiment_name, "distance_test2.npy")

    print('save distances of ', len(distances))
    print('recorded mean is ', np.mean(distances))
    with open(save_qual, "wb") as f:
        np.save(f, distances)


def save_heatmap(experiment_name, current_idx, heatmap, heatmaps, limit_heatmaps_size, items_in_h, flush=False):
    filepath = f'{base_path}/{experiment_name}/heatmaps.npz'

    items = items_in_h
    idx = current_idx

    if not flush:
        batch_size = heatmap.shape[0]
        for batch_idx in range(batch_size):
            heatmaps[idx] = heatmap[batch_idx]
            # append_heatmap(filepath, heatmap[batch_idx], idx)
            idx += 1
            items += 1

    if len(heatmaps) >= limit_heatmaps_size or flush:
        append_heatmap(filepath, heatmaps, items, idx)
        heatmaps = np.zeros((limit_heatmaps_size, 512, 512))

    return idx, heatmaps, items


def append_heatmap(npz_path, heatmaps, nbr_items, current_idx):
    if os.path.exists(npz_path):
        # np.load with allow_pickle=False returns an NpzFile
        existing = dict(np.load(npz_path, allow_pickle=False))
    else:
        existing = {}

    for idx in range(nbr_items):
        key = f"heatmap_{idx}"
        heatmap = heatmaps[idx]
        existing[key] = heatmap

    np.savez(npz_path, **existing)


def compute_distances(data, heatmap):
    grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened = data

    B, C, H, W = gt.shape
    idx_gt = gt.view(B, -1).argmax(dim=1)
    idx_pred = heatmap.view(B, -1).argmax(dim=1)

    c_gt, row_gt, col_gt = torch.unravel_index(idx_gt, (C, H, W))
    c_pred, row_pred, col_pred = torch.unravel_index(idx_pred, (C, H, W))

    # distances in pixels
    pixel_distances = torch.sqrt(
        (row_gt.float()  - row_pred.float()).pow(2) +
        (col_gt.float()  - col_pred.float()).pow(2)
    )

    # city → scale mapping
    scale_map = {
        'NewYork':      0.113248/512*640,
        'Seattle':      0.100817/512*640,
        'SanFrancisco': 0.118141/512*640,
        'Chicago':      0.111262/512*640,
    }
    scales = torch.tensor([scale_map[c] for c in city],
                          device=pixel_distances.device,
                          dtype=pixel_distances.dtype)

    meter_distances = pixel_distances * scales
    return meter_distances


def load_model(model_wrapper, base_model_path: str, model_name: str, epoch: int):
    epoch = str(epoch)
    model_path = os.path.join(base_model_path, model_name, epoch, "model.pt")
    model_wrapper.load_model(model_path)


if __name__ == "__main__":
    main()
