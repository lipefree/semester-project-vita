import torch
import os
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from wrappers.DAFWrapper import DAFWrapper
from dual_datasets import VIGORDataset
from utils import get_data_transforms, process_data
from tqdm import tqdm
import numpy as np
from registry import get_registry


base_path = '/work/vita/qngo/test_results'
# base_path = '/work/vita/qngo/test/test_results'

def main():
    # experiment_names = [('hard_select_fusion', 6), ('sat_hard_select_fusion', 7), ('random_score_matching_fusion_rerun', 10), ('score_matching_fusion_rerun', 7)]
    experiment_names = [('fine_hard_select_fusion', 8)]
    
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

    for experiment_name, epoch in experiment_names:
        print(f'model name {experiment_name}')
        print('at epoch ', epoch)
        if not os.path.exists(os.path.join(base_path, experiment_name)):
            os.mkdir(os.path.join(base_path, experiment_name))

        writer = SummaryWriter(log_dir=os.path.join('runs', f'{experiment_name}'))
        model_wrapper = get_registry(experiment_name)(experiment_name, device)

        base_model_path = "/work/vita/qngo/models/VIGOR/"
        load_model(model_wrapper, base_model_path, experiment_name, epoch=epoch)
        distances = run_test_dataset(experiment_name,
                         dataset=vigor, 
                         model_wrapper=model_wrapper, 
                         batch_size=batch_size,
                         device=device,
                         base_path=base_path,
                         debug=(debug:=False))
        
        save_distances(experiment_name, distances, base_path)
        writer.add_scalar("Test/mean_distance", 
                          np.mean(distances), 
                          0)
        writer.add_scalar("Test/median_distance", 
                          np.median(distances), 
                          0)


def run_test_dataset(experiment_name, dataset, model_wrapper, batch_size, device, base_path, debug=False):
    '''
      This method became very complicated because I tried to save the heatmaps but the GPU could not allocated
      the whole tensor at once. It results in a strategy where we load off the gpu when reaching {limit_heatmaps_size}
      samples to a numpy array on CPU.

      We pre-allocate a lot of stuff because it was taking more than a day before, but with all the optimization we are
      now at around 3h with this configuration.  
    '''
    model_wrapper.set_model_to_eval()
    if debug:
        dataset = Subset(dataset, [i for i in range(102)])
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print('len dataset ', len(dataset))
    print('data loader len ', len(test_dataloader))
    limit_heatmaps_size = 200

    np_heatmaps = np.zeros((len(dataset), 512, 512))

    with torch.no_grad():
        distance_in_meters = []
        
        heatmaps = torch.zeros(limit_heatmaps_size*batch_size, 1, 512, 512, device=device)
        items_in_h = 0
        for i, data in enumerate(tqdm(test_dataloader), 0):
            processed_data = process_data(data, device)
            _, loss, heatmap = model_wrapper.infer(processed_data)
            distances = compute_distances(processed_data, heatmap)

            # offload from gpu
            distances_cpu = distances.cpu().detach().tolist()
            distance_in_meters.extend(distances_cpu)
            
            current_batch_size = heatmap.shape[0]

            heatmaps[items_in_h*batch_size: items_in_h*batch_size + current_batch_size] = heatmap
            items_in_h += 1

            if items_in_h >= limit_heatmaps_size:
                cpu_heatmaps = heatmaps.squeeze(1).cpu().detach().numpy()
                # 0, 96 (32*3), 96, 96*2
                np_heatmaps[batch_size*(i + 1 - limit_heatmaps_size): batch_size*i + current_batch_size] = cpu_heatmaps
                items_in_h = 0
                # heatmaps = torch.zeros(limit_heatmaps_size*batch_size, 1, 512, 512, device=device)
                
        # handle last remaining items
        if items_in_h > 0:
            cpu_heatmaps = heatmaps.squeeze(1).cpu().detach().numpy()
            np_heatmaps[-((items_in_h - 1)*batch_size + current_batch_size):] = cpu_heatmaps[:(items_in_h - 1)*batch_size+current_batch_size]

        heatmaps = np_heatmaps
        # save_heatmap(experiment_name, heatmaps, len(dataset), base_path)

    return distance_in_meters


def save_distances(experiment_name, distances, base_path):
    save_qual = os.path.join(base_path, experiment_name, "distance_test.npy")

    print('save distances of ', len(distances))
    print('recorded mean is ', np.mean(distances))
    with open(save_qual, "wb") as f:
        np.save(f, distances)


def load_distances(experiment_name, base_path):
    return np.load(os.path.join(base_path, experiment_name, "distance_test.npy"))


def save_heatmap(experiment_name, heatmaps, items, base_path):
    filepath = f'{base_path}/{experiment_name}/heatmaps.npz'
    append_heatmap(filepath, heatmaps, items)



def append_heatmap(npz_path, heatmaps, nbr_items):
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
