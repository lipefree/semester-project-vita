import torch
import os
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from wrappers.DAFWrapper import DAFWrapper
from dual_datasets import VIGORDataset
from utils import get_data_transforms, process_data_augment
from tqdm import tqdm
import numpy as np
from registry import get_registry


base_path = "/work/vita/qngo/test_results"
# base_path = '/work/vita/qngo/test/test_results'


def main():
    experiment_names = [
        ("soft_patch_DAF_v3_push_perf", 8),
        ("score_matching_fusion_rerun", 12),
        ("soft_select_fusion_v3", 7),
    ]

    dataset_root = "/work/vita/qngo/VIGOR"
    batch_size = 64
    fov = 360
    ori_noise = 180
    use_augment = True  # a little bit missleading, if the model was trained with augmentation then True, it is not about using augment during test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    circular_padding = True  # apply circular padding along the horizontal direction in the ground feature extractor
    area = "samearea"
    training = False  # test dataset
    pos_only = True
    transform_grd, transform_sat = get_data_transforms()

    vigor = VIGORDataset(
        dataset_root,
        split=area,
        train=training,
        pos_only=pos_only,
        transform=(transform_grd, transform_sat),
        use_osm_tiles=True,
        ori_noise=ori_noise,
    )

    augments = ["missing_OSM", "missing_SAT", "noisy_OSM", "noisy_SAT"]

    for augment_type in augments:
        for experiment_name, epoch in experiment_names:
            mean_distances = []
            median_distances = []
            for i in range(1):
                print(f"model name {experiment_name}")
                print("at epoch ", epoch)
                if not os.path.exists(os.path.join(base_path, experiment_name)):
                    os.mkdir(os.path.join(base_path, experiment_name))

                current_experiment_name = experiment_name
                model_wrapper = get_registry(current_experiment_name)(
                    current_experiment_name, device
                )
                if use_augment:
                    current_experiment_name += "_use-augment"
                else:
                    current_experiment_name += "_no-augment"

                if ori_noise == 0:
                    current_experiment_name += "_known-orientation"

                base_model_path = "/work/vita/qngo/models/VIGOR/"
                load_model(model_wrapper, base_model_path, current_experiment_name, epoch=epoch)
                writer_experiment_name = f"{current_experiment_name}_{augment_type}"

                debug = False
                if debug:
                    writer_experiment_name += "_debug"
                writer = SummaryWriter(
                    log_dir=os.path.join("runs", f"{check_experiment_name(writer_experiment_name)}")
                )
                distances = run_test_dataset(
                    current_experiment_name,
                    dataset=vigor,
                    model_wrapper=model_wrapper,
                    batch_size=batch_size,
                    device=device,
                    base_path=base_path,
                    debug=debug,
                    augment_type=augment_type,
                )

                # save_distances(experiment_name, distances, base_path)
                writer.add_scalar(f"Test/mean_distance_{i}", np.mean(distances), 0)
                writer.add_scalar(f"Test/median_distance_{i}", np.median(distances), 0)
                mean_distances.append(np.mean(distances))
                median_distances.append(np.median(distances))

            writer.add_scalar(f"Test/mean_distance_mean", np.mean(mean_distances), 0)
            writer.add_scalar(f"Test/mean_distance_std", np.std(mean_distances), 0)
            writer.add_scalar(f"Test/median_distance_mean", np.mean(median_distances), 0)
            writer.add_scalar(f"Test/median_distance_str", np.std(median_distances), 0)


def run_test_dataset(
    experiment_name,
    dataset,
    model_wrapper,
    batch_size,
    device,
    base_path,
    debug=False,
    augment_type="none",
):
    """
    This method became very complicated because I tried to save the heatmaps but the GPU could not allocated
    the whole tensor at once. It results in a strategy where we load off the gpu when reaching {limit_heatmaps_size}
    samples to a numpy array on CPU.

    We pre-allocate a lot of stuff because it was taking more than a day before, but with all the optimization we are
    now at around 3h with this configuration.
    """
    model_wrapper.set_model_to_eval()
    if debug:
        dataset = Subset(dataset, [i for i in range(102)])
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("len dataset ", len(dataset))
    print("data loader len ", len(test_dataloader))

    with torch.no_grad():
        distance_in_meters = []

        for i, data in enumerate(tqdm(test_dataloader), 0):
            processed_data = process_data_augment(data, device, augment_type)
            _, loss, heatmap = model_wrapper.infer(processed_data)
            distances = compute_distances(processed_data, heatmap)

            # offload from gpu
            distances_cpu = distances.cpu().detach().tolist()
            distance_in_meters.extend(distances_cpu)

    return distance_in_meters


def save_distances(experiment_name, distances, base_path):
    save_qual = os.path.join(base_path, experiment_name, "distance_test.npy")

    print("save distances of ", len(distances))
    print("recorded mean is ", np.mean(distances))
    with open(save_qual, "wb") as f:
        np.save(f, distances)


def load_distances(experiment_name, base_path):
    return np.load(os.path.join(base_path, experiment_name, "distance_test.npy"))


def save_heatmap(experiment_name, heatmaps, items, base_path):
    filepath = f"{base_path}/{experiment_name}/heatmaps.npz"
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
        (row_gt.float() - row_pred.float()).pow(2) + (col_gt.float() - col_pred.float()).pow(2)
    )

    # city → scale mapping
    scale_map = {
        "NewYork": 0.113248 / 512 * 640,
        "Seattle": 0.100817 / 512 * 640,
        "SanFrancisco": 0.118141 / 512 * 640,
        "Chicago": 0.111262 / 512 * 640,
    }
    scales = torch.tensor(
        [scale_map[c] for c in city], device=pixel_distances.device, dtype=pixel_distances.dtype
    )

    meter_distances = pixel_distances * scales
    return meter_distances


def load_model(model_wrapper, base_model_path: str, model_name: str, epoch: int | str):
    epoch = str(epoch)
    model_path = os.path.join(base_model_path, model_name, epoch, "model.pt")
    model_wrapper.load_model(model_path)


def check_experiment_name(experiment_name: str):
    """
    When not careful, using the same experiment name will overwrite tensorboard runs.
    Additionnaly, with bad practice, we lose track of debug runs

    We adopt the following policy:
        if a name has 'debug' in it, it can collide with other experiment. This way it is also
        easier to just rm *_debug to clean up our runs

        if the name does not contain 'debug' and collides then we crash the run
    """
    if os.path.exists(os.path.join("runs", experiment_name)) and "debug" not in experiment_name:
        raise Exception(f"name already taken {experiment_name}")
    return experiment_name


if __name__ == "__main__":
    main()
