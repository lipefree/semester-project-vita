import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from dual_datasets import VIGORDataset
from utils import process_data_augment, process_data, get_data_transforms
from registry import get_registry
from test import load_model, run_test_dataset, save_distances
import wandb
import random


def main():
    weight_ori = 1e1
    weight_infoNCE = 1e4
    use_augment = False
    ori_noise = 180
    experiment_name = "CCVPE_sat_cosine_decay"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wrapper = get_registry(experiment_name)(
        experiment_name, device, weight_infoNCE, weight_ori
    )
    if use_augment:
        experiment_name += "_use-augment"
    else:
        experiment_name += "_no-augment"

    if ori_noise == 0:
        experiment_name += "_known-orientation"
    wandb.init(project="VITA", name=experiment_name)
    check_experiment_name(experiment_name)

    dataset_root = "/work/vita/qngo/VIGOR"
    learning_rate = 1e-4
    batch_size = 8
    num_epoch = 14
    area = "samearea"
    training = True
    pos_only = True
    transform_grd, transform_sat = get_data_transforms()

    vigor = VIGORDataset(
        dataset_root,
        split=area,
        train=training,
        pos_only=pos_only,
        ori_noise=ori_noise,
        transform=(transform_grd, transform_sat),
        use_osm_tiles=True,
    )

    optimizer = torch.optim.AdamW(
        model_wrapper.get_model().parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    )
    print(f"model name {experiment_name}")
    writer = SummaryWriter(log_dir=os.path.join("runs", experiment_name))

    train_dataloader, val_dataloader = get_dataloaders(
        vigor, batch_size, debug_mode=(debug := False)
    )
    if debug:
        num_epoch *= 100
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        pct_start=0.05,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epoch,
    )

    best_epoch = train(
        num_epoch,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        model_wrapper,
        writer,
        device,
        experiment_name,
        use_augment,
        debug,
    )

    if not debug:
        print("launch test on epoch ", best_epoch)
        test(experiment_name, best_epoch, dataset_root, device, writer)


def test(
    experiment_name,
    epoch,
    dataset_root,
    device,
    tb_writer,
    area="samearea",
    pos_only=True,
):
    base_path = "/work/vita/qngo/test_results"
    if not os.path.exists(os.path.join(base_path, experiment_name)):
        os.mkdir(os.path.join(base_path, experiment_name))

    training = False
    batch_size = 16
    transform_grd, transform_sat = get_data_transforms()
    vigor = VIGORDataset(
        dataset_root,
        split=area,
        train=training,
        pos_only=pos_only,
        transform=(transform_grd, transform_sat),
        use_osm_tiles=True,
    )

    model_wrapper = get_registry(experiment_name)(experiment_name, device)
    base_model_path = "/work/vita/qngo/models/VIGOR/"
    load_model(model_wrapper, base_model_path, experiment_name, epoch=epoch)
    distances = run_test_dataset(
        experiment_name,
        dataset=vigor,
        model_wrapper=model_wrapper,
        batch_size=batch_size,
        device=device,
        base_path=base_path,
    )

    save_distances(experiment_name, distances, base_path)
    tb_writer.add_scalar("Test/mean_distance", np.mean(distances), 0)
    tb_writer.add_scalar("Test/median_distance", np.median(distances), 0)


def train(
    num_epoch,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    model_wrapper,
    tb_writer,
    device,
    experiment_name,
    use_augment,
    debug,
):
    global_step = 0
    print("train")

    best_distance = 100
    best_epoch = -1
    for epoch in range(num_epoch):
        global_step = train_step(
            train_dataloader,
            optimizer,
            scheduler,
            model_wrapper,
            device,
            global_step,
            tb_writer,
            use_augment,
        )
        mean_distance = validation_step(val_dataloader, model_wrapper, device, epoch, tb_writer)
        if not debug and mean_distance < best_distance:
            save_model(model_wrapper, epoch, experiment_name)
            best_epoch = epoch
            best_distance = mean_distance

    return best_epoch


def train_step(
    train_dataloader,
    optimizer,
    scheduler,
    model_wrapper,
    device,
    global_step,
    tb_writer,
    use_augment,
):
    steps = global_step
    model_wrapper.set_model_to_train()
    for i, data in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        if use_augment:
            augment_list = ["none", "missing_OSM", "missing_SAT", "noisy_OSM", "noisy_SAT"]
            augment = random.choices(
                population=augment_list, weights=[0.4, 0.15, 0.15, 0.15, 0.15], k=1
            )[0]
        else:
            augment = "none"
        processed_data = process_data_augment(data, device, augment)
        loss = model_wrapper.train_step(processed_data, steps, tb_writer)
        loss.backward()
        optimizer.step()
        scheduler.step()
        steps += 1
        wandb.log({"train/loss": loss})
        wandb.log({"train/lr": scheduler.get_last_lr()})
    return steps


def validation_step(val_dataloader, model_wrapper, device, epoch, tb_writer):
    """
    Note that we use a validation state that can be different from model to model.
    It is used to log everything at the end when we have finished running over the entire
    validation set.
    """
    model_wrapper.set_model_to_eval()
    validation_state = model_wrapper.init_validation_state()
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            processed_data = process_data(data, device)
            validation_state = model_wrapper.validation_step(processed_data, validation_state)

    model_wrapper.validation_log(validation_state, epoch, tb_writer)
    mean_distance = np.mean(validation_state["distance"])
    wandb.log({"validation/mean_distance": mean_distance})

    return mean_distance


def save_model(model_wrapper, epoch, experiment_name):
    model = model_wrapper.get_model()
    scratch_path = "/work/vita/qngo"
    model_name = "models/VIGOR/" + experiment_name + "/" + str(epoch) + "/"
    model_dir = os.path.join(scratch_path, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.cpu().state_dict(), model_dir + "model.pt")  # saving model
    model.cuda()  # moving model to GPU for further training


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


def get_dataloaders(vigor_dataset: VIGORDataset, batch_size: int, debug_mode: bool = False):
    """
    debug mode is used to run overfit test and see if the whole training loop is not crashing
    """
    vigor = vigor_dataset
    index_list = np.arange(vigor.__len__())
    np.random.shuffle(index_list)
    train_indices = index_list[0 : int(len(index_list) * 0.8)]
    val_indices = index_list[int(len(index_list) * 0.8) :]

    if debug_mode:
        train_indices = index_list[0:10]
        val_indices = index_list[10:15]

    training_set = Subset(vigor, train_indices)
    val_set = Subset(vigor, val_indices)
    train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def check_for_nan_params(model):
    nan_found = False
    for name, param in model.named_parameters():
        if param is None:
            continue
        if torch.isnan(param).any():
            print(f"NaN detected in parameter: {name}")
            nan_found = True
    if not nan_found:
        print("No NaNs in any parameters.")


if __name__ == "__main__":
    main()
