import os
from torch.utils.data import Dataset
import PIL.Image
import torch
import numpy as np
import random
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
import math

import gzip
import pickle
from torchvision import transforms
from osm_tiles_helper import project_to_n
from maploc.osm.viz import Colormap

torch.manual_seed(17)
np.random.seed(0)

# ---------------------------------------------------------------------------------
# VIGOR

class VIGORDataset(Dataset):
    def __init__(
        self,
        root,
        label_root="splits_new",
        split="samearea",
        train=True,
        transform=None,
        pos_only=True,
        ori_noise=180,
        random_orientation=None,
        use_osm_tiles=False,
        use_50_n_osm_tiles=False,
        use_osm_rendered=False,
        use_concat=False,
    ):
        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.pos_only = pos_only
        self.ori_noise = ori_noise
        self.random_orientation = random_orientation
        self.use_osm_tiles = use_osm_tiles  # If using osm tiles instead of sat images TODO: rework this to a more modular way
        self.use_50_n_osm_tiles = use_50_n_osm_tiles  # If we want to transform the osm tiles to 50 layers (corresponding to the 50 classes for OSM objects) TODO: this is really a bad way to do it
        self.use_rendered_tiles = use_osm_rendered
        self.use_concat = use_concat

        if transform != None:
            self.grdimage_transform = transform[0]
            self.satimage_transform = transform[1]

        if self.split == "samearea":
            self.city_list = ["NewYork", "Seattle", "SanFrancisco", "Chicago"]
        elif self.split == "crossarea":
            if self.train:
                self.city_list = ["NewYork", "Seattle"]
            else:
                self.city_list = ["SanFrancisco", "Chicago"]

        # load sat list and OSM tiles
        self.sat_list = []
        self.sat_index_dict = {}

        self.osm_tiles = []

        idx = 0
        for city in self.city_list:

            # load pickle file for given city
            if self.use_osm_tiles:
                osm_tile_path = os.path.join(
                    self.root, city, "osm_tiles", "data.npy"
                )
                
                with open(osm_tile_path, 'rb') as f:
                    loaded_data = np.load(f)
                    
                # with gzip.open(osm_tile_path, "rb") as f:
                #     loaded_data = pickle.load(f)

                print(f"loaded data pikle")
                self.osm_tiles.extend(loaded_data)
                print(f"osm tiles loaded for {city}")

            sat_list_fname = os.path.join(
                self.root, label_root, city, "satellite_list.txt"
            )
            with open(sat_list_fname, "r") as file:
                for line in file.readlines():
                    self.sat_list.append(
                        os.path.join(
                            self.root, city, "satellite", line.replace("\n", "")
                        )
                    )
                    self.sat_index_dict[line.replace("\n", "")] = idx
                    idx += 1
            print("InputData::__init__: load", sat_list_fname, idx)
            
        self.sat_list = np.array(self.sat_list)
        self.sat_data_size = len(self.sat_list)
        print("Sat loaded, data size:{}".format(self.sat_data_size))

        # load grd list
        self.grd_list = []
        self.label = []
        self.sat_cover_dict = {}
        self.delta = []
        idx = 0
        for city in self.city_list:
            # load grd panorama list
            if self.split == "samearea":
                if self.train:
                    label_fname = os.path.join(
                        self.root, self.label_root, city, "same_area_balanced_train.txt"
                    )
                else:
                    label_fname = os.path.join(
                        self.root, label_root, city, "same_area_balanced_test.txt"
                    )
            elif self.split == "crossarea":
                label_fname = os.path.join(
                    self.root, self.label_root, city, "pano_label_balanced.txt"
                )

            with open(label_fname, "r") as file:
                for line in file.readlines():
                    data = np.array(line.split(" "))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.sat_index_dict[data[i]])
                    label = np.array(label).astype(int)
                    delta = np.array(
                        [data[2:4], data[5:7], data[8:10], data[11:13]]
                    ).astype(float)
                    self.grd_list.append(
                        os.path.join(self.root, city, "panorama", data[0])
                    )
                    self.label.append(label)
                    self.delta.append(delta)
                    if not label[0] in self.sat_cover_dict:
                        self.sat_cover_dict[label[0]] = [idx]
                    else:
                        self.sat_cover_dict[label[0]].append(idx)
                    idx += 1
            print("InputData::__init__: load ", label_fname, idx)
        self.data_size = len(self.grd_list)
        print("Grd loaded, data size:{}".format(self.data_size))
        self.label = np.array(self.label)
        self.delta = np.array(self.delta)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # full ground panorama
        try:
            grd = PIL.Image.open(os.path.join(self.grd_list[idx]))
            grd = grd.convert("RGB")
        except:
            print("unreadable image")
            grd = PIL.Image.new(
                "RGB", (320, 640)
            )  # if the image is unreadable, use a blank image
        grd = self.grdimage_transform(grd)

        # generate a random rotation
        if self.random_orientation is None:
            if self.ori_noise >= 180:
                rotation = np.random.uniform(low=0.0, high=1.0)  #
            else:
                rotation_range = self.ori_noise / 360
                rotation = np.random.uniform(low=-rotation_range, high=rotation_range)
        else:
            rotation = self.random_orientation[idx] / 360

        grd = torch.roll(
            grd,
            (torch.round(torch.as_tensor(rotation) * grd.size()[2]).int()).item(),
            dims=2,
        )

        orientation_angle = (
            rotation * 360
        )  # 0 means heading North, counter-clockwise increasing

        # satellite OR osm tiles

        transform_osm_tile = transforms.Compose(
            [
                # resize
                transforms.Resize([512, 512]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        pos_index = 0
        osm_idx = self.label[idx][pos_index]

        osm_tile: np.ndarray = self.osm_tiles[osm_idx]
           
        if self.use_rendered_tiles:
            osm_tile = np.array(Colormap.apply(osm_tile))
            osm_tile = np.moveaxis(osm_tile, -1, 0)

        # print(f'dimension is {osm_tile.shape}')

        if self.use_50_n_osm_tiles:
            osm_tile = project_to_n(osm_tile)

        _, width_raw, height_raw = osm_tile.shape

        osm_tile_tensor = torch.from_numpy(np.ascontiguousarray(osm_tile)).float()

        osm_tile = transform_osm_tile(osm_tile_tensor)
        _, height, width = osm_tile.size()
        pos_index = 0
        [row_offset, col_offset] = self.delta[idx, pos_index]
        row_offset = np.round(row_offset / height_raw * height)
        col_offset = np.round(col_offset / width_raw * width)

        if self.pos_only:  # load positives only
            pos_index = 0
            sat = PIL.Image.open(
                os.path.join(self.sat_list[self.label[idx][pos_index]])
            )
            [row_offset, col_offset] = self.delta[
                idx, pos_index
            ]  # delta = [delta_lat, delta_lon]
        else:  # load positives and semi-positives
            col_offset = 320
            row_offset = 320
            while (
                np.abs(col_offset) >= 320 or np.abs(row_offset) >= 320
            ):  # do not use the semi-positives where GT location is outside the image
                pos_index = random.randint(0, 3)
                sat = PIL.Image.open(
                    os.path.join(self.sat_list[self.label[idx][pos_index]])
                )
                [row_offset, col_offset] = self.delta[
                    idx, pos_index
                ]  # delta = [delta_lat, delta_lon]

        sat = sat.convert("RGB")
        width_raw, height_raw = sat.size

        sat = self.satimage_transform(sat)

        _, height, width = sat.size()
        row_offset = np.round(row_offset / height_raw * height)
        col_offset = np.round(col_offset / width_raw * width)

        # groundtruth location on the aerial image
        # Gaussian GT
        gt = np.zeros([1, height, width], dtype=np.float32)
        gt_with_ori = np.zeros([20, height, width], dtype=np.float32)
        x, y = np.meshgrid(
            np.linspace(-width / 2 + col_offset, width / 2 + col_offset, width),
            np.linspace(-height / 2 - row_offset, height / 2 - row_offset, height),
        )
        d = np.sqrt(x * x + y * y)
        sigma, mu = 4, 0.0
        gt[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        gt = torch.tensor(gt)

        if self.train:
            # find the ground truth orientation index, we use 20 orientation bins, and each bin is 18 degrees
            index = int(orientation_angle // 18)
            ratio = (orientation_angle % 18) / 18
            if index == 0:
                gt_with_ori[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * (
                    1 - ratio
                )
                gt_with_ori[19, :, :] = (
                    np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * ratio
                )
            else:
                gt_with_ori[20 - index, :, :] = np.exp(
                    -((d - mu) ** 2 / (2.0 * sigma**2))
                ) * (1 - ratio)
                gt_with_ori[20 - index - 1, :, :] = (
                    np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * ratio
                )
        gt_with_ori = torch.tensor(gt_with_ori)

        orientation = torch.full(
            [2, height, width], np.cos(orientation_angle * np.pi / 180)
        )
        orientation[1, :, :] = np.sin(orientation_angle * np.pi / 180)

        if "NewYork" in self.grd_list[idx]:
            city = "NewYork"
        elif "Seattle" in self.grd_list[idx]:
            city = "Seattle"
        elif "SanFrancisco" in self.grd_list[idx]:
            city = "SanFrancisco"
        elif "Chicago" in self.grd_list[idx]:
            city = "Chicago"

        return grd, sat, osm_tile, gt, gt_with_ori, orientation, city, orientation_angle

    def get_item_sat(self, idx):
        '''
           Helper function since when we use osm mode or fusion mode, we can't get both 

           It may change since fusion is supposed to be done in the model 
        '''
        if self.pos_only:  # load positives only
            pos_index = 0
            sat = PIL.Image.open(
                os.path.join(self.sat_list[self.label[idx][pos_index]])
            )
        else:  # load positives and semi-positives
            col_offset = 320
            row_offset = 320
            while (
                np.abs(col_offset) >= 320 or np.abs(row_offset) >= 320
            ):  # do not use the semi-positives where GT location is outside the image
                pos_index = random.randint(0, 3)
                sat = PIL.Image.open(
                    os.path.join(self.sat_list[self.label[idx][pos_index]])
                )
                [row_offset, col_offset] = self.delta[
                    idx, pos_index
                ]  # delta = [delta_lat, delta_lon]

        sat = sat.convert("RGB")
        width_raw, height_raw = sat.size

        sat = self.satimage_transform(sat)
        return sat

    def get_item_osm(self, idx):
        '''
           Same as 'get_item_sat' but for osm 
        '''
        transform_osm_tile = transforms.Compose(
            [
                # resize
                transforms.Resize([512, 512]),
            ]
        )

        pos_index = 0
        osm_idx = self.label[idx][pos_index]

        osm_tile: np.ndarray = self.osm_tiles[osm_idx]

        osm_tile_tensor = torch.from_numpy(np.ascontiguousarray(osm_tile)).float()

        osm_tile = transform_osm_tile(osm_tile_tensor)
        return osm_tile
