import os
from torch.utils.data import Dataset
import PIL.Image
import torch
import numpy as np
import random
from PIL import ImageFile
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms.functional as TF
from torchvision import transforms
from osm_tiles_helper import project_to_n
from maploc.osm.viz import Colormap
import torchvision.transforms.v2 as transforms
from enum import Enum

torch.manual_seed(17)
np.random.seed(0)


# Represent the current dataset used for training
class DatasetType(Enum):
    VIGOR = (1,)
    KITTI = (2,)


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
        use_osm_rendered=True,
        use_concat=False,
        augment=False,
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
        self.augment = (
            augment  # data augmentation such as rotations, random gaussian noise and random erasure
        )

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
                osm_tile_path = os.path.join(self.root, city, "osm_tiles", "data.npy")

                # with open(osm_tile_path, 'rb') as f:
                #     loaded_data = np.load(f)
                loaded_data = np.load(osm_tile_path, mmap_mode="r")

                # with gzip.open(osm_tile_path, "rb") as f:
                #     loaded_data = pickle.load(f)

                print(f"loaded data pikle")
                self.osm_tiles.extend(loaded_data)
                print(f"osm tiles loaded for {city}")

            sat_list_fname = os.path.join(self.root, label_root, city, "satellite_list.txt")
            with open(sat_list_fname, "r") as file:
                for line in file.readlines():
                    self.sat_list.append(
                        os.path.join(self.root, city, "satellite", line.replace("\n", ""))
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
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.grd_list.append(os.path.join(self.root, city, "panorama", data[0]))
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
            grd = PIL.Image.new("RGB", (320, 640))  # if the image is unreadable, use a blank image
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

        orientation_angle = rotation * 360  # 0 means heading North, counter-clockwise increasing

        # satellite OR osm tiles

        if self.use_osm_tiles:
            transform_osm_tile = transforms.Compose(
                [
                    # resize
                    transforms.Resize([512, 512]),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
            sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
            [row_offset, col_offset] = self.delta[idx, pos_index]  # delta = [delta_lat, delta_lon]
        else:  # load positives and semi-positives
            col_offset = 320
            row_offset = 320
            while (
                np.abs(col_offset) >= 320 or np.abs(row_offset) >= 320
            ):  # do not use the semi-positives where GT location is outside the image
                pos_index = random.randint(0, 3)
                sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
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
                gt_with_ori[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * (1 - ratio)
                gt_with_ori[19, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * ratio
            else:
                gt_with_ori[20 - index, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * (
                    1 - ratio
                )
                gt_with_ori[20 - index - 1, :, :] = (
                    np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * ratio
                )
        gt_with_ori = torch.tensor(gt_with_ori)

        orientation = torch.full([2, height, width], np.cos(orientation_angle * np.pi / 180))
        orientation[1, :, :] = np.sin(orientation_angle * np.pi / 180)

        if "NewYork" in self.grd_list[idx]:
            city = "NewYork"
        elif "Seattle" in self.grd_list[idx]:
            city = "Seattle"
        elif "SanFrancisco" in self.grd_list[idx]:
            city = "SanFrancisco"
        elif "Chicago" in self.grd_list[idx]:
            city = "Chicago"

        # print(self.sat_list[self.label[idx][pos_index]])
        # print("at idx ", idx)

        if self.use_osm_tiles:
            return (
                grd,
                sat,
                osm_tile,
                gt,
                gt_with_ori,
                orientation,
                city,
                orientation_angle,
            )
        else:
            return grd, sat, sat, gt, gt_with_ori, orientation, city, orientation_angle

    def get_item_sat(self, idx):
        """
        Helper function since when we use osm mode or fusion mode, we can't get both

        It may change since fusion is supposed to be done in the model
        """
        if self.pos_only:  # load positives only
            pos_index = 0
            sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
        else:  # load positives and semi-positives
            col_offset = 320
            row_offset = 320
            while (
                np.abs(col_offset) >= 320 or np.abs(row_offset) >= 320
            ):  # do not use the semi-positives where GT location is outside the image
                pos_index = random.randint(0, 3)
                sat = PIL.Image.open(os.path.join(self.sat_list[self.label[idx][pos_index]]))
                [row_offset, col_offset] = self.delta[
                    idx, pos_index
                ]  # delta = [delta_lat, delta_lon]

        sat = sat.convert("RGB")
        width_raw, height_raw = sat.size

        sat = self.satimage_transform(sat)
        return sat

    def get_item_osm(self, idx):
        """
        Same as 'get_item_sat' but for osm
        """
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


# ---------------------------------------------------------------------------------
# KITTI, our code is developed based on https://github.com/shiyujiao/HighlyAccurate
Default_lat = 49.015
Satmap_zoom = 18
SatMap_original_sidelength = 512
SatMap_process_sidelength = 512
satmap_dir = "satmap"
osmtile_dir = "osm_tiles"
grdimage_dir = "raw_data"
oxts_dir = "oxts/data"
left_color_camera_dir = "image_02/data"
CameraGPS_shift_left = [1.08, 0.26]


def get_meter_per_pixel(
    lat=Default_lat,
    zoom=Satmap_zoom,
    scale=SatMap_process_sidelength / SatMap_original_sidelength,
):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi / 180.0) / (2**zoom)
    meter_per_pixel /= 2  # because use scale 2 to get satmap
    meter_per_pixel /= scale
    return meter_per_pixel


class KITTIDataset(Dataset):
    def __init__(
        self,
        root,
        file,
        transform=None,
        shift_range_lat=20,
        shift_range_lon=20,
        rotation_range=10,
    ):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = (
            shift_range_lat / self.meter_per_pixel
        )  # shift range is in terms of pixels
        self.shift_range_pixels_lon = (
            shift_range_lon / self.meter_per_pixel
        )  # shift range is in terms of pixels

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.grdimage_transform = transform[0]
            self.satmap_transform = transform[1]

        self.pro_grdimage_dir = "raw_data"

        self.satmap_dir = satmap_dir
        self.osmtile_dir = osmtile_dir

        with open(file, "r") as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with PIL.Image.open(SatMap_name, "r") as SatMap:
            sat_map = SatMap.convert("RGB")

        # =================== read OSM tiles ========================================
        osm_tile_name = os.path.join(self.root, self.osmtile_dir, file_name.replace("png", "npy"))
        osm_tile_arr = np.load(osm_tile_name)
        map_viz = Colormap.apply(osm_tile_arr)
        osm_map = Image.fromarray(np.uint8(map_viz * 255)).convert("RGB")

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(
            self.root,
            grdimage_dir,
            drive_dir,
            oxts_dir,
            image_no.lower().replace(".png", ".txt"),
        )

        with open(oxts_file_name, "r") as f:
            content = f.readline().split(" ")
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(
                self.root,
                self.pro_grdimage_dir,
                drive_dir,
                left_color_camera_dir,
                image_no.lower(),
            )
            with PIL.Image.open(left_img_name, "r") as GrdImg:
                grd_img_left = GrdImg.convert("RGB")
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

        sat_rot = sat_map.rotate(
            (-heading) / np.pi * 180
        )  # make the east direction the vehicle heading
        sat_align_cam = sat_rot.transform(
            sat_rot.size,
            PIL.Image.AFFINE,
            (
                1,
                0,
                CameraGPS_shift_left[0] / self.meter_per_pixel,
                0,
                1,
                CameraGPS_shift_left[1] / self.meter_per_pixel,
            ),
            resample=PIL.Image.BILINEAR,
        )

        osm_rot = osm_map.rotate(-heading / np.pi * 180)

        osm_align_cam = osm_rot.transform(
            osm_rot.size,
            PIL.Image.AFFINE,
            (
                1,
                0,
                CameraGPS_shift_left[0] / self.meter_per_pixel,
                0,
                1,
                CameraGPS_shift_left[1] / self.meter_per_pixel,
            ),
            resample=PIL.Image.BILINEAR,
        )

        # randomly generate shift
        gt_shift_x = np.random.uniform(
            -1, 1
        )  # --> right as positive, parallel to the heading direction
        gt_shift_y = np.random.uniform(
            -1, 1
        )  # --> up as positive, vertical to the heading direction

        sat_rand_shift = sat_align_cam.transform(
            sat_align_cam.size,
            PIL.Image.AFFINE,
            (
                1,
                0,
                gt_shift_x * self.shift_range_pixels_lon,
                0,
                1,
                -gt_shift_y * self.shift_range_pixels_lat,
            ),
            resample=PIL.Image.BILINEAR,
        )

        osm_rand_shift = osm_align_cam.transform(
            osm_align_cam.size,
            PIL.Image.AFFINE,
            (
                1,
                0,
                gt_shift_x * self.shift_range_pixels_lon,
                0,
                1,
                -gt_shift_y * self.shift_range_pixels_lat,
            ),
            resample=PIL.Image.BILINEAR,
        )

        # randomly generate roation
        random_ori = (
            np.random.uniform(-1, 1) * self.rotation_range
        )  # 0 means the arrow in aerial image heading Easting, counter-clockwise increasing
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)

        sat_map = TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)

        osm_rand_shift_rand_rot = osm_rand_shift.rotate(random_ori)

        osm_map = TF.center_crop(osm_rand_shift_rand_rot, SatMap_process_sidelength)
        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)
            osm_map = self.satmap_transform(osm_map)

        # gt heat map
        x_offset = int(
            gt_shift_x * self.shift_range_pixels_lon * np.cos(random_ori / 180 * np.pi)
            - gt_shift_y * self.shift_range_pixels_lat * np.sin(random_ori / 180 * np.pi)
        )  # horizontal direction
        y_offset = int(
            -gt_shift_y * self.shift_range_pixels_lat * np.cos(random_ori / 180 * np.pi)
            - gt_shift_x * self.shift_range_pixels_lon * np.sin(random_ori / 180 * np.pi)
        )  # vertical direction

        x, y = np.meshgrid(
            np.linspace(-256 + x_offset, 256 + x_offset, 512),
            np.linspace(-256 + y_offset, 256 + y_offset, 512),
        )
        d = np.sqrt(x * x + y * y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        gt = torch.tensor(gt)

        # orientation gt
        orientation_angle = 90 - random_ori
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360

        gt_with_ori = np.zeros([16, 512, 512], dtype=np.float32)
        index = int(orientation_angle // 22.5)
        ratio = (orientation_angle % 22.5) / 22.5
        if index == 0:
            gt_with_ori[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * (1 - ratio)
            gt_with_ori[15, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * ratio
        else:
            gt_with_ori[16 - index, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * (
                1 - ratio
            )
            gt_with_ori[16 - index - 1, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * ratio
        gt_with_ori = torch.tensor(gt_with_ori)

        orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi / 180))
        orientation_map[1, :, :] = np.sin(orientation_angle * np.pi / 180)

        city = "Karlsruhe"
        return (
            grd_left_imgs[0],
            sat_map,
            osm_map,
            gt,
            gt_with_ori,
            orientation_map,
            city,
            orientation_angle,
        )


class KITTIDatasetTest(Dataset):
    def __init__(
        self,
        root,
        file,
        transform=None,
        shift_range_lat=20,
        shift_range_lon=20,
        rotation_range=10,
    ):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = (
            shift_range_lat / self.meter_per_pixel
        )  # shift range is in terms of meters
        self.shift_range_pixels_lon = (
            shift_range_lon / self.meter_per_pixel
        )  # shift range is in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.grdimage_transform = transform[0]
            self.satmap_transform = transform[1]

        self.pro_grdimage_dir = "raw_data"

        self.satmap_dir = satmap_dir
        self.osmtile_dir = osmtile_dir

        with open(file, "r") as f:
            file_name = f.readlines()

        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(" ")
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with PIL.Image.open(SatMap_name, "r") as SatMap:
            sat_map = SatMap.convert("RGB")

        # =================== read OSM tiles ========================================
        osm_tile_name = os.path.join(self.root, self.osmtile_dir, file_name.replace("png", "npy"))
        osm_tile_arr = np.load(osm_tile_name)
        map_viz = Colormap.apply(osm_tile_arr)
        osm_map = Image.fromarray(np.uint8(map_viz * 255)).convert("RGB")

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        grd_left_depths = torch.tensor([])
        # image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(
            self.root,
            grdimage_dir,
            drive_dir,
            oxts_dir,
            image_no.lower().replace(".png", ".txt"),
        )
        with open(oxts_file_name, "r") as f:
            content = f.readline().split(" ")
            # get heading
            lat = float(content[0])
            lon = float(content[1])
            heading = float(content[5])

            left_img_name = os.path.join(
                self.root,
                self.pro_grdimage_dir,
                drive_dir,
                left_color_camera_dir,
                image_no.lower(),
            )
            with PIL.Image.open(left_img_name, "r") as GrdImg:
                grd_img_left = GrdImg.convert("RGB")
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

        sat_rot = sat_map.rotate(-heading / np.pi * 180)

        sat_align_cam = sat_rot.transform(
            sat_rot.size,
            PIL.Image.AFFINE,
            (
                1,
                0,
                CameraGPS_shift_left[0] / self.meter_per_pixel,
                0,
                1,
                CameraGPS_shift_left[1] / self.meter_per_pixel,
            ),
            resample=PIL.Image.BILINEAR,
        )

        osm_rot = osm_map.rotate(-heading / np.pi * 180)

        osm_align_cam = osm_rot.transform(
            osm_rot.size,
            PIL.Image.AFFINE,
            (
                1,
                0,
                CameraGPS_shift_left[0] / self.meter_per_pixel,
                0,
                1,
                CameraGPS_shift_left[1] / self.meter_per_pixel,
            ),
            resample=PIL.Image.BILINEAR,
        )

        # load the shifts
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = sat_align_cam.transform(
            sat_align_cam.size,
            PIL.Image.AFFINE,
            (
                1,
                0,
                gt_shift_x * self.shift_range_pixels_lon,
                0,
                1,
                -gt_shift_y * self.shift_range_pixels_lat,
            ),
            resample=PIL.Image.BILINEAR,
        )

        osm_rand_shift = osm_align_cam.transform(
            osm_align_cam.size,
            PIL.Image.AFFINE,
            (
                1,
                0,
                gt_shift_x * self.shift_range_pixels_lon,
                0,
                1,
                -gt_shift_y * self.shift_range_pixels_lat,
            ),
            resample=PIL.Image.BILINEAR,
        )

        random_ori = float(theta) * self.rotation_range  # degree
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)

        sat_map = TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)
        osm_rand_shift_rand_rot = osm_rand_shift.rotate(random_ori)

        osm_map = TF.center_crop(osm_rand_shift_rand_rot, SatMap_process_sidelength)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)
            osm_map = self.satmap_transform(osm_map)

        # gt heat map
        x_offset = int(
            gt_shift_x * self.shift_range_pixels_lon * np.cos(random_ori / 180 * np.pi)
            - gt_shift_y * self.shift_range_pixels_lat * np.sin(random_ori / 180 * np.pi)
        )  # horizontal direction
        y_offset = int(
            -gt_shift_y * self.shift_range_pixels_lat * np.cos(random_ori / 180 * np.pi)
            - gt_shift_x * self.shift_range_pixels_lon * np.sin(random_ori / 180 * np.pi)
        )  # vertical direction

        x, y = np.meshgrid(
            np.linspace(-256 + x_offset, 256 + x_offset, 512),
            np.linspace(-256 + y_offset, 256 + y_offset, 512),
        )
        d = np.sqrt(x * x + y * y)
        sigma, mu = 4, 0.0
        gt = np.zeros([1, 512, 512], dtype=np.float32)
        gt[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        gt = torch.tensor(gt)

        # orientation gt
        orientation_angle = 90 - random_ori
        if orientation_angle < 0:
            orientation_angle = orientation_angle + 360
        elif orientation_angle > 360:
            orientation_angle = orientation_angle - 360

        gt_with_ori = np.zeros([16, 512, 512], dtype=np.float32)
        index = int(orientation_angle // 22.5)
        ratio = (orientation_angle % 22.5) / 22.5
        if index == 0:
            gt_with_ori[0, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * (1 - ratio)
            gt_with_ori[15, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * ratio
        else:
            gt_with_ori[16 - index, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * (
                1 - ratio
            )
            gt_with_ori[16 - index - 1, :, :] = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) * ratio
        gt_with_ori = torch.tensor(gt_with_ori)

        orientation_map = torch.full([2, 512, 512], np.cos(orientation_angle * np.pi / 180))
        orientation_map[1, :, :] = np.sin(orientation_angle * np.pi / 180)

        city = "Karlsruhe"
        return (
            grd_left_imgs[0],
            sat_map,
            osm_map,
            gt,
            gt_with_ori,
            orientation_map,
            city,
            orientation_angle,
        )
