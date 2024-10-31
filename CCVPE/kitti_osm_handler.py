from ast import Tuple
import os
import gzip
import pickle

from typing import List
from maploc.demo import process_latlong
from maploc.osm.tiling import TileManager
import numpy as np

from tqdm import tqdm

# only works for KITTI dataset
root = "/work/vita/datasets/KITTI"  # cluster test
KITTI_TILE_SIZE = 102.4  # m


def prepare_osm_data(dataset_root: str, test_mode: bool = False) -> None:
    """
    We want to download the tiles if they are not already downloaded

    path : KITTI/osm_tiles/data.pkl.gz

    Main function of this file
    """
    # check if osm data is already downloaded
    downloaded: bool = is_osm_tiles_downloaded(dataset_root)

    if downloaded:
        print(
            "OSM tiles detected",
            "\n remove all VIGOR/{city}/osm_tiles/data.pkl.gy to redownload tiles",
        )
        return  # already downloaded, exit

    print(
        "OSM tiles are not present, will be downloaded now. It will take a very long."
    )

    # subfolder 'osm_tiles' with the data
    create_dirs(dataset_root)

    download_tiles(dataset_root, test_mode)

    print("OSM tiles are now downloaded")


def download_tiles(dataset_root: str, test_mode: bool = False) -> None:
    """
    We are downloading in order from train_files.txt, test{1,2}_files.txt and we create 3 subdir

    TODO: host the files somewhere instead
    """
    for file in file_list(dataset_root):
        download_per_file(dataset_root, file)


def already_downloaded_for(dataset_root: str, city: str) -> bool:
    return os.path.isfile(os.path.join(dataset_root, city, "osm_tiles", "data.pkl.gz"))


def test_download_per_date(dataset_root: str, date: str) -> None:
    """
    Since downloading osm tiles takes around 1 day, this function helps debug the model while the tiles are downloaded
    """

    latlong_list = list_latlong(dataset_root, date)
    print(len(latlong_list), " tiles for ", date)

    print("TEST MODE")
    rasterized_map_list = []

    holder = get_osm_raster(latlong_list[0][1])
    for name, latlong in tqdm(latlong_list, desc=f"Processing tiles for {date}"):
        rasterized_map_list.append((name, holder))

    osm_dir_path = os.path.join(dataset_root, date, "osm_tiles")
    with gzip.open(os.path.join(osm_dir_path, "data.pkl.gz"), "wb") as f:
        pickle.dump(rasterized_map_list, f)


def download_per_file(dataset_root: str, file: str) -> None:
    latlong_list = list_latlong(dataset_root, file)

    rasterized_map_list = []
    for name, latlong in tqdm(latlong_list, desc=f"Processing tiles for {file}"):
        rasterized_map_list.append((name, get_osm_raster(latlong)))

    osm_dir_path = os.path.join(dataset_root, "osm_tiles", file)
    with gzip.open(os.path.join(osm_dir_path, "data.pkl.gz"), "wb") as f:
        pickle.dump(rasterized_map_list, f)


def get_osm_raster(latlong: tuple[float, float]) -> np.ndarray:
    # Orienternet black box magic but we do get our osm tile (canvas.raster) at the end

    proj, bbox = process_latlong(
        prior_latlon=latlong,
        tile_size_meters=KITTI_TILE_SIZE,
        )
    ppm = 640 / KITTI_TILE_SIZE / 2  # To get 640x640 pixels at the end
    tiler = TileManager.from_bbox(proj, bbox, ppm)  # type: ignore
    canvas = tiler.query(bbox)

    return canvas.raster


def list_latlong(dataset_root: str, file: str) -> List[tuple[str, tuple[float, float]]]:
    """
    2011_10_03/2011_10_03_drive_0042_sync/0000000939.png -0.3455326 -0.9653194 -0.0077375486
    """

    file_path = os.path.join("kitti_split", file)
    latlong_list = []

    with open(file_path, "r") as file:
        for line in file.readlines():
            oxts_file_name = get_oxts_file_name(dataset_root, line)
            with open(oxts_file_name, "r") as longlat_file:
                content = longlat_file.readline().split(" ")
                # get heading
                lat = float(content[0])
                lon = float(content[1])
                latlong_list.append((oxts_file_name, (lat, lon)))

    return latlong_list


def get_oxts_file_name(dataset_root: str, line: str) -> str:
    data = line.split()[0]  # apparently we ignore the 3 floats
    drive_dir = data[:38]
    image_no = data[38:].lower().replace(".png", ".txt")
    return os.path.join(dataset_root, "raw_data", drive_dir, image_no)


def create_dirs(dataset_root: str) -> None:
    path = os.path.join(dataset_root, "osm_tiles")
    for file in file_list():
        try:
            os.makedirs(os.path.join(path, "file"))
        except:
            print(f"dir for {file} tiles already present")


def is_osm_tiles_downloaded(dataset_root: str) -> bool:

    # check if osm data is already downloaded by looking if dir exists
    dir_data_path = os.path.join(dataset_root, "osm_tiles")
    if not os.path.isdir(dir_data_path):
        return False
    else:
        if not os.path.isfile(os.path.join(dir_data_path, "data.pkl.gz")):
            return False

    return True


def date_list(dataset_root: str) -> List[str]:
    return os.listdir(os.path.join(dataset_root, "raw_data"))


def file_list() -> List[str]:
    return os.listdir("kitti_split")
