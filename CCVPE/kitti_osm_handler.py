from ast import Tuple
import os
import gzip
import pickle

from typing import List
from maploc.latlong_utils import process_latlong
from maploc.osm.viz import GeoPlotter
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images
import numpy as np

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

# only works for KITTI dataset
root = "/work/vita/qngo/KITTI"  # cluster test
KITTI_TILE_SIZE = 250.6604 / 2  # m


def prepare_osm_data(dataset_root: str, test_mode: bool = False) -> None:
    """
    We want to download the tiles if they are not already downloaded

    path : KITTI/osm_tiles/data.pkl.gz

    Main function of this file
    """
    if not is_dataset_present(dataset_root):
        print("Provited dataset root does not exist")
        return

    # check if osm data is already downloaded
    downloaded: bool = is_osm_tiles_downloaded(dataset_root)

    if downloaded:
        print(
            "OSM tiles detected",
            "\n remove all VIGOR/{city}/osm_tiles/data.pkl.gy to redownload tiles",
        )
        return  # already downloaded, exit

    print("OSM tiles are not present, will be downloaded now. It will take a very long.")

    # subfolder 'osm_tiles' with the data
    create_dirs(dataset_root)

    download_tiles(dataset_root, test_mode)

    print("OSM tiles are now downloaded")


def download_tiles(dataset_root: str, test_mode: bool = False) -> None:
    """
    We are downloading in order from train_files.txt, test{1,2}_files.txt and we create 3 subdir

    TODO: host the files somewhere instead
    """
    for file in split_list()[1:]:
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


def _download_one_tile(args):
    dataset_root, name, latlong = args
    tile = get_osm_raster(latlong)

    # reconstruct output path
    path_parts = Path(name).parts
    file_name = path_parts[-1].replace(".txt", ".npy")
    middle_path = path_parts[-5:-3]
    out_dir = os.path.join(dataset_root, "osm_tiles", middle_path[0], middle_path[1])
    out_path = os.path.join(out_dir, file_name)

    with open(out_path, "wb") as f:
        np.save(f, tile)

    return name


def download_per_file(dataset_root: str, file: str) -> None:
    latlong_list = list_latlong(dataset_root, file)
    args = [(dataset_root, name, ll) for name, ll in latlong_list]

    print("cpu count ", cpu_count())
    with Pool(
        processes=min(cpu_count(), 3)
    ) as pool:  # unfortunatly, we will DDOS the overpass instance with more parallelization
        # imap_unordered gives you results as they come back
        for _ in tqdm(
            pool.imap_unordered(_download_one_tile, args),
            total=len(args),
            desc=f"Processing tiles for {file}",
        ):
            pass


def get_osm_raster(latlong: tuple[float, float]) -> np.ndarray:
    # Orienternet black box magic but we do get our osm tile (canvas.raster) at the end

    proj, bbox = process_latlong(
        prior_latlon=latlong,
        tile_size_meters=KITTI_TILE_SIZE,
    )
    ppm = 1280 / KITTI_TILE_SIZE / 2  # To get 1280x1280 pixels at the end
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
    return os.path.join(
        dataset_root,
        "raw_data",
        drive_dir,
        "oxts",
        "data",
        image_no,
    )


def create_dirs(dataset_root: str) -> None:
    path = os.path.join(dataset_root, "osm_tiles")

    # KITTI is 2 level deep before we get the sat images
    for file in file_list(dataset_root):
        try:
            os.makedirs(os.path.join(path, file))
        except:
            print(f"dir for {file} tiles already present")
        # Now we will handle the second level
        for subdirs in os.listdir(os.path.join(dataset_root, "satmap", file)):
            try:
                os.makedirs(os.path.join(path, file, subdirs))
            except:
                print(f"dir for {subdirs} tiles already present")


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


def file_list(dataset_root) -> List[str]:
    return os.listdir(os.path.join(dataset_root, "satmap"))


def split_list() -> List[str]:
    return os.listdir(os.path.join("kitti_split"))


def is_dataset_present(dataset_root: str) -> bool:
    return os.path.isdir(dataset_root)


prepare_osm_data(root)
