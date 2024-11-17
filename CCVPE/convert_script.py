import matplotlib.pyplot as plt

from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images
from maploc.osm.parser import Patterns
from maploc.osm.download import get_osm

from torchvision import transforms
from PIL import Image
import numpy as np
from vigor_osm_handler import *

import os, pickle, gzip

root = "../../VIGOR"
city = "NewYork"
osm_tile_path = os.path.join(root, city, "osm_tiles", "data.pkl.gz")
with gzip.open(osm_tile_path, "rb") as f:
    loaded_data = pickle.load(f)

name, m = loaded_data[0]
new_data = np.zeros((len(loaded_data), 3, 640, 640))
for i, data in tqdm(enumerate(loaded_data)):
    name, m = data
    new_data[i] = m

npsave_tile_path = os.path.join(root, city, "osm_tiles", "data.npy")
with open(npsave_tile_path, 'wb') as f:
    np.save(f, new_data)