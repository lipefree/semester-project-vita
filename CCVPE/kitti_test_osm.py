from datasets import SatGrdDataset

root = "/work/vita/datasets/KITTI"
train_file = "kitti_split/train_files.txt"

dataset = SatGrdDataset(root, train_file)
