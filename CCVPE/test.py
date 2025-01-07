import numpy as np 
import os


def test_seed(seed):
    print('---')
    print('seed ', seed)
    np.random.seed(seed)

    index_list = np.arange(52609)
    np.random.shuffle(index_list)

    dists_path = '/work/vita/qngo/dists'
    dist_file = "dist_sub_9.npy"
    dist_indices = np.load(os.path.join(dists_path, dist_file))
    np.random.shuffle(dist_indices)
    train_indices = dist_indices[:int(len(dist_indices)*0.8)] # The thing is we sadly can't remove all hard example from the validation set
    # train_indices = index_list[0: int(len(index_list)*0.8)]
    # train_indices = index_list[0:3000]
    val_indices = index_list[int(len(index_list)*0.8):]

    # first 10
    # [44301 43259 20958  7134 42385 22482 44211 15818 38390  8809]
    print(f'size before clashing {len(val_indices)}')

    print(f'first 10 val indices {val_indices[:10]}')

    len_before = len(val_indices)
    to_del = []
    for idx, val in enumerate(val_indices):
        if val in train_indices:
            to_del.append(idx)

    val_indices = np.delete(val_indices, to_del)
    print(f'size after clashing {len(val_indices)}')
    print(f'difference is {len_before - len(val_indices)}')

for i in range(30):
    test_seed(i)
