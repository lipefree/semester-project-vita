import numpy as np


def project_to_n(
    matrix,
    n: int = 50,
    dim: tuple[int, int] = (640, 640),
    partition: list[int] = [7, 10, 33],
):
    """
    Project the matrix (expected to be [3, 640, 640]) to [n, 640, 640] where each layer n represent one OSM object.
    The input matrix is expected to comes from Orienternet representation of the rasterized map.

    layer 0: represent object 1 etc

    For speed, this function expect to get the partition and Orienternet have this default partition :
        areas: 7
        ways: 10
        nodes: 33

    """
    dims = (n,) + dim
    n_matrix = np.zeros(dims)
    layer = 0
    start_index = 0
    partition_aug = partition + [n]
    for _ in range(len(partition_aug) - 1):
        for i in range(start_index, partition[layer]):
            n_matrix[i] = matrix[layer] == i + 1

        start_index = partition_aug[layer] + 1
        layer += 1

    return n_matrix
