import numpy as np

def compare_dist(distance1, distance2):
    '''
        Take 2 arrays 
        return :
            20 indices where distance1 is better
            20 indices where distance2 is better

        format of return:
            Dict[List[tuple(idx, d1, d2, diff)]]
    '''

    assert len(distance1) == len(distance2)

    count_d1 = 0
    count_d2 = 0

    diff = distance1 - distance2

    n = 20
    ind = np.argpartition(diff, -n)[-n:]
    d1_distances_idx = ind[np.argsort(diff[ind])]

    d1_res = []
    for i in range(len(d1_distances_idx)):
        idx = d1_distances_idx[i]
        val = (idx, distance1[idx], distance2[idx], diff[idx])
        d1_res.append(val)

    ind = np.argpartition(diff, n)[:n]
    d2_distances_idx = ind[np.argsort(diff[ind])]

    d2_res = []
    for i in range(len(d2_distances_idx)):
        idx = d2_distances_idx[i]
        val = (idx, distance1[idx], distance2[idx], diff[idx])
        d2_res.append(val)

    return {'distance1': d2_res, 'distance2': d1_res}
