import os 
import numpy as np

def print_result(distance_array, selected_model):
    mean = np.mean(distance_array)
    median = np.median(distance_array)
    print(f'for model : {selected_model}')
    print(f'mean in meter {mean}, median in meter {median}')

    save_qual_res = os.path.join("/work/vita/qngo", "qualitative", selected_model, "test_quantitative_results.txt")
    with open(save_qual_res, 'w') as f:
        f.write(f'mean in meter : {mean}, median in meter : {median}')

def load_array(selected_model):
    save_qual = os.path.join("/work/vita/qngo", "qualitative", selected_model, "distance_test_new.npy")
    array = np.load(save_qual)
    return array

selected_model = "samearea_HFoV360_samearea_lr_1e-04_osm_rendered_tile"
array = load_array(selected_model)
print_result(distance_array, selected_model)
