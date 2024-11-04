import numpy as np
import math

def get_location(heatmap):
    '''
        Given a heatmap, return [x, y] the argmax location
    '''
    current_pred = heatmap
    loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)

    return loc_pred


def get_meter_distance(loc_gt, loc_pred, city: str, batch_idx) -> float:
    '''
       distance in meters between groundtruth location (x,y) and predicted one
    '''
    pixel_distance = np.sqrt((loc_gt[1]-loc_pred[1])**2+(loc_gt[2]-loc_pred[2])**2)
    meter_distance = - 1
    if city == 'NewYork':
        meter_distance = pixel_distance * 0.113248 / 512 * 640
    elif city == 'Seattle':
        meter_distance = pixel_distance * 0.100817 / 512 * 640
    elif city == 'SanFrancisco':
        meter_distance = pixel_distance * 0.118141 / 512 * 640
    elif city == 'Chicago':
        meter_distance = pixel_distance * 0.111262 / 512 * 640
    if meter_distance == -1:
        print('problem finding the corresponding city')
        meter_distance = math.inf

    return meter_distance


def get_orientation_distance(gt_orientation, pred_orientation, loc_gt, loc_pred, batch_idx) -> float:
    cos_pred, sin_pred = pred_orientation[batch_idx, :, loc_pred[1], loc_pred[2]]

    if np.abs(cos_pred) <= 1 and np.abs(sin_pred) <= 1:
        a_acos_pred = math.acos(cos_pred)
        if sin_pred < 0:
            angle_pred = math.degrees(-a_acos_pred) % 360
        else:
            angle_pred = math.degrees(a_acos_pred)

        cos_gt, sin_gt = gt_orientation[batch_idx, :, loc_gt[1], loc_gt[2]]
        a_acos_gt = math.acos(cos_gt)
        if sin_gt < 0:
            angle_gt = math.degrees(-a_acos_gt) % 360
        else:
            angle_gt = math.degrees(a_acos_gt)

    return np.min([np.abs(angle_gt-angle_pred), 360-np.abs(angle_gt-angle_pred)])
