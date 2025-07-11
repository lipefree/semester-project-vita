import numpy as np
import math


def get_location(heatmap):
    """
    Given a heatmap, return [x, y] the argmax location
    """
    current_pred = heatmap
    loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)

    return loc_pred


def get_meter_distance(loc_gt, loc_pred, city: str, batch_idx) -> float:
    """
    distance in meters between groundtruth location (x,y) and predicted one

    TODO: those cities should be turned into an enum
    """
    pixel_distance = np.sqrt((loc_gt[1] - loc_pred[1]) ** 2 + (loc_gt[2] - loc_pred[2]) ** 2)
    meter_distance = -1
    if city == "NewYork":
        meter_distance = pixel_distance * 0.113248 / 512 * 640
    elif city == "Seattle":
        meter_distance = pixel_distance * 0.100817 / 512 * 640
    elif city == "SanFrancisco":
        meter_distance = pixel_distance * 0.118141 / 512 * 640
    elif city == "Chicago":
        meter_distance = pixel_distance * 0.111262 / 512 * 640
    elif city == "Karlsruhe":
        meter_distance = pixel_distance * get_meter_per_pixel_kitti()
    else:
        raise Exception(f"{city} was provided as a city but it is not recognized")

    return meter_distance


def get_orientation_distance(
    gt_orientation, pred_orientation, loc_gt, loc_pred, batch_idx
) -> float:
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

        return np.min([np.abs(angle_gt - angle_pred), 360 - np.abs(angle_gt - angle_pred)])
    else:
        return None


def get_meter_per_pixel_kitti(lat=49.015, zoom=18, scale=1):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi / 180.0) / (2**zoom)
    meter_per_pixel /= 2  # because use scale 2 to get satmap
    meter_per_pixel /= scale
    return meter_per_pixel
