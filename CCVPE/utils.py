import torch
from torchvision import transforms
from torchvision.transforms.v2 import GaussianNoise


def process_data(data, device):
    grd, sat, osm, gt, gt_with_ori, gt_orientation, city, _ = data
    grd = grd.to(device)
    sat = sat.to(device)
    osm = osm.to(device)
    gt = gt.to(device)
    gt_with_ori = gt_with_ori.to(device)
    gt_orientation = gt_orientation.to(device)

    gt_flattened = torch.flatten(gt, start_dim=1)
    gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

    return grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened


def get_data_transforms():
    """
    We always use the same base transforms
    """
    transform_grd = transforms.Compose(
        [
            transforms.Resize([320, 640]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_sat = transforms.Compose(
        [
            # resize
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform_grd, transform_sat


def gaussian_noise(x, scale=0.0):
    return GaussianNoise(sigma=scale, clip=False)(x)


def process_data_augment(data, device, augment_type):
    """
    We support 4 types of augmentation
    - missing_OSM: OSM is now a zero tensor
    - missing_SAT: SAT is now a zero tensor
    - noisy_OSM: OSM has added Gaussian noise
    - noisy_SAT: SAT has added Gaussian noise
    - none: no augmentation is done
    """
    grd, sat, osm, gt, gt_with_ori, gt_orientation, city, _ = data
    grd = grd.to(device)
    sat = sat.to(device)
    osm = osm.to(device)

    match augment_type:
        case "missing_OSM":
            osm = torch.zeros_like(osm)
        case "missing_SAT":
            sat = torch.zeros_like(sat)
        case "noisy_OSM":
            osm = gaussian_noise(osm, scale=0.5)
        case "noisy_SAT":
            sat = gaussian_noise(sat, scale=0.5)
        case "none":
            pass
        case _:
            raise Exception("Unsuported augment type")

    gt = gt.to(device)
    gt_with_ori = gt_with_ori.to(device)
    gt_orientation = gt_orientation.to(device)

    gt_flattened = torch.flatten(gt, start_dim=1)
    gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

    return grd, sat, osm, gt, gt_with_ori, gt_orientation, city, gt_flattened
