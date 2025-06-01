import torch
from torchvision import transforms


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
    '''
       We always use the same base transforms
    '''
    transform_grd = transforms.Compose([
        transforms.Resize([320, 640]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_sat = transforms.Compose([
        # resize
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_grd, transform_sat
