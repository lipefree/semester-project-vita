import torch
import torch.nn as nn


def infoNCELoss(scores, labels, temperature=0.1):
    """
    Contrastive loss over matching score. Adapted from https://arxiv.org/pdf/2004.11362.pdf Eq.2
    We extraly weigh the positive samples using the ground truth likelihood on those positions

    loss = - 1/sum(weights) * sum(inner_element*weights)
    inner_element = log( exp(score_pos/temperature) / sum(exp(score/temperature)) )
    """

    exp_scores = torch.exp(scores / temperature)
    bool_mask = (
        labels > 1e-2
    )  # elements with a likelihood > 1e-2 are considered as positive samples in contrastive learning

    denominator = torch.sum(exp_scores, dim=1, keepdim=True)
    inner_element = torch.log(torch.masked_select(exp_scores / denominator, bool_mask))
    loss = -torch.sum(
        inner_element * torch.masked_select(labels, bool_mask)
    ) / torch.sum(torch.masked_select(labels, bool_mask))

    return loss


def cross_entropy_loss(logits, labels):
    return -torch.sum(labels * nn.LogSoftmax(dim=1)(logits)) / logits.size()[0]


def orientation_loss(ori, gt_orientation, gt):
    return (
        torch.sum(
            torch.sum(torch.square(gt_orientation - ori), dim=1, keepdim=True) * gt
        )
        / ori.size()[0]
    )


def loss_ccvpe(
    output, gt, gt_orientation, gt_with_ori, weight_infoNCE, weight_ori
) -> float:

    
    if(len(output) == 9 ):
        (
            logits_flattened,
            heatmap,
            ori,
            matching_score_stacked,
            matching_score_stacked2,
            matching_score_stacked3,
            matching_score_stacked4,
            matching_score_stacked5,
            matching_score_stacked6,
        ) = output
    else:
        (   _,
            logits_flattened,
            heatmap,
            ori,
            matching_score_stacked,
            matching_score_stacked2,
            matching_score_stacked3,
            matching_score_stacked4,
            matching_score_stacked5,
            matching_score_stacked6,
        ) = output


    gt_bottleneck = nn.MaxPool2d(64, stride=64)(gt_with_ori)
    gt_bottleneck2 = nn.MaxPool2d(32, stride=32)(gt_with_ori)
    gt_bottleneck3 = nn.MaxPool2d(16, stride=16)(gt_with_ori)
    gt_bottleneck4 = nn.MaxPool2d(8, stride=8)(gt_with_ori)
    gt_bottleneck5 = nn.MaxPool2d(4, stride=4)(gt_with_ori)
    gt_bottleneck6 = nn.MaxPool2d(2, stride=2)(gt_with_ori)

    gt_flattened = gt_flattened = torch.flatten(gt, start_dim=1)
    gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

    loss_ori = orientation_loss(ori, gt_orientation, gt)
    loss_infoNCE = infoNCELoss(
        torch.flatten(matching_score_stacked, start_dim=1),
        torch.flatten(gt_bottleneck, start_dim=1),
    )
    loss_infoNCE2 = infoNCELoss(
        torch.flatten(matching_score_stacked2, start_dim=1),
        torch.flatten(gt_bottleneck2, start_dim=1),
    )
    loss_infoNCE3 = infoNCELoss(
        torch.flatten(matching_score_stacked3, start_dim=1),
        torch.flatten(gt_bottleneck3, start_dim=1),
    )
    loss_infoNCE4 = infoNCELoss(
        torch.flatten(matching_score_stacked4, start_dim=1),
        torch.flatten(gt_bottleneck4, start_dim=1),
    )
    loss_infoNCE5 = infoNCELoss(
        torch.flatten(matching_score_stacked5, start_dim=1),
        torch.flatten(gt_bottleneck5, start_dim=1),
    )
    loss_infoNCE6 = infoNCELoss(
        torch.flatten(matching_score_stacked6, start_dim=1),
        torch.flatten(gt_bottleneck6, start_dim=1),
    )
    loss_ce = cross_entropy_loss(logits_flattened, gt_flattened)

    weighted_infoNCE = (
        weight_infoNCE
        * (
            loss_infoNCE
            + loss_infoNCE2
            + loss_infoNCE3
            + loss_infoNCE4
            + loss_infoNCE5
            + loss_infoNCE6
        )
        / 6
    )
    loss = loss_ce + weighted_infoNCE + weight_ori * loss_ori
    return loss


def loss_router(weight, logits, gt_choice, sat_dist, osm_dist, t) -> float:

    # Binary cross-entropy loss
    ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
    loss1 = ce_loss(logits, gt_choice)  # Binary classification loss for routing decision

    # print(f'chosen {chosen}')

    # Difference in distances scaled by the routing decision

    distance = torch.abs(sat_dist - osm_dist)
    mask = (distance >= -1).float()
    
    # distance = torch.clamp(distance, max=50) # sometimes the distance is over 200 which is too much

    # loss2 = torch.exp(torch.div(distance, 5))

    # loss2 = distance * 0.0125

    # d = 0 -> 1
    # d = 40 -> 1.5
    # a*40 + 1 = 1.5 
    # a = 0.5/40 = 0.0125
    loss2 = torch.sigmoid(distance - t)

    # print(f'loss 2 {loss2}')
    return loss1, loss2

def get_distances(gt, osm_heatmap, sat_heatmap, city):
    gt_pred = get_max_coordinates(gt)

    # print(f'gt coordinates {gt_pred}')
    # First we need to define what is the label
    # print(f'osm_heatmap shape {osm_heatmap.size()}')
    # print(f'gt shape {gt.size()}')
    osm_pred = get_max_coordinates(osm_heatmap)
    # print(f'osm coordinates {osm_pred}')
    # print(f'predicted shape {osm_pred.size()}')
    # print(f'pred {osm_pred}')
    osm_dist = distance(osm_pred, gt_pred, city)
    # print(f'osm dist {osm_dist}')

    sat_pred = get_max_coordinates(sat_heatmap)
    # print(f'sat coordinates {sat_pred}')
    sat_dist = distance(sat_pred, gt_pred, city)

    # print(f'osm dist {osm_dist}')
    # print(f'sat dist {sat_dist}')

    gt_choices = (sat_dist > osm_dist).float() # This means : 0 when sat is better, smaller dist is better (closer to gt)
    # gt_choices = gt_choices.unsqueeze(1)

    # print(f'gt_choices {gt_choices}')

    return gt_choices, sat_dist, osm_dist

def distance(x, y, city):
    
    CITY_SCALE = {
        'NewYork':       0.113248,
        'Seattle':       0.100817,
        'SanFrancisco':  0.118141,
        'Chicago':       0.111262
    }

    pixel_distance = torch.sqrt((x[:,1]-y[:,1])**2+(x[:,0]-y[:,0])**2)
    scale_list = []

    for c in city:
        if c in CITY_SCALE:
            scale_list.append(CITY_SCALE[c])
        else:
            scale_list.append(CITY_SCALE['NewYork'])

    scale_t = torch.tensor(scale_list, dtype=pixel_distance.dtype, device=pixel_distance.device)
    meter_distance = pixel_distance * scale_t / 512.0 * 640.0

    return meter_distance
     
     

def get_max_coordinates(heatmap):
    """
    Given a heatmap of shape (B, 1, H, W), return the (x, y) coordinates
    of the max value in the grid for each batch using divmod.
    
    Args:
        heatmap (torch.Tensor): Tensor of shape (B, 1, H, W)
    
    Returns:
        torch.Tensor: Tensor of shape (B, 2) where each row is (x, y)
    """
    # Find the index of the max value for each batch
    flat_indices = torch.argmax(heatmap.view(heatmap.size(0), -1), dim=1)

    # Convert flat index to 2D coordinates
    height, width = heatmap.shape[1], heatmap.shape[2]
    y_coords = flat_indices // width  # Row indices
    x_coords = flat_indices % width  # Column indices

    # Combine coordinates into a tensor of shape (B, 2)
    coordinates = torch.stack((x_coords, y_coords), dim=1)

    return coordinates
