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
    bool_mask = labels>1e-2 # elements with a likelihood > 1e-2 are considered as positive samples in contrastive learning    

    denominator = torch.sum(exp_scores, dim=1, keepdim=True)
    inner_element = torch.log(torch.masked_select(exp_scores/denominator, bool_mask))
    loss = -torch.sum(inner_element*torch.masked_select(labels, bool_mask)) / torch.sum(torch.masked_select(labels, bool_mask))

    return loss


def cross_entropy_loss(logits, labels):
    return -torch.sum(labels * nn.LogSoftmax(dim=1)(logits)) / logits.size()[0]


def orientation_loss(ori, gt_orientation, gt):    
    return torch.sum(torch.sum(torch.square(gt_orientation-ori), dim=1, keepdim=True) * gt) / ori.size()[0]


def loss_ccvpe(output, gt, gt_orientation, gt_with_ori, weight_infoNCE, weight_ori) -> float:

    logits_flattened, heatmap, ori, matching_score_stacked, matching_score_stacked2, matching_score_stacked3, matching_score_stacked4, matching_score_stacked5, matching_score_stacked6 = output 

    gt_bottleneck = nn.MaxPool2d(64, stride=64)(gt_with_ori)
    gt_bottleneck2 = nn.MaxPool2d(32, stride=32)(gt_with_ori)
    gt_bottleneck3 = nn.MaxPool2d(16, stride=16)(gt_with_ori)
    gt_bottleneck4 = nn.MaxPool2d(8, stride=8)(gt_with_ori)
    gt_bottleneck5 = nn.MaxPool2d(4, stride=4)(gt_with_ori)
    gt_bottleneck6 = nn.MaxPool2d(2, stride=2)(gt_with_ori)

    gt_flattened = gt_flattened = torch.flatten(gt, start_dim=1)
    gt_flattened = gt_flattened / torch.sum(gt_flattened, dim=1, keepdim=True)

    loss_ori = orientation_loss(ori, gt_orientation, gt)
    loss_infoNCE = infoNCELoss(torch.flatten(matching_score_stacked, start_dim=1), torch.flatten(gt_bottleneck, start_dim=1))
    loss_infoNCE2 = infoNCELoss(torch.flatten(matching_score_stacked2, start_dim=1), torch.flatten(gt_bottleneck2, start_dim=1))
    loss_infoNCE3 = infoNCELoss(torch.flatten(matching_score_stacked3, start_dim=1), torch.flatten(gt_bottleneck3, start_dim=1))
    loss_infoNCE4 = infoNCELoss(torch.flatten(matching_score_stacked4, start_dim=1), torch.flatten(gt_bottleneck4, start_dim=1))
    loss_infoNCE5 = infoNCELoss(torch.flatten(matching_score_stacked5, start_dim=1), torch.flatten(gt_bottleneck5, start_dim=1))
    loss_infoNCE6 = infoNCELoss(torch.flatten(matching_score_stacked6, start_dim=1), torch.flatten(gt_bottleneck6, start_dim=1))
    loss_ce =  cross_entropy_loss(logits_flattened, gt_flattened)

    weighted_infoNCE = weight_infoNCE*(loss_infoNCE+loss_infoNCE2+loss_infoNCE3+loss_infoNCE4+loss_infoNCE5+loss_infoNCE6)/6 
    loss = loss_ce + weighted_infoNCE + weight_ori*loss_ori
    return loss
