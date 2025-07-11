from wrappers.DAFWrapper import DAFWrapper
from wrappers.DAF3Wrapper import DAF3Wrapper
from wrappers.multiMAEWrapper import MultiMAEWrapper
from wrappers.CCVPEWrapper import CCVPEWrapper
from wrappers.groundWrapper import GroundWrapper
from wrappers.L2Wrapper import L2Wrapper
from wrappers.satDAFWrapper import SatDAFWrapper
from wrappers.pruneSatDAFWrapper import PruneSatDAFWrapper
from wrappers.sharedGroundWrapper import SharedGroundWrapper
from wrappers.satGroundWrapper import SatGroundWrapper
from wrappers.scoreMatchWrapper import ScoreMatchWrapper
from wrappers.hardSelectWrapper import HardSelectWrapper
from wrappers.wholeHardSelectWrapper import WholeHardSelectWrapper
from wrappers.fineHardSelectWrapper import FineHardSelectWrapper
from wrappers.softSelectWrapper import SoftSelectWrapper
from wrappers.patchDAFWrapper import PatchDAFWrapper
from wrappers.hardPatchDAFWrapper import HardPatchDAFWrapper
from wrappers.scoreSoftSelectWrapper import ScoreSoftSelectWrapper
from wrappers.scorePatchDAFWrapper import ScorePatchDAFWrapper
from wrappers.convnextScorePatchDAFWrapper import ConvNextScorePatchDAFWrapper
from wrappers.convnextFineScorePatchDAFWrapper import ConvNextFineScorePatchDAFWrapper
from wrappers.scoreFineSoftSelectWrapper import ScoreFinePatchDAFWrapper
from wrappers.patchDAFV3Wrapper import PatchDAFV3Wrapper
from wrappers.softSelectV3Wrapper import SoftSelectV3Wrapper
from wrappers.satScoreMatchWrapper import SatScoreMatchWrapper
from wrappers.osmScoreMatchWrapper import OsmScoreMatchWrapper
from wrappers.HCNetWrapper import HCNetWrapper
from wrappers.KittiCCVPEWrapper import KittiCCVPEWrapper
from functools import partial

"""
The registry is a storage of all experiments and how to run them again

It also has some minor motivation on why we did the experiments
"""
registry = {
    # Base CCVPE experiment that is unimodal
    "CCVPE_sat": partial(CCVPEWrapper, use_osm=False),
    "CCVPE_osm": partial(CCVPEWrapper, use_osm=True),
    "CCVPE_sat_debug": partial(CCVPEWrapper, use_osm=False),
    "CCVPE_osm_debug": partial(CCVPEWrapper, use_osm=True),
    # Experiments to test different strategy to balance OSM and SAT
    # alpha is always 0.5
    "fused_image_ccvpe_alpha0": partial(DAFWrapper, alpha_type=0),
    # alpha is a learnable parameter but result in the same for each test sample
    "fused_image_ccvpe_alpha1": partial(DAFWrapper, alpha_type=1),
    # alpha can adapt to the inputs
    "fused_image_ccvpe_alpha2": partial(DAFWrapper, alpha_type=2),
    # alpha can adapt to the inputs
    "fused_image_ccvpe_alpha2_rerun": partial(DAFWrapper, alpha_type=2),
    # alpha can adapt to the inputs
    "fused_image_ccvpe_alpha2_debug": partial(DAFWrapper, alpha_type=2),
    # Can image loss help with the fusion aspects ? What we are afraid is that one of the modality is not used properly (like osm), by forcing
    # the image loss, we wanted to force the model to use osm features
    "fused_image_ccvpe_image-loss": partial(DAFWrapper, alpha_type=2, fusion_loss=True),
    # The main idea is that we have 3 ccvpe decoder, 1 for fusion, 1 for osm and the last for sat
    # we then apply the loss to each branch to encourage all features to be used
    # The features comes from deformables attention before fusion for single modality
    "fused_image_ccvpe3": partial(DAF3Wrapper, alpha_type=2),
    "fused_image_ccvpe3_debug": partial(DAF3Wrapper, alpha_type=2),
    # Experiments using multiMAE as the fusion module
    "multimae_sat-features_debug": partial(MultiMAEWrapper, use_sat_features=False),
    # Experiments to include the ground in the fusion process
    "ground_fusion_debug": partial(GroundWrapper),
    "ground_fusion": partial(GroundWrapper),
    # Experiment where we use the base deformable attention but with MSE loss instead of CE loss
    "l2loss_debug": partial(L2Wrapper, alpha_type=2, l2_weight=10e3),
    "l2loss": partial(L2Wrapper, alpha_type=2, l2_weight=10e3),
    # We will feed only sat of our deformable fusion module and see how it performs
    # some papers suggest that it degrades when using a pretrained model on multi-modal for uni-modal training
    # no pretrain and OSM input is replaced by tensor of 0
    "single_daf": partial(SatDAFWrapper, pretrained=False, double_input=False),
    # no pretrain and we prune all the network related to OSM
    "prune_single_daf": partial(PruneSatDAFWrapper, pretrained=False, double_input=False),
    # no pretrain and we prune all the network related to OSM
    "prune_single_daf_rerun": partial(PruneSatDAFWrapper, pretrained=False, double_input=False),
    # no pretrain and we prune all the network related to OSM
    "prune_single_daf_debug": partial(PruneSatDAFWrapper, pretrained=False, double_input=False),
    # pretrain and OSM input is replaced by tensor of 0
    "pretrained_single_daf": partial(SatDAFWrapper, pretrained=True, double_input=False),
    # no pretrain and we input twice SAT to the model
    "double_daf": partial(SatDAFWrapper, pretrained=False, double_input=True),
    # no pretrain and we input twice SAT to the model
    "sat_single_daf": partial(SatDAFWrapper, pretrained=False, double_input=False),
    # ground fusion but this time, we use the same transformer for each level and simply conv to adapt the embed dims
    "shared_ground_fusion": partial(SharedGroundWrapper),
    "shared_ground_fusion_debug": partial(SharedGroundWrapper),
    # we just run sat only on this architecture ground fusion to be sure
    "single_ground_fusion": partial(SatGroundWrapper),
    "double_ground_fusion_debug": partial(SatGroundWrapper),
    # We want to guide the fusion with the matching score provided by a first pass on CCVPE
    "score_matching_fusion_debug": partial(ScoreMatchWrapper),
    "score_matching_fusion": partial(ScoreMatchWrapper),
    # best epoch: 10
    # OSM input is replaced by all 0
    "sat_score_matching_fusion": partial(ScoreMatchWrapper, only_sat=True),
    # OSM input is replaced by all 0
    "sat_score_matching_fusion_debug": partial(ScoreMatchWrapper, only_sat=True),
    "random_score_matching_fusion": partial(ScoreMatchWrapper, random=True),
    "random_score_matching_fusion_debug": partial(ScoreMatchWrapper, random=True),
    # We want to guide the fusion with the matching score provided by a first pass on CCVPE
    "score_matching_fusion_rerun": partial(ScoreMatchWrapper),
    # best epoch: 10
    # OSM input is replaced by all 0
    "sat_score_matching_fusion_rerun": partial(ScoreMatchWrapper, only_sat=True),
    "random_score_matching_fusion_rerun": partial(ScoreMatchWrapper, random=True),
    # From previous experiments, we see that the fusion rule may be the cause of confusion for our model
    # Now we will try to have a selection of the tokens that are stronger (one modality or the other)
    # It is operating at a patch size 16 for all scales
    "hard_select_fusion": partial(HardSelectWrapper),
    "hard_select_fusion_debug": partial(HardSelectWrapper),
    # this time, since noise level should be low, sat only should match the performance of base CCVPE
    "sat_hard_select_fusion": partial(HardSelectWrapper, only_sat=True),
    # this time, since noise level should be low, sat only should match the performance of base CCVPE
    "sat_hard_select_fusion_debug": partial(HardSelectWrapper, only_sat=True),
    # If we observe that the model has the tendency to collapse to one modality, we will try different augmentation
    "random_hard_select_fusion_debug": partial(HardSelectWrapper),
    "schedule_hard_select_fusion": partial(
        HardSelectWrapper, schedule_temp=True
    ),  # The selection is hard at around 30k steps, before that it is schedules for training stability
    "schedule_hard_select_fusion_debug": partial(HardSelectWrapper, schedule_temp=True),
    "whole_hard_select_fusion_debug": partial(WholeHardSelectWrapper),
    "whole_hard_select_fusion": partial(
        WholeHardSelectWrapper
    ),  # Still multi scale but at each scale it's one modality or the other
    "fine_hard_select_fusion_debug": partial(
        FineHardSelectWrapper
    ),  # Still multi scale but at each scale we have more than 1 patch
    "fine_hard_select_fusion": partial(
        FineHardSelectWrapper
    ),  # Still multi scale but at each scale we have more than 1 patch
    "fine_hard_select_fusion_rerun": partial(
        FineHardSelectWrapper
    ),  # Still multi scale but at each scale we have more than 1 patch
    "soft_select_fusion_debug": partial(SoftSelectWrapper),  # select at a patch level but soft
    "soft_select_fusion": partial(SoftSelectWrapper),  # select at a patch level but soft
    "soft_select_fusion_v2": partial(
        SoftSelectWrapper
    ),  # select at a patch level but soft, but with rescale
    "soft_patch_DAF_debug": partial(
        PatchDAFWrapper
    ),  # missleading name: its actually score matching guided fusion at a patch level with soft selection
    "soft_patch_DAF": partial(PatchDAFWrapper),
    "soft_patch_DAF_v2_debug": partial(  # v2 fixes softmax scaling
        PatchDAFWrapper
    ),  # missleading name: its actually score matching guided fusion at a patch level with soft selection
    "soft_patch_DAF_v2": partial(PatchDAFWrapper),
    "hard_patch_DAF_debug": partial(
        HardPatchDAFWrapper
    ),  # missleading name: its actually score matching guided fusion at a patch level with hard selection
    "hard_patch_DAF": partial(HardPatchDAFWrapper),
    "score_soft_select_fusion_debug": partial(
        ScoreSoftSelectWrapper
    ),  # select at a patch level but soft and guided by the score
    "score_soft_select_fusion": partial(
        ScoreSoftSelectWrapper
    ),  # select at a patch level but soft and guided by the score
    "score_soft_patch_DAF_debug": partial(  # v2 fixes softmax scaling
        ScorePatchDAFWrapper
    ),  # missleading name: its actually score matching guided fusion at a patch level with soft selection
    "score_soft_patch_DAF": partial(ScorePatchDAFWrapper),
    "convnext_score_soft_patch_DAF_debug": partial(
        ConvNextScorePatchDAFWrapper
    ),  # socre matching with soft select but with convnext
    "convnext_tiny_score_soft_patch_DAF_debug": partial(
        ConvNextScorePatchDAFWrapper, convnext_type="tiny"
    ),  # socre matching with soft select but with convnext
    "convnext_tiny_score_soft_patch_DAF": partial(
        ConvNextScorePatchDAFWrapper, convnext_type="tiny"
    ),  # socre matching with soft select but with convnext, tiny is 28M param (efficientnet is 5M)
    "convnext_small_score_soft_patch_DAF_debug": partial(
        ConvNextScorePatchDAFWrapper, convnext_type="small"
    ),  # socre matching with soft select but with convnext
    "convnext_small_score_soft_patch_DAF": partial(
        ConvNextScorePatchDAFWrapper, convnext_type="small"
    ),  # socre matching with soft select but with convnext, small is 50M param
    "convnext_tiny_fine_score_soft_patch_DAF_debug": partial(
        ConvNextFineScorePatchDAFWrapper, convnext_type="tiny"
    ),  # socre matching with soft select but with convnext
    "convnext_tiny_fine_score_soft_patch_DAF": partial(
        ConvNextFineScorePatchDAFWrapper, convnext_type="tiny"
    ),  # socre matching with soft select but with convnext, every scale has 16 patch (except volume at 8)
    "score_fine_soft_patch_DAF_debug": partial(  # v2 fixes softmax scaling
        ScoreFinePatchDAFWrapper
    ),  # missleading name: its actually score matching guided fusion at a patch level with soft selection
    "score_fine_soft_patch_DAF": partial(ScoreFinePatchDAFWrapper),
    "soft_patch_DAF_v3_debug": partial(  # v2 fixes softmax scaling
        PatchDAFV3Wrapper
    ),  # missleading name: its actually score matching guided fusion at a patch level with soft selection
    "soft_patch_DAF_v3": partial(PatchDAFV3Wrapper),
    "soft_select_fusion_v3": partial(
        SoftSelectV3Wrapper
    ),  # select at a patch level but soft, but with rescale
    "soft_select_fusion_v3_debug": partial(
        SoftSelectV3Wrapper
    ),  # select at a patch level but soft, but with rescale
    "soft_patch_DAF_v3_push_perf": partial(PatchDAFV3Wrapper),
    "soft_patch_DAF_v3_push_perf_debug": partial(PatchDAFV3Wrapper),
    "sat_true_score_matching_fusion_debug": partial(
        SatScoreMatchWrapper
    ),  # real implementation with sat branch only
    "sat_true_score_matching_fusion": partial(
        SatScoreMatchWrapper
    ),  # real implementation with sat branch only
    "osm_true_score_matching_fusion_debug": partial(
        OsmScoreMatchWrapper
    ),  # real implementation with osm branch only
    "osm_true_score_matching_fusion": partial(
        OsmScoreMatchWrapper
    ),  # real implementation with osm branch only
    "CCVPE_sat_cosine_decay": partial(
        CCVPEWrapper, use_osm=False
    ),  # Rerun to test cosine decay impact
    "kitti_CCVPE_sat_debug": partial(KittiCCVPEWrapper),  # Initial run on KITTI
    "kitti_CCVPE_osm_debug": partial(KittiCCVPEWrapper, use_osm=True),  # Initial run on KITTI
    "kitti_CCVPE_sat": partial(KittiCCVPEWrapper),  # Initial run on KITTI
    "kitti_CCVPE_osm": partial(KittiCCVPEWrapper, use_osm=True),  # Initial run on KITTI
}

hcnet_registry = {
    "HC-net": partial(HCNetWrapper),
}


def get_registry(experiment_name):
    try:
        return registry[experiment_name]
    except KeyError:
        names = "".join([f"{keys} \n" for keys in registry])
        print(
            "The experiment name specified does not exist. Add it or pick from :\n",
            names,
        )
        raise KeyError


def get_hcnet_registry(experiment_name):
    try:
        return hcnet_registry[experiment_name]
    except KeyError:
        names = "".join([f"{keys} \n" for keys in hcnet_registry])
        print(
            "The experiment name specified does not exist. Add it or pick from :\n",
            names,
        )
        raise KeyError
