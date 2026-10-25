"""PBR-ReID losses."""

from .adaptive_margin import AdaptiveMarginSeparation, sphere_mean
from .batch_hard_triplet import batch_hard_triplet
from .brm import FeatureMixUp, should_apply_mixup
from .cosine import CosineSimilarityLoss
from .heads import ArcFaceLoss, CircleLoss, CosFaceLoss, VPLArcFaceLoss
from .normalized import NormalizedLoss
from .prototype_losses import CombinedLoss, FeaturePrototypeTripletLoss, PrototypeCenterLoss

__all__ = [
    "AdaptiveMarginSeparation",
    "sphere_mean",
    "batch_hard_triplet",
    "FeatureMixUp",
    "should_apply_mixup",
    "CosineSimilarityLoss",
    "ArcFaceLoss",
    "CosFaceLoss",
    "CircleLoss",
    "VPLArcFaceLoss",
    "NormalizedLoss",
    "CombinedLoss",
    "FeaturePrototypeTripletLoss",
    "PrototypeCenterLoss",
]
