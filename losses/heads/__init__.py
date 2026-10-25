"""Prototype-head baselines used in Table 4 (each replaces the PBR-ReID objective)."""

from .arcface import ArcFaceLoss
from .circle_loss import CircleLoss
from .cosface import CosFaceLoss
from .vpl import VPLArcFaceLoss

__all__ = ["ArcFaceLoss", "CosFaceLoss", "CircleLoss", "VPLArcFaceLoss"]
