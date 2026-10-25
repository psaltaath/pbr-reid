"""Dataset and data-loading configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple

from .base import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """Datasets, input geometry, augmentation and sampling (paper Sec. 4.1)."""

    # One of: market1501, dukemtmcreid, msmt17v2, cuhk03, iustpersonreid,
    # veri, vehicleid (see docs/DATASETS.md for the expected folder layout).
    dataset_name: str = "market1501"
    # Root folder holding the dataset folders. Overridable with PBR_DATA_ROOT.
    root_dir: str = "data"

    # CUHK03 (new protocol, Zhong et al. 2017): detected | labeled.
    cuhk03_mode: str = "detected"
    cuhk03_protocol: str = "new"  # new | classic
    cuhk03_split_index: int = 0  # classic protocol only (0-19)

    # VehicleID test splits to evaluate: S (800), M (1600), L (2400), XL, XXL, Full.
    vehicleid_test_variants: List[str] = field(default_factory=lambda: ["S", "M", "L"])

    # Input geometry (H, W). Images are resized directly to this size on both
    # train and test (aspect preserved for 2:1 person crops). Persons: 384x192;
    # vehicles (VeRi, VehicleID): 192x384. Must be divisible by the patch size
    # (16 for DINOv3/CLIP/ImageNet ViT-B/16, 14 for DINOv2 ViT-B/14).
    size_train: Tuple[int, int] = (384, 192)
    size_test: Tuple[int, int] = (384, 192)

    # Train augmentation: random horizontal flip, zero-padding + random crop,
    # random erasing (paper Sec. 4.1).
    horizontal_flip_prob: float = 0.5
    padding: int = 10
    random_erase_prob: float = 0.5

    # Class-balanced PK sampling: B = K' x N_k identities x instances
    # (paper: K' = 8, N_k = 4, B = 32). sampler = "random" gives plain shuffling.
    sampler: str = "pk"  # pk | random
    train_batch_size: int = 32
    num_instance: int = 4

    test_batch_size: int = 128
    num_workers: int = 8
    pin_memory: bool = True
