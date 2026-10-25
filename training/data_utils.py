"""Datasets, transforms and data loaders."""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from config import UnifiedConfig
from config.manager import resolve_path
from datasets.bases import ImageDataset
from datasets.cuhk03 import CUHK03
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.iustpersonreid import IUSTPersonReID
from datasets.market1501 import Market1501
from datasets.msmt17v2 import MSMT17V2
from datasets.sampler import RandomIdentitySampler
from datasets.vehicleid import VehicleID
from datasets.veri import VeRi
from models.backbones import build_backbone_from_config

DATASET_FACTORY = {
    "market1501": Market1501,
    "dukemtmcreid": DukeMTMCreID,
    "msmt17v2": MSMT17V2,
    "iustpersonreid": IUSTPersonReID,
    "veri": VeRi,
    "cuhk03": CUHK03,
    "vehicleid": VehicleID,
}

# Patch size per backbone — train/test image (H, W) must be divisible by it.
_BACKBONE_PATCH = {"dinov3": 16, "imagenet": 16, "clip": 16, "dinov2": 14}


def data_root(config: UnifiedConfig) -> str:
    return resolve_path(config.data.root_dir, "PBR_DATA_ROOT")


def validate_backbone_geometry(config: UnifiedConfig) -> None:
    p = _BACKBONE_PATCH.get(config.model.backbone)
    if p is None:
        return
    for which, (h, w) in (("size_train", config.data.size_train), ("size_test", config.data.size_test)):
        if h % p or w % p:
            hint = "[392,196]" if p == 14 else "[384,192]"
            raise ValueError(
                f"data.{which}={[h, w]} is not divisible by the {config.model.backbone} patch size ({p}). "
                f"Use e.g. data.{which}={hint}."
            )


def train_collate_fn(batch):
    imgs, pids, camids, viewids, _, _ = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
    )


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, _ = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        pids,
        camids,
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
        img_paths,
    )


def create_transforms(config: UnifiedConfig, base_transform: v2.Compose):
    """Train / test transforms (paper Sec. 4.1).

    Both resize directly to the (H, W) ReID target (aspect preserved,
    train == test geometry). Train adds zero-padding + random crop, random
    horizontal flip and random erasing. ``base_transform`` only contributes the
    backbone's ToImage / ToDtype / Normalize ops.
    """
    ops = base_transform.transforms
    to_image = next(t for t in ops if isinstance(t, v2.ToImage))
    to_dtype = next(t for t in ops if isinstance(t, v2.ToDtype))
    normalize = next(t for t in ops if isinstance(t, v2.Normalize))

    h_tr, w_tr = config.data.size_train
    h_te, w_te = config.data.size_test

    val_transforms = v2.Compose([to_image, v2.Resize((h_te, w_te), antialias=True), to_dtype, normalize])
    train_transforms = v2.Compose(
        [
            to_image,
            v2.Resize((h_tr, w_tr), antialias=True),
            v2.Pad(config.data.padding),
            v2.RandomCrop((h_tr, w_tr)),
            v2.RandomHorizontalFlip(p=config.data.horizontal_flip_prob),
            to_dtype,
            normalize,
            v2.RandomErasing(p=config.data.random_erase_prob),
        ]
    )
    return train_transforms, val_transforms


def build_dataset(config: UnifiedConfig):
    name = config.data.dataset_name
    if name not in DATASET_FACTORY:
        raise ValueError(f"Dataset {name!r} not supported. Available: {sorted(DATASET_FACTORY)}")
    root = data_root(config)
    if name == "cuhk03":
        return CUHK03(
            root=root,
            mode=config.data.cuhk03_mode,
            protocol=config.data.cuhk03_protocol,
            split_index=config.data.cuhk03_split_index,
        )
    if name == "vehicleid":
        return VehicleID(root=root, test_variants=config.data.vehicleid_test_variants)
    return DATASET_FACTORY[name](root=root)


def _loader(config, dataset, collate_fn, worker_init_fn, sampler=None, shuffle=False, batch_size=None):
    return DataLoader(
        dataset,
        batch_size=batch_size or config.data.test_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )


def create_test_loaders(config: UnifiedConfig, dataset, val_transforms, worker_init_fn=None):
    """{variant: (query_loader, gallery_loader)}; 'default' for single-split datasets."""
    test_loaders = {}
    if hasattr(dataset, "test_sets") and isinstance(dataset.test_sets, dict):
        splits = {k: (v["query"], v["gallery"]) for k, v in dataset.test_sets.items()}
    else:
        splits = {"default": (dataset.query, dataset.gallery)}
    for variant, (query, gallery) in splits.items():
        q = _loader(config, ImageDataset(query, val_transforms), val_collate_fn, worker_init_fn)
        g = _loader(config, ImageDataset(gallery, val_transforms), val_collate_fn, worker_init_fn)
        test_loaders[variant] = (q, g)
    return test_loaders


def create_data_loaders(config: UnifiedConfig, worker_init_fn=None):
    """Build the encoder (for its Normalize stats), the dataset and all loaders.

    Returns: (train_loader, test_loaders, num_classes, backbone, base_transform)
    """
    validate_backbone_geometry(config)
    backbone, base_transform = build_backbone_from_config(config)
    train_transforms, val_transforms = create_transforms(config, base_transform)

    dataset = build_dataset(config)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)
    if config.data.sampler.lower() == "pk":
        sampler = RandomIdentitySampler(dataset.train, config.data.train_batch_size, config.data.num_instance)
        train_loader = _loader(
            config, train_set, train_collate_fn, worker_init_fn, sampler=sampler,
            batch_size=config.data.train_batch_size,
        )
    elif config.data.sampler.lower() == "random":
        train_loader = _loader(
            config, train_set, train_collate_fn, worker_init_fn, shuffle=True,
            batch_size=config.data.train_batch_size,
        )
    else:
        raise ValueError(f"data.sampler must be 'pk' or 'random', got {config.data.sampler!r}")

    test_loaders = create_test_loaders(config, dataset, val_transforms, worker_init_fn)
    return train_loader, test_loaders, num_classes, backbone, base_transform
