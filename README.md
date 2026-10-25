<div align="center">

# PBR-ReID

### Learning Discriminative Prototypes via Boundary Repulsion MixUp for Object Re-Identification

**Athena Psalta · Vasileios Tsironis · Konstantinos Karantzalos**

National Technical University of Athens

[![WACV 2027](https://img.shields.io/badge/WACV-2027-1f6feb.svg)](https://wacv.thecvf.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="docs/figs/methodology1.png" width="90%">

</div>

🎉 **Accepted at WACV 2027!** This repository is the official reference implementation.

**PBR-ReID** is a prototype-based object re-identification framework built on a
DINOv3 backbone. Its core component, **Boundary Repulsion MixUp (BRM)**, is a
geometry-aware regularizer. BRM finds pairs of identities the model confuses,
builds an explicit boundary proxy between them and pushes their features away
from it. The same model works for both person and vehicle re-ID. At test time
the prototypes are dropped and retrieval is plain cosine nearest-neighbour
search, with no re-ranking and no camera embeddings.

## ✨ Highlights

- **Learnable identity prototypes:** each identity has one prototype that is
  trained jointly with the encoder. The prototype similarity matrix **S** shows
  which identity pairs are confusable at every training step.
- **Boundary Repulsion MixUp:** MixUp between a confusable pair estimates
  the decision boundary between them. BRM pulls the mixed samples toward that
  boundary and pushes the real features away from it. An adaptive margin keeps
  the prototypes apart.
- **Stable two-stage training:** a prototype warmup with a frozen backbone,
  followed by EMA-stabilized joint fine-tuning with dual learning rates.
- **Simple inference:** cosine nearest-neighbour retrieval on DINOv3
  features, with no re-ranking, camera embeddings or prototypes needed at test
  time.

## 🔍 How it works

<div align="center">
<img src="docs/figs/methodology2.png" width="95%">
</div>

1. **Cosine similarity:** each feature is compared against all identity
   prototypes.
2. **Confusable pair:** a different identity's prototype that scores above the
   threshold τ is marked as confusable with the reference identity.
3. **MixUp:** mixing the two prototypes, and the two features, with weight λ
   gives a proxy for the boundary between the identities.
4. **BRM loss:** the mixed samples are pulled toward the estimated boundary,
   and the real features are pushed out of a margin *m* around it.

The full training objective combines the BRM loss with two triplet losses
(feature-to-prototype and batch-hard between features), a prototype center loss
and a soft unit-norm regularizer.

## 📊 Results at a glance

DINOv3 ViT-B/16, single query, **no re-ranking**. All values are in %.

| Dataset | Domain | mAP | Rank-1 |
|:--|:--:|:--:|:--:|
| Market-1501 | Person | 94.0 | 97.1 |
| DukeMTMC-reID | Person | 86.1 | 93.1 |
| MSMT17 | Person | 79.7 | 91.2 |
| CUHK03-NP (detected) | Person | 88.4 | 89.8 |
| IUST | Person | 55.2 | 63.8 |
| VeRi-776 | Vehicle | 88.0 | 98.1 |
| VehicleID (large) | Vehicle | – | 90.2 |

VehicleID is evaluated with Rank-k only, so no mAP is reported.

The full tables are in the paper.

## 📑 Contents

1. [Installation](#1-installation)
2. [Weights and datasets](#2-weights-and-datasets)
3. [Training and evaluation](#3-training-and-evaluation)
4. [Repository layout](#4-repository-layout)
5. [Citation](#-citation)

## 1. Installation

```bash
git clone --recursive https://github.com/psaltaath/pbr-reid.git && cd pbr-reid
uv sync                      # or: pip install -e . (Python >= 3.12, PyTorch >= 2.8)
```

## 2. Weights and datasets

**DINOv3 weights.** The backbone code is the `facebookresearch/dinov3` submodule
in `third_party/dinov3`. Meta gates the pretrained checkpoints, so download them
from https://github.com/facebookresearch/dinov3 into `weights/dinov3/` and keep
the official filenames, e.g. `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`. To
use a different folder, set `model.dinov3_weights_dir` or
`export PBR_DINOV3_WEIGHTS=/path`.

**Other encoders (Table 3).** `uv run scripts/download_backbones.py` downloads
the CLIP ViT-B/16, ImageNet ViT-B/16 and DINOv2 ViT-B/14 weights into
`weights/backbones/`.

**Datasets.** Place them under `data/`, or `export PBR_DATA_ROOT=/path`, using
the folder names listed in [docs/DATASETS.md](docs/DATASETS.md).

## 3. Training and evaluation

```bash
uv run train.py market1501                 # paper recipe; logs + checkpoints in runs/market1501
uv run train.py veri                       # vehicles use the 192x384 geometry (configs/datasets/veri.yaml)
uv run train.py market1501 vitl16          # stack any configs/**.yaml by name
uv run train.py dukemtmc -f configs/ablations/losses/l5_mixup.yaml --system.seed 123
uv run eval.py market1501 --checkpoint runs/market1501/latest.pth [--rerank]
```

Training Market-1501 with ViT-B/16 takes about a day on one RTX 4090.

<details>
<summary><b>Paper recipe (Sec. 4.1)</b></summary>

`configs/base.yaml` holds these settings:

- **Batches and inputs:** 8 identities × 4 instances, 384×192 inputs.
- **Schedule:** 200 epochs, including 10 warmup epochs.
- **Optimizer:** AdamW with an encoder LR of 1e-5 → 1e-7 (cosine) and a
  prototype LR of 1e-1 (step 15), plus EMA 0.999.
- **Loss weights:** λ_fpt = λ_bht = 1, λ_cen = 0.5, λ_norm = 0.005,
  λ_BRM = 0.4.
- **BRM:** τ = 0.7, α = 0.4, m_BRM = 0.5, β = 0.7, γ_sep = 1.5.

[docs/CONFIG.md](docs/CONFIG.md) maps every setting to its symbol in the paper.

</details>

## 4. Repository layout

```
train.py / eval.py        entry points
configs/                  base.yaml (paper recipe), datasets/, backbones/, ablations/
config/                   typed configuration (dataclasses + YAML + CLI overrides)
datasets/                 Market-1501, DukeMTMC, MSMT17, CUHK03-NP, IUST, VeRi-776, VehicleID loaders + PK sampler
models/                   DINOv3 / DINOv2 / CLIP / ImageNet encoders, PrototypeReID
losses/                   L_fpt, L_cen, L_norm, L_bht, MixUp + BRM, adaptive-margin separation, Table 4 heads
training/                 two-stage Trainer, optimizer factories, data pipeline
evaluation/               mAP / CMC, k-reciprocal re-ranking
tools/                    figure scripts (retrieval comparison, confusable-pair KDE, prototype t-SNE)
scripts/                  download_backbones.py, reproduce.sh
tests/                    CPU unit tests (uv run pytest)
third_party/dinov3        DINOv3 submodule
```

## 📝 Citation

```bibtex
@inproceedings{psalta2027pbrreid,
  title     = {Learning Discriminative Prototypes via Boundary Repulsion MixUp for Object Re-Identification},
  author    = {Psalta, Athena and Tsironis, Vasileios and Karantzalos, Konstantinos},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## 🙏 Acknowledgements

This work builds on [DINOv3](https://github.com/facebookresearch/dinov3) (Meta,
DINOv3 License). 

The dataset loaders follow
[TransReID](https://github.com/damo-cv/TransReID) and
[CLIP-ReID](https://github.com/Syliz517/CLIP-ReID), and the re-ranking follows
Zhong et al. (CVPR 2017).

## 📄 License

The code is released under the MIT License. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
