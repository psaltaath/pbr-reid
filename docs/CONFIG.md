# Configuration

Every run is `configs/base.yaml` (the paper recipe) + zero or more named
configs + optional `--config-file` YAMLs + `--section.key value` overrides, in
that order of precedence:

```bash
uv run train.py market1501                                   # paper recipe
uv run train.py market1501 vitl16                            # + configs/backbones/vitl16.yaml
uv run train.py veri -f configs/ablations/losses/l5_mixup.yaml
uv run train.py msmt17 --system.seed 123 --training.brm_lambda 0.7
uv run train.py --list-configs
```

Named configs are found recursively under `configs/` (`datasets/`, `backbones/`,
`ablations/<group>/`). Lists are written `--data.size_train 192,384`.

## Paths and environment variables

| Config key                   | Env override           | Default               |
|------------------------------|------------------------|-----------------------|
| `data.root_dir`              | `PBR_DATA_ROOT`        | `data`                |
| `model.dinov3_repo_dir`      | `PBR_DINOV3_REPO`      | `third_party/dinov3`  |
| `model.dinov3_weights_dir`   | `PBR_DINOV3_WEIGHTS`   | `weights/dinov3`      |
| `model.backbone_weights_dir` | `PBR_BACKBONE_WEIGHTS` | `weights/backbones`   |

## Paper symbol ↔ config key

| Paper                                   | Key                                                   | Default |
|-----------------------------------------|-------------------------------------------------------|---------|
| λ_fpt, m_fpt, γ_fpt (Eq. 2–4)           | `loss.fpt_weight`, `loss.fpt_margin`, `loss.fpt_similarity_weight_strength` | 1.0, 0.3, 2.0 |
| λ_bht, m_bht (Eq. 5–6)                  | `loss.bht_weight`, `loss.bht_margin`                  | 1.0, 0.3 |
| λ_cen, γ_center^sim, γ_center^var (Eq. 7–11) | `loss.center_weight`, `loss.center_similarity_weight_strength`, `loss.center_variance_weight_strength` | 0.5, 1.0, 2.5 |
| Fréchet mean μ_k (Eq. 7)                | `loss.center_mean_type` (`frechet` / `euclidean_projected`) | `frechet` |
| λ_norm (Eq. 12)                         | `loss.norm_weight`                                    | 0.005 |
| τ, α, m_BRM, β (Eq. 8–13)               | `training.brm_tau`, `training.mixup_alpha`, `training.brm_margin`, `training.brm_beta` | 0.7, 0.4, 0.5, 0.7 |
| λ_BRM (Eq. 1)                           | `training.brm_lambda`                                 | 0.4 |
| γ_sep, σ̂ EMA decay (Eq. 13)             | `training.sep_gamma`, `training.sep_ema_decay`        | 1.5, 0.9 |
| confusable class pairs P_S (Eq. 13)     | `training.sep_pair_scope` (`global` / `batch`)        | `global` |
| K', N_k (Sec. 3.2.1)                    | `data.train_batch_size / data.num_instance`, `data.num_instance` | 8, 4 |
| warmup epochs, total epochs             | `training.prototype_warmup_epochs`, `training.num_epochs` | 10, 200 |
| backbone LR 1e-5 → 1e-7 (cosine)        | `training.encoder_lr`, `training.encoder_min_lr`      | 1e-5, 1e-7 |
| prototype LR 1e-1, step 15              | `training.prototype_lr`, `training.prototype_step_size`, `training.prototype_gamma` | 0.1, 15, 0.5 |
| AdamW weight decay                      | `training.weight_decay`                               | 0.01 |
| Xavier gain, γ_ema (Eq. 14)             | `model.prototype_init_gain`, `model.ema_decay`        | 0.5, 0.999 |
| Table 4 heads                           | `loss.enable_head`, `loss.head` (+ `arcface_*`, `cosface_*`, `circle_*`, `vpl_*`) | off |
| Table 5 NG MixUp                        | `training.mixup_mode: neighbor_hard_target`, `training.mixup_neighbor_k` | — |

Terms are switched with `loss.enable_fpt / enable_bht / enable_center /
enable_norm`, `training.enable_mixup / enable_brm / enable_adaptive_separation`.
L_bht, MixUp and BRM are always disabled during the warmup stage (Sec. 3.2.3).
