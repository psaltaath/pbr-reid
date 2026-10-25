# Reproducing the paper

All commands run from the repository root with datasets and weights in place
(see [DATASETS.md](DATASETS.md) and the README). Each run writes
`train.log`, `checkpoint_epoch_<N>.pth` (new best mAP) and `latest.pth` to
`system.checkpoint_dir` (`runs/<dataset>` by default; override per run with
`--system.checkpoint_dir`). Every number in the paper is single-query cosine
retrieval **without** re-ranking (`evaluation.rerank: false`).

`scripts/reproduce.sh` wraps everything below (`scripts/reproduce.sh table1`,
`... table5 market1501`, ...).

## Table 1 / Table 2 — main results (DINOv3 ViT-B/16)

```bash
for ds in market1501 dukemtmc msmt17 cuhk03 iust veri vehicleid; do
  uv run train.py $ds
done
```

## Supp. Table 1 — backbone capacity

```bash
for bb in vitl16 vith16plus; do
  uv run train.py market1501 $bb --system.checkpoint_dir runs/market1501_$bb
  uv run train.py dukemtmc   $bb --system.checkpoint_dir runs/dukemtmc_$bb
done
```

## Table 3 — other ViT-B encoders

```bash
uv run scripts/download_backbones.py            # once: CLIP / ImageNet / DINOv2 weights
for bb in clip imagenet dinov2; do
  uv run train.py market1501 $bb --system.checkpoint_dir runs/market1501_$bb
  uv run train.py dukemtmc   $bb --system.checkpoint_dir runs/dukemtmc_$bb
done
# vehicles with DINOv2 use the 196x392 geometry: uv run train.py veri dinov2_vehicle
```

## Table 4 — prototype-head objectives

```bash
for h in arcface cosface circleloss vpl; do
  for ds in market1501 dukemtmc; do
    uv run train.py $ds -f configs/ablations/heads/$h.yaml --system.checkpoint_dir runs/${ds}_$h
  done
done
```

## Table 5 — loss terms and BRM

```bash
for r in l1_bht l2_fpt l3_center l4_norm l5_mixup l5_ng_mixup; do
  for ds in market1501 dukemtmc; do
    uv run train.py $ds -f configs/ablations/losses/$r.yaml --system.checkpoint_dir runs/${ds}_$r
  done
done
# the last row (+BRM) is the plain `uv run train.py $ds`
# Sec. 4.3 stop-gradient ablation:
uv run train.py market1501 -f configs/ablations/losses/center_stop_gradient.yaml
```

## Figure 3 — hyper-parameters, warmup, confusable-pair distances

```bash
for f in configs/ablations/brm_sweeps/*.yaml configs/ablations/warmup/*.yaml; do
  uv run train.py market1501 -f $f --system.checkpoint_dir runs/market1501_$(basename $f .yaml)
done
# KDE of prototype distances between confusable pairs (early vs final checkpoint)
uv run tools/confusable_distance_curve.py --dataset market1501 --config market1501 \
    --early_ckpt runs/market1501/checkpoint_epoch_20.pth --early_epoch 20 \
    --final_ckpt runs/market1501/checkpoint_epoch_190.pth --final_epoch 190
```

## Supp. Table 2 — training stabilization

```bash
for f in configs/ablations/stabilization/*.yaml; do
  for ds in market1501 dukemtmc; do
    uv run train.py $ds -f $f --system.checkpoint_dir runs/${ds}_$(basename $f .yaml)
  done
done
```

## Supp. Table 3 — CUHK03-L and VehicleID splits

```bash
uv run train.py cuhk03_labeled
uv run train.py vehicleid                     # reports S, M and L every evaluation
```

## Supp. Figures 1–3 — retrieval comparison (no-BRM vs BRM)

```bash
uv run tools/retrieval_comparison.py --dataset msmt17 --config msmt17 \
    --no_brm_ckpt runs/msmt17_l5_mixup/latest.pth --brm_ckpt runs/msmt17/latest.pth --top_k 10
```

## Evaluating a checkpoint

```bash
uv run eval.py market1501 --checkpoint runs/market1501/latest.pth           # plain
uv run eval.py market1501 --checkpoint runs/market1501/latest.pth --rerank  # + k-reciprocal
```