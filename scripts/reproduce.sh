#!/usr/bin/env bash
# Runs the training commands behind each table and figure of the paper.
#   scripts/reproduce.sh table1                 # all seven datasets
#   scripts/reproduce.sh table3  [dataset ...]  # CLIP / ImageNet / DINOv2 encoders
#   scripts/reproduce.sh table4  [dataset ...]  # ArcFace / CosFace / Circle / VPL
#   scripts/reproduce.sh table5  [dataset ...]  # loss ladder + MixUp / NG MixUp
#   scripts/reproduce.sh fig3    [dataset]      # BRM hyper-parameter + warmup sweeps
#   scripts/reproduce.sh supp1   [dataset ...]  # ViT-L/16, ViT-H+/16
#   scripts/reproduce.sh supp2   [dataset ...]  # EMA / dual-LR stabilization
#   scripts/reproduce.sh supp3                  # CUHK03-L, VehicleID
set -euo pipefail
cd "$(dirname "$0")/.."
what=${1:-}; shift || true
datasets=("$@"); [ ${#datasets[@]} -eq 0 ] && datasets=(market1501 dukemtmc)

run() { echo "+ uv run train.py $*"; uv run train.py "$@"; }

case "$what" in
  table1) for ds in market1501 dukemtmc msmt17 cuhk03 iust veri vehicleid; do run $ds; done ;;
  table3) for ds in "${datasets[@]}"; do for bb in clip imagenet dinov2; do
            run $ds $bb --system.checkpoint_dir runs/${ds}_$bb; done; done ;;
  table4) for ds in "${datasets[@]}"; do for h in arcface cosface circleloss vpl; do
            run $ds -f configs/ablations/heads/$h.yaml --system.checkpoint_dir runs/${ds}_$h; done; done ;;
  table5) for ds in "${datasets[@]}"; do for r in l1_bht l2_fpt l3_center l4_norm l5_mixup l5_ng_mixup; do
            run $ds -f configs/ablations/losses/$r.yaml --system.checkpoint_dir runs/${ds}_$r; done; run $ds; done ;;
  fig3)   ds=${datasets[0]}; for f in configs/ablations/brm_sweeps/*.yaml configs/ablations/warmup/*.yaml; do
            run $ds -f $f --system.checkpoint_dir runs/${ds}_$(basename $f .yaml); done ;;
  supp1)  for ds in "${datasets[@]}"; do for bb in vitl16 vith16plus; do
            run $ds $bb --system.checkpoint_dir runs/${ds}_$bb; done; done ;;
  supp2)  for ds in "${datasets[@]}"; do for f in configs/ablations/stabilization/*.yaml; do
            run $ds -f $f --system.checkpoint_dir runs/${ds}_$(basename $f .yaml); done; done ;;
  supp3)  run cuhk03_labeled; run vehicleid ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
