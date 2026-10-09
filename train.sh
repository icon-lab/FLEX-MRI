#!/bin/bash
# Train FLEX-MRI on M4Raw.
#
#   ./train.sh <data_dir> [extra python args...]
#
# <data_dir> holds multicoil_<phase>/*.h5; the train split's coil maps must be
# cached first (precompute_coil_maps.py --phase train).
#
# Overridable from the environment, e.g.
#   GPU=1 SAVE_DIR=runs/flex ./train.sh /path/to/M4Raw/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${1:-}"
if [ -z "$DATA_DIR" ]; then
    echo "usage: $0 <data_dir> [extra python args...]" >&2
    exit 1
fi
shift

GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
SAVE_DIR="${SAVE_DIR:-exp}"
CONTRAST="${CONTRAST:-all}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
STEPS="${STEPS:-125000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25000}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
# Architecture, shared with test.sh.
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CHANNELS="${NUM_CHANNELS:-128}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-3}"
DROPOUT="${DROPOUT:-0.3}"
# Must match test.sh: t indexes a point on the training trajectory, so a
# different count at sampling changes what every t means.
TRANSITION_STEPS="${TRANSITION_STEPS:-10}"
# Every sample is undersampled with a fresh mask at an acceleration in
# [R_MIN, R_MAX], using the same generator test.sh sweeps over.
R_MIN="${R_MIN:-2.0}"
R_MAX="${R_MAX:-8.0}"
# 8 iterations per optimizer step; 5k-iteration linear warmup, then cosine
# decay to MIN_LR over the remaining iterations.
LR="${LR:-1e-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
MIN_LR="${MIN_LR:-1e-6}"

echo "Training on ${DATA_DIR} -> ${SAVE_DIR}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/train.py" \
    --data_dir "$DATA_DIR" \
    --contrast "$CONTRAST" \
    --num_workers "$NUM_WORKERS" \
    --save_dir "$SAVE_DIR" \
    --log_interval "$LOG_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --lr_anneal_steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --num_channels "$NUM_CHANNELS" \
    --num_res_blocks "$NUM_RES_BLOCKS" \
    --dropout "$DROPOUT" \
    --transition_steps "$TRANSITION_STEPS" \
    --r_min "$R_MIN" \
    --r_max "$R_MAX" \
    --lr "$LR" \
    --accumulation_steps "$ACCUMULATION_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --min_lr "$MIN_LR" \
    --ema_rate 0.9999 \
    "$@"
