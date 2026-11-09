#!/bin/bash
# Reconstruct the M4Raw test split with a trained FLEX-MRI checkpoint, sweeping
# acceleration rates and contrasts. Score the results with evaluate.py.
#
#   ./test.sh <data_dir> <model_path> [extra python args...]
#
# Use the EMA weights (ema_0.9999_<step>.pt), e.g.
# exp/ema_0.9999_125000.pt from train.sh.
#
# Overridable from the environment, e.g.
#   R_VALUES="4 8" CONTRASTS="T2" GPU=0 ./test.sh /path/to/M4Raw/ exp/ema_0.9999_125000.pt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${1:-}"
MODEL_PATH="${2:-}"
if [ -z "$DATA_DIR" ] || [ -z "$MODEL_PATH" ]; then
    echo "usage: $0 <data_dir> <model_path> [extra python args...]" >&2
    exit 1
fi
shift 2

GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
R_VALUES="${R_VALUES:-2 4 6 8}"
CONTRASTS="${CONTRASTS:-T1 T2 FLAIR}"
PHASE="${PHASE:-test}"
# Architecture: must match training (train.sh).
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_CHANNELS="${NUM_CHANNELS:-128}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-3}"
# Must equal the value train.sh used, for every R: the states each t was
# trained on are defined by J. GAMMA is the base perturbation rate: before
# transition j the estimate loses each k-space coefficient with probability
# GAMMA * (unacquired fraction) * (j-1)/J (0 = deterministic recovery).
TRANSITION_STEPS="${TRANSITION_STEPS:-10}"
GAMMA="${GAMMA:-0.1}"
SEED="${SEED:-0}"
# Default: results/<checkpoint directory name>/<R>x_<contrast>
SAVE_ROOT="${SAVE_ROOT:-results/$(basename "$(dirname "$MODEL_PATH")")}"

for R in $R_VALUES; do
    for contrast in $CONTRASTS; do
        echo "Running: R=${R}x, Contrast=${contrast}, Transitions=${TRANSITION_STEPS}, gamma=${GAMMA}"

        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$SCRIPT_DIR/sample.py" \
            --model_path "$MODEL_PATH" \
            --data_path "$DATA_DIR" \
            --phase "$PHASE" \
            --image_size "$IMAGE_SIZE" \
            --num_channels "$NUM_CHANNELS" \
            --num_res_blocks "$NUM_RES_BLOCKS" \
            --transition_steps "$TRANSITION_STEPS" \
            --gamma "$GAMMA" \
            --seed "$SEED" \
            --save_path "${SAVE_ROOT}/${R}x_${contrast}" \
            --R "$R" \
            --contrast "$contrast" \
            "$@"

    done
done

echo "Score with: python $SCRIPT_DIR/evaluate.py --data_path $DATA_DIR --results $SAVE_ROOT"
