#!/usr/bin/env bash
# Runs on the pod, detached (runpod_train.py start). Trains the classifier,
# the detector, or both, logging to /workspace/runs/<run>/train.log and
# leaving DONE or FAILED there when finished.
#
#   CLS_ARGS="--epochs 60" DET_ARGS="--epochs 40" bash run.sh <dataset-name> <run-name> [classifier|detector|both]
set -uo pipefail

DATASET=/workspace/datasets/$1
RUN=/workspace/runs/$2
WHAT=${3:-both}
read -r -a CLS <<< "${CLS_ARGS:-}"
read -r -a DET <<< "${DET_ARGS:-}"

mkdir -p "$RUN"
rm -f "$RUN/DONE" "$RUN/FAILED"
cd /workspace/code

(
    set -e
    echo "Started $(date -u) on $(hostname): dataset $DATASET, training $WHAT"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    if [ "$WHAT" = classifier ] || [ "$WHAT" = both ]; then
        python -m training.train_classifier --dataset "$DATASET" --out "$RUN/classifier" "${CLS[@]}"
    fi
    if [ "$WHAT" = detector ] || [ "$WHAT" = both ]; then
        LEGACY=()
        [ -d /workspace/legacy/corner_detector/train/images ] && LEGACY=(--legacy /workspace/legacy/corner_detector)
        python -m training.train_detector --dataset "$DATASET" --out "$RUN/detector" "${LEGACY[@]}" "${DET[@]}"
    fi
    echo "Finished $(date -u)"
) >> "$RUN/train.log" 2>&1

if [ $? -eq 0 ]; then touch "$RUN/DONE"; else touch "$RUN/FAILED"; fi
