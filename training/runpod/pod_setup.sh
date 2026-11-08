#!/usr/bin/env bash
# Runs on the pod (via runpod_train.py push). Installs training deps and makes
# sure the legacy detector data is unpacked on the network volume.
#
# Volume layout (/workspace is the network volume, so it survives the pod):
#   /workspace/legacy/corner_detector.zip   uploaded once (see docs runbook)
#   /workspace/legacy/corner_detector/      unpacked from it on first setup
#   /workspace/datasets/<name>/             exported datasets (pushed per run)
#   /workspace/code/                        training code (replaced each push)
#   /workspace/runs/<run>/                  outputs, pulled back afterwards
set -euo pipefail

# Ubuntu 24.04 images mark the system Python as externally managed (PEP 668);
# a throwaway container is the one place overriding that is fine
python -m pip install -q --break-system-packages \
    "ultralytics==8.4.12" "openvino>=2024.0.0" opencv-python-headless

LEGACY=/workspace/legacy
if [ ! -d "$LEGACY/corner_detector/train/images" ]; then
    if [ -f "$LEGACY/corner_detector.zip" ]; then
        echo "Unpacking legacy detector data (one-off, a few minutes)..."
        mkdir -p "$LEGACY"
        python -c "import zipfile; zipfile.ZipFile('$LEGACY/corner_detector.zip').extractall('$LEGACY')"
    else
        echo "WARNING: no legacy detector data at $LEGACY; detector will train on new tiles only"
    fi
fi
[ -d "$LEGACY/corner_detector/train/images" ] && \
    echo "Legacy detector train images: $(ls "$LEGACY/corner_detector/train/images" | wc -l)"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Setup OK"
