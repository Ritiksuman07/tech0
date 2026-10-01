#!/usr/bin/env bash
# Tech0 bootstrap for Kaggle notebooks (P100 / 2xT4, 16GB each).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/tech0}"
WORK_DIR="${WORK_DIR:-/kaggle/working}"

echo "[tech0] Installing project (Kaggle already has a CUDA torch build)"
pip install -q --upgrade pip
pip install -q -e "${REPO_DIR}[teacher,dev]" || pip install -q -e "${REPO_DIR}" "transformers>=4.45" "accelerate>=0.34" "bitsandbytes>=0.43"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.device_count(), "device(s)")
PY

echo "[tech0] Persist artefacts to \$WORK_DIR = ${WORK_DIR} (survives 12h session caps)."
echo "[tech0] Ready."
