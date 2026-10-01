#!/usr/bin/env bash
# Tech0 bootstrap for Google Colab (T4 / P100, 16GB).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/content/tech0}"

echo "[tech0] Installing project (Colab already has a CUDA torch build)"
pip install -q --upgrade pip
pip install -q -e "${REPO_DIR}[teacher,dev]" || pip install -q -e "${REPO_DIR}" "transformers>=4.45" "accelerate>=0.34" "bitsandbytes>=0.43"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("fp16 supported:", torch.cuda.is_available())
PY

echo "[tech0] (Optional) mount Google Drive for persistent checkpoints:"
echo "  from google.colab import drive; drive.mount('/content/drive')"
echo "  then set  out_dir: /content/drive/MyDrive/tech0/checkpoints/..."
echo "[tech0] Ready."
