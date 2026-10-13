#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
