#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "[1/4] Updating apt metadata"
apt-get update

echo "[2/4] Installing system dependencies for ecCodes and model training"
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  build-essential \
  libeccodes-dev

echo "[3/4] Installing Python dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "[4/4] Verifying NOAA GFS runtime"
python scripts/verify_noaa_runtime.py

echo
echo "RunPod NOAA setup complete."
echo "Next steps:"
echo "  python scripts/download_noaa_gfs_point_archive.py --output-dir data"
echo "  python scripts/train_all_models.py --artifacts-dir artifacts"
