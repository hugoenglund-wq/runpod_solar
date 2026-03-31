#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[1/4] Updating apt metadata"
apt-get update

echo "[2/4] Installing system dependencies for ecCodes and model training"
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  build-essential \
  libeccodes-dev \
  python3.11 \
  python3.11-venv

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[3/5] Creating virtual environment"
if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "[4/5] Installing Python dependencies"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

echo "[5/5] Verifying NOAA GFS runtime"
"${VENV_DIR}/bin/python" scripts/verify_noaa_runtime.py

echo
echo "RunPod NOAA setup complete."
echo "Next steps:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python scripts/download_noaa_gfs_point_archive.py --output-dir data --workers 6"
echo "  python scripts/train_all_models.py --artifacts-dir artifacts"
