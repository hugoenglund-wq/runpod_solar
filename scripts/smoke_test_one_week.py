#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "scripts" / "train_all_models.py"),
        "--artifacts-dir",
        str(root / "artifacts_smoke_week"),
        "--start-date",
        "2024-03-20",
        "--end-date",
        "2024-03-27",
        "--validation-days",
        "2",
        "--quick-test",
        "--min-train-rows",
        "20",
        "--min-val-rows",
        "10",
    ]
    completed = subprocess.run(cmd, cwd=root, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
