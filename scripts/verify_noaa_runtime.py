#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "requests",
    "sklearn",
    "xgboost",
    "eccodes",
)


def main() -> int:
    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        print(f"Missing Python modules: {', '.join(missing)}")
        return 1

    try:
        import eccodes

        version = eccodes.codes_get_api_version()
    except Exception as exc:  # noqa: BLE001
        print(f"ecCodes import succeeded, but runtime verification failed: {exc}")
        return 1

    print("NOAA runtime verification passed.")
    print(f"ecCodes API version: {version}")
    print(f"Python: {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
