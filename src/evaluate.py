from __future__ import annotations

import math

import numpy as np
import pandas as pd


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    baseline_pred: np.ndarray | None = None,
    system_capacity_w: float | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(np.square(err))))
    bias = float(np.mean(err))
    out = {
        "mae_w": mae,
        "rmse_w": rmse,
        "bias_w": bias,
    }
    if system_capacity_w and system_capacity_w > 0:
        out["nmae_capacity"] = float(mae / system_capacity_w)
        out["nrmse_capacity"] = float(rmse / system_capacity_w)
    if baseline_pred is not None:
        baseline_pred = np.asarray(baseline_pred, dtype=float)
        baseline_mae = float(np.mean(np.abs(baseline_pred - y_true)))
        out["baseline_mae_w"] = baseline_mae
        out["skill_vs_baseline"] = float(1.0 - (mae / baseline_mae)) if baseline_mae > 0 else 0.0
    return out


def baseline_persistence(frame: pd.DataFrame) -> np.ndarray:
    if "power_w" not in frame.columns:
        raise ValueError("Frame must include power_w for persistence baseline.")
    return frame["power_w"].astype(float).to_numpy()


def baseline_daily(frame: pd.DataFrame) -> np.ndarray:
    lag_col = "lag_power_96"
    if lag_col in frame.columns:
        return frame[lag_col].astype(float).fillna(frame["power_w"]).to_numpy()
    return baseline_persistence(frame)


def aggregate_metrics_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
