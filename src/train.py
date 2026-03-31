from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.config import SYSTEM_CAPACITY_W
from src.data_loader import (
    build_horizon_training_frame,
    default_project_paths,
    load_system_metadata,
)
from src.evaluate import (
    aggregate_metrics_table,
    baseline_daily,
    baseline_persistence,
    evaluate_predictions,
)
from src.feature_engineering import (
    DEFAULT_FEATURE_CONFIG,
    QUICK_TEST_FEATURE_CONFIG,
    build_feature_frame,
    prepare_model_matrix,
)
from src.models import ModelConfig, fit_regressor, predict_regressor
from src.splits import SplitConfig, time_train_validation_split


@dataclass(frozen=True)
class ModelSpec:
    name: str
    horizon_steps: int
    require_forecast_archive: bool

    @property
    def horizon_hours(self) -> float:
        return self.horizon_steps / 4.0


DEFAULT_MODEL_SPECS = (
    ModelSpec(name="same_day_intraday", horizon_steps=4, require_forecast_archive=False),
    ModelSpec(name="day_plus_1", horizon_steps=96, require_forecast_archive=True),
    ModelSpec(name="day_plus_2", horizon_steps=192, require_forecast_archive=True),
    ModelSpec(name="day_plus_3", horizon_steps=288, require_forecast_archive=True),
)


@dataclass(frozen=True)
class TrainConfig:
    validation_days: int = 30
    min_train_rows: int = 500
    min_val_rows: int = 100
    start_date: str | None = None
    end_date: str | None = None
    quick_test: bool = False
    random_state: int = 42


def train_all_models(
    *,
    artifacts_dir: Path,
    model_specs: tuple[ModelSpec, ...] = DEFAULT_MODEL_SPECS,
    train_config: TrainConfig = TrainConfig(),
    project_root: Path | None = None,
) -> pd.DataFrame:
    paths = default_project_paths(project_root)
    metadata = load_system_metadata(paths)
    ensure_dir(artifacts_dir)
    ensure_dir(artifacts_dir / "models")
    ensure_dir(artifacts_dir / "metrics")
    ensure_dir(artifacts_dir / "predictions")
    ensure_dir(artifacts_dir / "datasets")

    rows: list[dict[str, object]] = []
    for spec in model_specs:
        row = train_single_model(
            spec=spec,
            train_config=train_config,
            artifacts_dir=artifacts_dir,
            paths=paths,
            latitude=metadata.latitude,
            longitude=metadata.longitude,
            system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
        )
        rows.append(row)

    metrics = aggregate_metrics_table(rows)
    metrics_path = artifacts_dir / "metrics" / "all_models_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    return metrics


def train_single_model(
    *,
    spec: ModelSpec,
    train_config: TrainConfig,
    artifacts_dir: Path,
    paths,
    latitude: float,
    longitude: float,
    system_capacity_w: float,
) -> dict[str, object]:
    frame = build_horizon_training_frame(
        horizon_steps=spec.horizon_steps,
        paths=paths,
        include_history_weather=True,
        include_leakage_safe_forecast=spec.require_forecast_archive,
        drop_missing_forecast=spec.require_forecast_archive,
    )
    frame = filter_date_range(frame, train_config.start_date, train_config.end_date)
    if frame.empty:
        raise ValueError(f"{spec.name}: no rows after date filtering.")

    feature_cfg = QUICK_TEST_FEATURE_CONFIG if train_config.quick_test else DEFAULT_FEATURE_CONFIG
    feat = build_feature_frame(
        frame,
        config=feature_cfg,
        latitude=latitude,
        longitude=longitude,
    )
    split_cfg = SplitConfig(
        validation_days=train_config.validation_days,
        purge_hours=max(1, int(math.ceil(spec.horizon_hours))),
    )
    train_df, val_df = time_train_validation_split(feat, config=split_cfg)

    if len(train_df) < train_config.min_train_rows:
        raise ValueError(f"{spec.name}: train set too small ({len(train_df)} rows).")
    if len(val_df) < train_config.min_val_rows:
        raise ValueError(f"{spec.name}: validation set too small ({len(val_df)} rows).")

    X_train, y_train, meta_train, feature_cols = prepare_model_matrix(train_df)
    X_val, y_val, meta_val, _ = prepare_model_matrix(val_df)

    model = fit_regressor(
        X_train,
        y_train,
        config=ModelConfig(random_state=train_config.random_state),
    )
    y_pred = predict_regressor(model, X_val)

    baseline_persist = baseline_persistence(val_df)
    baseline_day = baseline_daily(val_df)
    metrics_persist = evaluate_predictions(
        y_val.to_numpy(),
        y_pred,
        baseline_pred=baseline_persist,
        system_capacity_w=system_capacity_w,
    )
    metrics_day = evaluate_predictions(
        y_val.to_numpy(),
        y_pred,
        baseline_pred=baseline_day,
        system_capacity_w=system_capacity_w,
    )

    model_dir = ensure_dir(artifacts_dir / "models" / spec.name)
    metrics_dir = ensure_dir(artifacts_dir / "metrics")
    pred_dir = ensure_dir(artifacts_dir / "predictions")
    data_dir = ensure_dir(artifacts_dir / "datasets")

    model_path = model_dir / "model.joblib"
    model_meta = {
        "name": spec.name,
        "horizon_steps": spec.horizon_steps,
        "horizon_hours": spec.horizon_hours,
        "backend": model.backend,
        "feature_columns": feature_cols,
    }
    joblib.dump(
        {
            "model_meta": model_meta,
            "fitted_model": model,
        },
        model_path,
    )

    train_df.to_csv(data_dir / f"{spec.name}_train.csv", index=False)
    val_df.to_csv(data_dir / f"{spec.name}_validation.csv", index=False)

    pred_frame = meta_val.copy()
    pred_frame["y_pred"] = y_pred
    pred_frame["baseline_persistence"] = baseline_persist
    pred_frame["baseline_daily"] = baseline_day
    pred_frame.to_csv(pred_dir / f"{spec.name}_validation_predictions.csv", index=False)

    metrics_payload = {
        "model": model_meta,
        "validation_rows": int(len(y_val)),
        "metrics_vs_persistence": metrics_persist,
        "metrics_vs_daily": metrics_day,
    }
    (metrics_dir / f"{spec.name}_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    return {
        "model_name": spec.name,
        "horizon_steps": spec.horizon_steps,
        "horizon_hours": spec.horizon_hours,
        "backend": model.backend,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "mae_w": float(metrics_day["mae_w"]),
        "rmse_w": float(metrics_day["rmse_w"]),
        "bias_w": float(metrics_day["bias_w"]),
        "skill_vs_daily": float(metrics_day.get("skill_vs_baseline", 0.0)),
        "skill_vs_persistence": float(metrics_persist.get("skill_vs_baseline", 0.0)),
        "model_path": str(model_path),
    }


def filter_date_range(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    out = frame.copy()
    if start_date:
        out = out.loc[out["origin_time"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        out = out.loc[out["origin_time"] <= pd.Timestamp(end_date)].copy()
    return out.reset_index(drop=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
