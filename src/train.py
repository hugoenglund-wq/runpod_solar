from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.config import SYSTEM_CAPACITY_W
from src.data_loader import default_project_paths, load_system_metadata
from src.day_features import (
    DEFAULT_FEATURE_CONFIG,
    QUICK_TEST_FEATURE_CONFIG,
    prepare_model_matrix,
)
from src.day_frames import DEFAULT_DAY_MODEL_SPECS, DayModelSpec, build_day_model_frame
from src.evaluate import (
    aggregate_metrics_table,
    clip_physical_predictions,
    evaluate_by_group,
    evaluate_prediction_frame,
)
from src.models import ModelConfig, fit_regressor, predict_regressor
from src.splits import (
    BacktestConfig,
    build_issue_date_backtest_folds,
    split_frame_for_fold,
    summarize_season_coverage,
)


@dataclass(frozen=True)
class TrainConfig:
    evaluation_fraction: float = 0.20
    max_folds: int = 8
    preferred_window_days: int = 30
    min_train_issue_days: int = 60
    min_train_rows: int = 500
    min_val_rows: int = 100
    start_date: str | None = None
    end_date: str | None = None
    quick_test: bool = False
    random_state: int = 42


def train_all_models(
    *,
    artifacts_dir: Path,
    model_specs: tuple[DayModelSpec, ...] = DEFAULT_DAY_MODEL_SPECS,
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
            metadata=metadata,
        )
        rows.append(row)

    metrics = aggregate_metrics_table(rows)
    metrics_path = artifacts_dir / "metrics" / "all_models_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    return metrics


def train_single_model(
    *,
    spec: DayModelSpec,
    train_config: TrainConfig,
    artifacts_dir: Path,
    paths,
    metadata,
) -> dict[str, object]:
    feature_cfg = QUICK_TEST_FEATURE_CONFIG if train_config.quick_test else DEFAULT_FEATURE_CONFIG
    frame = build_day_model_frame(
        spec,
        paths=paths,
        metadata=metadata,
        feature_config=feature_cfg,
        daylight_only_targets=True,
    )
    frame = filter_date_range(frame, train_config.start_date, train_config.end_date)
    frame = frame.loc[frame["target_power_w"].notna()].copy()
    if frame.empty:
        raise ValueError(f"{spec.name}: no rows after filtering.")

    backtest_cfg = BacktestConfig(
        evaluation_fraction=train_config.evaluation_fraction,
        preferred_window_days=train_config.preferred_window_days,
        max_folds=train_config.max_folds,
        min_train_issue_days=train_config.min_train_issue_days,
        purge_issue_days=max(1, spec.day_offset),
    )
    folds = build_issue_date_backtest_folds(frame, config=backtest_cfg)

    first_train, first_val = split_frame_for_fold(frame, folds[0])
    selected_backend = choose_backend(
        spec=spec,
        train_df=first_train,
        val_df=first_val,
        random_state=train_config.random_state,
        system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
    )

    model_dir = ensure_dir(artifacts_dir / "models" / spec.name)
    metrics_dir = ensure_dir(artifacts_dir / "metrics")
    pred_dir = ensure_dir(artifacts_dir / "predictions")
    data_dir = ensure_dir(artifacts_dir / "datasets")

    fold_metric_rows: list[dict[str, object]] = []
    validation_frames: list[pd.DataFrame] = []
    seasonal_metric_frames: list[pd.DataFrame] = []

    for fold in folds:
        train_df, val_df = split_frame_for_fold(frame, fold)
        if len(train_df) < train_config.min_train_rows:
            raise ValueError(f"{spec.name}/{fold.fold_name}: train set too small ({len(train_df)} rows).")
        if len(val_df) < train_config.min_val_rows:
            raise ValueError(f"{spec.name}/{fold.fold_name}: validation set too small ({len(val_df)} rows).")

        X_train, y_train, _, feature_cols = prepare_model_matrix(train_df)
        X_val, _, meta_val, _ = prepare_model_matrix(val_df)
        model = fit_regressor(
            X_train,
            y_train,
            config=ModelConfig(
                random_state=train_config.random_state,
                backend=selected_backend,
            ),
        )
        y_pred = clip_physical_predictions(
            predict_regressor(model, X_val),
            system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
        )

        overall_metrics = evaluate_prediction_frame(
            val_df,
            y_pred,
            baseline_col="baseline_previous_day_power_w",
            system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
        )
        persistence_metrics = evaluate_prediction_frame(
            val_df,
            y_pred,
            baseline_col="baseline_issue_persistence_w",
            system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
        )
        weekly_metrics = evaluate_prediction_frame(
            val_df,
            y_pred,
            baseline_col="baseline_previous_week_power_w",
            system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
        )

        fold_metric_rows.append(
            {
                "fold_name": fold.fold_name,
                "fold_id": fold.fold_id,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(val_df)),
                "backend": model.backend,
                "mae_w": float(overall_metrics["mae_w"]),
                "rmse_w": float(overall_metrics["rmse_w"]),
                "bias_w": float(overall_metrics["bias_w"]),
                "skill_vs_daily": float(overall_metrics.get("skill_vs_baseline", 0.0)),
                "skill_vs_persistence": float(persistence_metrics.get("skill_vs_baseline", 0.0)),
                "skill_vs_weekly": float(weekly_metrics.get("skill_vs_baseline", 0.0)),
            }
        )

        pred_frame = meta_val.copy()
        pred_frame["fold_name"] = fold.fold_name
        pred_frame["model_name"] = spec.name
        pred_frame["y_pred"] = y_pred
        pred_frame["target_power_w"] = val_df["target_power_w"].astype(float).to_numpy()
        pred_frame["target_season"] = val_df["target_season"].astype(str).to_numpy()
        validation_frames.append(pred_frame)

        seasonal_metrics = evaluate_by_group(
            val_df,
            y_pred,
            group_col="target_season",
            baseline_col="baseline_previous_day_power_w",
            system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
        )
        seasonal_metrics["fold_name"] = fold.fold_name
        seasonal_metric_frames.append(seasonal_metrics)

    validation_backtest = pd.concat(validation_frames, ignore_index=True)
    seasonal_metrics_all = pd.concat(seasonal_metric_frames, ignore_index=True)
    overall_backtest_metrics = evaluate_prediction_frame(
        validation_backtest,
        validation_backtest["y_pred"].to_numpy(),
        baseline_col="baseline_previous_day_power_w",
        system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
    )
    persistence_backtest_metrics = evaluate_prediction_frame(
        validation_backtest,
        validation_backtest["y_pred"].to_numpy(),
        baseline_col="baseline_issue_persistence_w",
        system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
    )
    weekly_backtest_metrics = evaluate_prediction_frame(
        validation_backtest,
        validation_backtest["y_pred"].to_numpy(),
        baseline_col="baseline_previous_week_power_w",
        system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
    )

    season_coverage = summarize_season_coverage(validation_backtest, season_col="target_season")
    warnings: list[str] = []
    if season_coverage["season_count"] < 4:
        warnings.append(
            "Validation data does not span all four seasons with the currently available leakage-safe inputs."
        )

    X_full, y_full, _, feature_cols = prepare_model_matrix(frame)
    final_model = fit_regressor(
        X_full,
        y_full,
        config=ModelConfig(
            random_state=train_config.random_state,
            backend=selected_backend,
        ),
    )

    model_path = model_dir / "model.joblib"
    model_meta = {
        "name": spec.name,
        "day_offset": spec.day_offset,
        "issue_schedule": spec.issue_schedule,
        "backend": final_model.backend,
        "selected_backend_request": selected_backend,
        "feature_columns": feature_cols,
        "daylight_only_targets": True,
        "season_coverage": season_coverage,
        "warnings": warnings,
    }
    joblib.dump(
        {
            "model_meta": model_meta,
            "fitted_model": final_model,
        },
        model_path,
    )

    frame.to_csv(data_dir / f"{spec.name}_train_pool.csv.gz", index=False, compression="gzip")
    validation_backtest.to_csv(
        data_dir / f"{spec.name}_validation_backtest.csv.gz",
        index=False,
        compression="gzip",
    )
    validation_backtest.to_csv(
        pred_dir / f"{spec.name}_validation_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(fold_metric_rows).to_csv(metrics_dir / f"{spec.name}_fold_metrics.csv", index=False)
    seasonal_metrics_all.to_csv(metrics_dir / f"{spec.name}_seasonal_metrics.csv", index=False)

    metrics_payload = {
        "model": model_meta,
        "train_rows": int(len(frame)),
        "validation_rows": int(len(validation_backtest)),
        "backtest_overall_vs_daily": overall_backtest_metrics,
        "backtest_overall_vs_persistence": persistence_backtest_metrics,
        "backtest_overall_vs_weekly": weekly_backtest_metrics,
        "fold_metrics": fold_metric_rows,
        "season_coverage": season_coverage,
        "warnings": warnings,
    }
    (metrics_dir / f"{spec.name}_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    return {
        "model_name": spec.name,
        "day_offset": spec.day_offset,
        "backend": final_model.backend,
        "train_rows": int(len(frame)),
        "validation_rows": int(len(validation_backtest)),
        "mae_w": float(overall_backtest_metrics["mae_w"]),
        "rmse_w": float(overall_backtest_metrics["rmse_w"]),
        "bias_w": float(overall_backtest_metrics["bias_w"]),
        "skill_vs_daily": float(overall_backtest_metrics.get("skill_vs_baseline", 0.0)),
        "skill_vs_persistence": float(persistence_backtest_metrics.get("skill_vs_baseline", 0.0)),
        "skill_vs_weekly": float(weekly_backtest_metrics.get("skill_vs_baseline", 0.0)),
        "season_count": int(season_coverage["season_count"]),
        "model_path": str(model_path),
    }


def choose_backend(
    *,
    spec: DayModelSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    random_state: int,
    system_capacity_w: float,
) -> str:
    if spec.day_offset == 0:
        return "lightgbm"

    candidates = ["xgboost", "lightgbm"]
    results: list[tuple[str, float]] = []
    X_train, y_train, _, _ = prepare_model_matrix(train_df)
    X_val, _, _, _ = prepare_model_matrix(val_df)
    for backend in candidates:
        model = fit_regressor(
            X_train,
            y_train,
            config=ModelConfig(random_state=random_state, backend=backend),
        )
        y_pred = clip_physical_predictions(
            predict_regressor(model, X_val),
            system_capacity_w=system_capacity_w,
        )
        metrics = evaluate_prediction_frame(
            val_df,
            y_pred,
            baseline_col="baseline_previous_day_power_w",
            system_capacity_w=system_capacity_w,
        )
        results.append((model.backend, float(metrics["mae_w"])))

    if not results:
        return "auto"

    results = sorted(results, key=lambda item: item[1])
    return results[0][0]


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
