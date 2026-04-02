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
from src.models import ResidualFittedModel, fit_segmented_regressor
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
    min_train_issue_days: int = 365
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
    model_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    paths = default_project_paths(project_root)
    metadata = load_system_metadata(paths)
    selected_specs = _filter_model_specs(model_specs, model_names=model_names)
    ensure_dir(artifacts_dir)
    ensure_dir(artifacts_dir / "models")
    ensure_dir(artifacts_dir / "metrics")
    ensure_dir(artifacts_dir / "predictions")
    ensure_dir(artifacts_dir / "datasets")

    rows: list[dict[str, object]] = []
    for spec in selected_specs:
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
    use_segmented_model = bool(spec.day_offset == 0 and "lead_bucket_code" in frame.columns)
    residual_anchor_col = choose_residual_anchor_feature(frame, spec=spec)
    first_train_weights = build_training_weights(first_train, spec=spec)
    segment_backend_overrides: dict[int, str] = {}
    if use_segmented_model:
        selected_backend, segment_backend_overrides = choose_segmented_backends(
            train_df=first_train,
            val_df=first_val,
            random_state=train_config.random_state,
            system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
            anchor_feature_col=residual_anchor_col,
            train_weights=first_train_weights,
        )
    else:
        selected_backend = choose_backend(
            spec=spec,
            train_df=first_train,
            val_df=first_val,
            random_state=train_config.random_state,
            system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
            segmented=False,
            anchor_feature_col=residual_anchor_col,
            train_weights=first_train_weights,
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
        train_weights = build_training_weights(train_df, spec=spec)
        y_train_fit = build_training_target(y_train, X_train, anchor_feature_col=residual_anchor_col)
        model_config = ModelConfig(
            random_state=train_config.random_state,
            backend=selected_backend,
        )
        if use_segmented_model:
            fitted_model = fit_segmented_regressor(
                X_train,
                y_train_fit,
                segment_col="lead_bucket_code",
                config=model_config,
                min_segment_rows=max(500, train_config.min_train_rows),
                segment_backend_overrides=segment_backend_overrides,
                sample_weight=train_weights,
            )
        else:
            fitted_model = fit_regressor(
                X_train,
                y_train_fit,
                config=model_config,
                sample_weight=train_weights,
            )
        model = wrap_with_residual_anchor(fitted_model, anchor_feature_col=residual_anchor_col)
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
    lead_bucket_metrics = pd.DataFrame()
    if "lead_bucket_label" in validation_backtest.columns:
        lead_bucket_metrics = evaluate_by_group(
            validation_backtest,
            validation_backtest["y_pred"].to_numpy(),
            group_col="lead_bucket_label",
            baseline_col="baseline_previous_day_power_w",
            system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
        )
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
    full_train_weights = build_training_weights(frame, spec=spec)
    y_full_fit = build_training_target(y_full, X_full, anchor_feature_col=residual_anchor_col)
    final_model_config = ModelConfig(
        random_state=train_config.random_state,
        backend=selected_backend,
    )
    if use_segmented_model:
        fitted_final_model = fit_segmented_regressor(
            X_full,
            y_full_fit,
            segment_col="lead_bucket_code",
            config=final_model_config,
            min_segment_rows=max(2_000, train_config.min_train_rows),
            segment_backend_overrides=segment_backend_overrides,
            sample_weight=full_train_weights,
        )
    else:
        fitted_final_model = fit_regressor(
            X_full,
            y_full_fit,
            config=final_model_config,
            sample_weight=full_train_weights,
        )
    final_model = wrap_with_residual_anchor(fitted_final_model, anchor_feature_col=residual_anchor_col)

    model_path = model_dir / "model.joblib"
    model_meta = {
        "name": spec.name,
        "day_offset": spec.day_offset,
        "issue_schedule": spec.issue_schedule,
        "backend": final_model.backend,
        "selected_backend_request": selected_backend,
        "segment_backends": getattr(final_model, "segment_backends", None),
        "feature_columns": feature_cols,
        "daylight_only_targets": True,
        "segmented_by": "lead_bucket_code" if use_segmented_model else None,
        "residual_anchor_feature": residual_anchor_col,
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
    if not lead_bucket_metrics.empty:
        lead_bucket_metrics.to_csv(metrics_dir / f"{spec.name}_lead_bucket_metrics.csv", index=False)

    metrics_payload = {
        "model": model_meta,
        "train_rows": int(len(frame)),
        "validation_rows": int(len(validation_backtest)),
        "backtest_overall_vs_daily": overall_backtest_metrics,
        "backtest_overall_vs_persistence": persistence_backtest_metrics,
        "backtest_overall_vs_weekly": weekly_backtest_metrics,
        "fold_metrics": fold_metric_rows,
        "lead_bucket_metrics": lead_bucket_metrics.to_dict(orient="records") if not lead_bucket_metrics.empty else [],
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
    segmented: bool,
    anchor_feature_col: str | None = None,
    train_weights: pd.Series | None = None,
) -> str:
    candidates = ["xgboost", "lightgbm"]
    results: list[tuple[str, float]] = []
    train_sample = _sample_frame_for_backend_selection(train_df, max_rows=250_000)
    val_sample = _sample_frame_for_backend_selection(val_df, max_rows=100_000)
    X_train, y_train, _, _ = prepare_model_matrix(train_sample)
    X_val, _, _, _ = prepare_model_matrix(val_sample)
    y_train_fit = build_training_target(y_train, X_train, anchor_feature_col=anchor_feature_col)
    train_weight_sample = None
    if train_weights is not None:
        train_weight_sample = train_weights.loc[train_sample.index].copy()
    for backend in candidates:
        model_config = ModelConfig(random_state=random_state, backend=backend)
        if segmented and "lead_bucket_code" in X_train.columns:
            fitted_model = fit_segmented_regressor(
                X_train,
                y_train_fit,
                segment_col="lead_bucket_code",
                config=model_config,
                min_segment_rows=max(250, int(len(X_train) * 0.01)),
                sample_weight=train_weight_sample,
            )
        else:
            fitted_model = fit_regressor(
                X_train,
                y_train_fit,
                config=model_config,
                sample_weight=train_weight_sample,
            )
        model = wrap_with_residual_anchor(fitted_model, anchor_feature_col=anchor_feature_col)
        y_pred = clip_physical_predictions(
            predict_regressor(model, X_val),
            system_capacity_w=system_capacity_w,
        )
        metrics = evaluate_prediction_frame(
            val_sample,
            y_pred,
            baseline_col="baseline_previous_day_power_w",
            system_capacity_w=system_capacity_w,
        )
        results.append((model.backend, float(metrics["mae_w"])))

    if not results:
        return "auto"

    results = sorted(results, key=lambda item: item[1])
    return results[0][0]


def choose_segmented_backends(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    random_state: int,
    system_capacity_w: float,
    anchor_feature_col: str | None = None,
    train_weights: pd.Series | None = None,
) -> tuple[str, dict[int, str]]:
    candidate_backends = ["xgboost", "lightgbm", "sklearn_hist_gradient_boosting"]
    fallback_train = _sample_frame_for_backend_selection(train_df, max_rows=300_000)
    fallback_val = _sample_frame_for_backend_selection(val_df, max_rows=120_000)
    X_train, y_train, _, _ = prepare_model_matrix(fallback_train)
    X_val, _, _, _ = prepare_model_matrix(fallback_val)
    y_train_fit = build_training_target(y_train, X_train, anchor_feature_col=anchor_feature_col)
    fallback_train_weights = None
    if train_weights is not None:
        fallback_train_weights = train_weights.loc[fallback_train.index].copy()

    fallback_results: list[tuple[str, float]] = []
    for backend in candidate_backends:
        fitted_model = fit_regressor(
            X_train,
            y_train_fit,
            config=ModelConfig(random_state=random_state, backend=backend),
            sample_weight=fallback_train_weights,
        )
        model = wrap_with_residual_anchor(fitted_model, anchor_feature_col=anchor_feature_col)
        y_pred = clip_physical_predictions(
            predict_regressor(model, X_val),
            system_capacity_w=system_capacity_w,
        )
        metrics = evaluate_prediction_frame(
            fallback_val,
            y_pred,
            baseline_col="baseline_previous_day_power_w",
            system_capacity_w=system_capacity_w,
        )
        fallback_results.append((model.backend, float(metrics["mae_w"])))

    fallback_results.sort(key=lambda item: item[1])
    fallback_backend = fallback_results[0][0] if fallback_results else "auto"

    segment_backends: dict[int, str] = {}
    if "lead_bucket_code" not in train_df.columns or "lead_bucket_code" not in val_df.columns:
        return fallback_backend, segment_backends

    for segment_value in sorted(train_df["lead_bucket_code"].dropna().unique()):
        train_part = train_df.loc[train_df["lead_bucket_code"] == segment_value].copy()
        val_part = val_df.loc[val_df["lead_bucket_code"] == segment_value].copy()
        if len(train_part) < 1_000 or len(val_part) < 250:
            continue

        train_part = _sample_frame_for_backend_selection(train_part, max_rows=150_000)
        val_part = _sample_frame_for_backend_selection(val_part, max_rows=60_000)
        X_train, y_train, _, _ = prepare_model_matrix(train_part)
        X_val, _, _, _ = prepare_model_matrix(val_part)
        y_train_fit = build_training_target(y_train, X_train, anchor_feature_col=anchor_feature_col)
        part_train_weights = None
        if train_weights is not None:
            part_train_weights = train_weights.loc[train_part.index].copy()

        results: list[tuple[str, float]] = []
        for backend in candidate_backends:
            fitted_model = fit_regressor(
                X_train,
                y_train_fit,
                config=ModelConfig(random_state=random_state, backend=backend),
                sample_weight=part_train_weights,
            )
            model = wrap_with_residual_anchor(fitted_model, anchor_feature_col=anchor_feature_col)
            y_pred = clip_physical_predictions(
                predict_regressor(model, X_val),
                system_capacity_w=system_capacity_w,
            )
            metrics = evaluate_prediction_frame(
                val_part,
                y_pred,
                baseline_col="baseline_previous_day_power_w",
                system_capacity_w=system_capacity_w,
            )
            results.append((model.backend, float(metrics["mae_w"])))

        if results:
            results.sort(key=lambda item: item[1])
            segment_backends[int(segment_value)] = results[0][0]

    return fallback_backend, segment_backends


def _sample_frame_for_backend_selection(frame: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=42).sort_values("origin_time")


def choose_residual_anchor_feature(frame: pd.DataFrame, *, spec: DayModelSpec) -> str | None:
    if spec.day_offset != 0:
        return None
    preferred_cols = (
        "feature_intraday_fcst_adjusted_blend_w",
        "feature_intraday_baseline_blend_w",
        "feature_persistence_clear_sky_scaled_w",
        "feature_baseline_issue_persistence_w",
    )
    for col in preferred_cols:
        if col in frame.columns:
            return col
    return None


def build_training_target(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    anchor_feature_col: str | None,
) -> pd.Series:
    if not anchor_feature_col or anchor_feature_col not in X.columns:
        return y.copy()
    return y.astype(float) - X[anchor_feature_col].astype(float)


def wrap_with_residual_anchor(
    model,
    *,
    anchor_feature_col: str | None,
):
    if not anchor_feature_col:
        return model
    return ResidualFittedModel(
        backend=f"residual_{model.backend}",
        residual_model=model,
        anchor_feature_col=anchor_feature_col,
    )


def build_training_weights(frame: pd.DataFrame, *, spec: DayModelSpec) -> pd.Series | None:
    if spec.day_offset != 0:
        return None

    weights = np.ones(len(frame), dtype=float)
    if "lead_bucket_label" in frame.columns:
        bucket_weights = frame["lead_bucket_label"].map(
            {
                "0_1h": 4.0,
                "1_3h": 2.5,
                "3_6h": 1.6,
                "6h_plus": 1.0,
            }
        ).fillna(1.0)
        weights *= bucket_weights.to_numpy(dtype=float)
    elif "lead_hours" in frame.columns:
        lead_hours = frame["lead_hours"].to_numpy(dtype=float)
        weights *= np.where(
            lead_hours <= 1.0,
            4.0,
            np.where(lead_hours <= 3.0, 2.5, np.where(lead_hours <= 6.0, 1.6, 1.0)),
        )

    if "target_clear_sky_capacity_ratio" in frame.columns:
        peak_weight = 1.0 + 0.75 * np.clip(frame["target_clear_sky_capacity_ratio"].to_numpy(dtype=float), 0.0, 1.0)
        weights *= peak_weight

    return pd.Series(weights, index=frame.index, dtype=float)


def _filter_model_specs(
    model_specs: tuple[DayModelSpec, ...],
    *,
    model_names: tuple[str, ...] | None,
) -> tuple[DayModelSpec, ...]:
    if not model_names:
        return model_specs
    requested = {name.strip() for name in model_names if name.strip()}
    selected = tuple(spec for spec in model_specs if spec.name in requested)
    if not selected:
        raise ValueError(f"No matching model names found in requested subset: {sorted(requested)}")
    return selected


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
