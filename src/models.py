from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    max_iter: int = 350
    learning_rate: float = 0.05
    max_depth: int = 8
    backend: str = "auto"


@dataclass(frozen=True)
class FittedModel:
    backend: str
    estimator: object
    feature_columns: list[str]


@dataclass(frozen=True)
class SegmentedFittedModel:
    backend: str
    segment_col: str
    segment_models: dict[int, FittedModel]
    fallback_model: FittedModel


def fit_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: ModelConfig = ModelConfig(),
) -> FittedModel:
    backend = config.backend.lower()
    if backend == "lightgbm":
        lightgbm_model = _try_fit_lightgbm(X_train, y_train, config)
        if lightgbm_model is not None:
            return lightgbm_model
    elif backend == "xgboost":
        xgboost_model = _try_fit_xgboost(X_train, y_train, config)
        if xgboost_model is not None:
            return xgboost_model
    elif backend == "auto":
        preferred = ["xgboost", "lightgbm"] if len(X_train) <= 50_000 else ["lightgbm", "xgboost"]
        for candidate in preferred:
            fitted = (
                _try_fit_xgboost(X_train, y_train, config)
                if candidate == "xgboost"
                else _try_fit_lightgbm(X_train, y_train, config)
            )
            if fitted is not None:
                return fitted

    estimator = HistGradientBoostingRegressor(
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        max_iter=config.max_iter,
        random_state=config.random_state,
        min_samples_leaf=30,
        l2_regularization=0.01,
    )
    estimator.fit(X_train, y_train)
    return FittedModel(
        backend="sklearn_hist_gradient_boosting",
        estimator=estimator,
        feature_columns=list(X_train.columns),
    )


def fit_segmented_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    segment_col: str,
    config: ModelConfig = ModelConfig(),
    min_segment_rows: int = 1_000,
) -> SegmentedFittedModel:
    if segment_col not in X_train.columns:
        raise ValueError(f"Segment column not found in training matrix: {segment_col}")

    base_backend = config.backend
    if base_backend.lower().startswith("segmented_"):
        base_backend = base_backend[len("segmented_") :]
    base_config = ModelConfig(
        random_state=config.random_state,
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        backend=base_backend,
    )

    fallback_model = fit_regressor(X_train, y_train, config=base_config)
    segment_models: dict[int, FittedModel] = {}
    for segment_value, segment_index in X_train.groupby(segment_col, sort=True).groups.items():
        if len(segment_index) < min_segment_rows:
            continue
        X_part = X_train.loc[segment_index].copy()
        y_part = y_train.loc[segment_index].copy()
        segment_models[int(segment_value)] = fit_regressor(X_part, y_part, config=base_config)

    return SegmentedFittedModel(
        backend=f"segmented_{fallback_model.backend}",
        segment_col=segment_col,
        segment_models=segment_models,
        fallback_model=fallback_model,
    )


def predict_regressor(model: FittedModel | SegmentedFittedModel, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, SegmentedFittedModel):
        if model.segment_col not in X.columns:
            return _predict_with_single_model(model.fallback_model, X)

        segment_values = X[model.segment_col].to_numpy()
        predictions = np.full(len(X), np.nan, dtype=float)
        for segment_value, segment_model in model.segment_models.items():
            mask = segment_values == segment_value
            if not np.any(mask):
                continue
            predictions[mask] = _predict_with_single_model(segment_model, X.loc[mask])

        missing_mask = np.isnan(predictions)
        if np.any(missing_mask):
            predictions[missing_mask] = _predict_with_single_model(model.fallback_model, X.loc[missing_mask])
        return predictions

    return _predict_with_single_model(model, X)


def _predict_with_single_model(model: FittedModel, X: pd.DataFrame) -> np.ndarray:
    X_aligned = X.reindex(columns=model.feature_columns, fill_value=np.nan)
    return np.asarray(model.estimator.predict(X_aligned), dtype=float)


def _try_fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig,
) -> FittedModel | None:
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return None

    estimator = LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=config.random_state,
    )
    estimator.fit(X_train, y_train)
    return FittedModel(
        backend="lightgbm",
        estimator=estimator,
        feature_columns=list(X_train.columns),
    )


def _try_fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig,
) -> FittedModel | None:
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None

    estimator = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=config.max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=config.random_state,
        tree_method="hist",
        n_jobs=0,
    )
    estimator.fit(X_train, y_train)
    return FittedModel(
        backend="xgboost",
        estimator=estimator,
        feature_columns=list(X_train.columns),
    )
