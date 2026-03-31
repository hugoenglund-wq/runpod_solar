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


def predict_regressor(model: FittedModel, X: pd.DataFrame) -> np.ndarray:
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
