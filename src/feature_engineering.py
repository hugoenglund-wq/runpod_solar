from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]


DEFAULT_FEATURE_CONFIG = FeatureConfig(
    lags=(1, 2, 4, 8, 16, 96, 192, 672),
    rolling_windows=(4, 16, 96),
)

QUICK_TEST_FEATURE_CONFIG = FeatureConfig(
    lags=(1, 2, 4, 8, 16, 96),
    rolling_windows=(4, 16),
)


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    latitude: float,
    longitude: float,
    timezone_offset_hours: float | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out = _add_time_features(out)
    out = _add_solar_geometry_features(
        out,
        latitude=latitude,
        longitude=longitude,
        timezone_offset_hours=timezone_offset_hours,
    )
    out = _add_power_lag_features(out, config.lags)
    out = _add_power_rolling_features(out, config.rolling_windows)
    out = _add_normalized_power_features(out)
    return out


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = out["origin_time"]
    out["hour"] = ts.dt.hour
    out["minute"] = ts.dt.minute
    out["day_of_week"] = ts.dt.dayofweek
    out["day_of_year"] = ts.dt.dayofyear
    out["month"] = ts.dt.month
    out["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    out["sin_hour"] = (2.0 * math.pi * out["hour"] / 24.0).map(math.sin)
    out["cos_hour"] = (2.0 * math.pi * out["hour"] / 24.0).map(math.cos)
    out["sin_day_of_year"] = (2.0 * math.pi * out["day_of_year"] / 366.0).map(math.sin)
    out["cos_day_of_year"] = (2.0 * math.pi * out["day_of_year"] / 366.0).map(math.cos)
    return out


def _add_power_lag_features(df: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_power_{lag}"] = out["power_w"].shift(lag)
    return out


def _add_power_rolling_features(df: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    shifted = out["power_w"].shift(1)
    for window in windows:
        rolled = shifted.rolling(window=window, min_periods=1)
        out[f"roll_mean_power_{window}"] = rolled.mean()
        out[f"roll_max_power_{window}"] = rolled.max()
        out[f"roll_std_power_{window}"] = rolled.std()
    return out


def _add_normalized_power_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["power_kw"] = out["power_w"] / 1000.0
    out["is_daylight_proxy"] = (out["power_w"] > 5.0).astype(int)
    return out


def _add_solar_geometry_features(
    df: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    timezone_offset_hours: float | None,
) -> pd.DataFrame:
    out = df.copy()
    ts = out["origin_time"]
    day_of_year = ts.dt.dayofyear.to_numpy(dtype=float)
    hour = ts.dt.hour.to_numpy(dtype=float)
    minute = ts.dt.minute.to_numpy(dtype=float)

    tz_offset = timezone_offset_hours
    if tz_offset is None:
        # Fallback: infer standard timezone offset from longitude.
        tz_offset = round(longitude / 15.0)

    gamma = (2.0 * np.pi / 365.0) * (day_of_year - 1.0 + ((hour - 12.0) / 24.0))
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )

    time_offset = eqtime + (4.0 * longitude) - (60.0 * tz_offset)
    true_solar_minutes = (hour * 60.0) + minute + time_offset
    true_solar_minutes = np.mod(true_solar_minutes, 1440.0)
    hour_angle_deg = (true_solar_minutes / 4.0) - 180.0

    lat_rad = np.deg2rad(latitude)
    hour_angle_rad = np.deg2rad(hour_angle_deg)
    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle_rad)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith_rad = np.arccos(cos_zenith)
    elevation_deg = 90.0 - np.rad2deg(zenith_rad)
    cos_zenith_clipped = np.clip(cos_zenith, 0.0, 1.0)

    azimuth_rad = np.arctan2(
        np.sin(hour_angle_rad),
        (np.cos(hour_angle_rad) * np.sin(lat_rad)) - (np.tan(decl) * np.cos(lat_rad)),
    )
    azimuth_deg = (np.rad2deg(azimuth_rad) + 180.0) % 360.0

    out["solar_elevation_deg"] = elevation_deg
    out["solar_azimuth_deg"] = azimuth_deg
    out["solar_cos_zenith"] = cos_zenith_clipped
    out["is_daylight_solar"] = (elevation_deg > 0.0).astype(int)
    return out


def prepare_model_matrix(
    frame: pd.DataFrame,
    *,
    target_col: str = "target_power_w",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    work = frame.copy()
    drop_cols = {
        target_col,
        "origin_time",
        "target_time",
        "forecast_valid_hour",
        "forecast_issue_time",
        "forecast_valid_time",
        "is_missing_target_power",
    }
    meta_cols = ["origin_time", "target_time", target_col]
    if "forecast_available_at_origin" in work.columns:
        meta_cols.append("forecast_available_at_origin")

    feature_candidates = [
        col for col in work.columns if col not in drop_cols and col not in meta_cols
    ]
    numeric_feature_cols = [
        col for col in feature_candidates if pd.api.types.is_numeric_dtype(work[col])
    ]
    X = work.loc[:, numeric_feature_cols].copy()
    y = work[target_col].astype(float).copy()
    meta = work.loc[:, [col for col in meta_cols if col in work.columns]].copy()
    return X, y, meta, numeric_feature_cols
