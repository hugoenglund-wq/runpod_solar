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

MIN_CLEAR_SKY_PROXY_W = 250.0
LEAD_BUCKET_BOUNDS_HOURS = (1.0, 3.0, 6.0)
LEAD_BUCKET_LABELS = {
    0: "0_1h",
    1: "1_3h",
    2: "3_6h",
    3: "6h_plus",
}


def build_issue_feature_frame(
    issue_frame: pd.DataFrame,
    *,
    config: FeatureConfig,
    latitude: float,
    longitude: float,
    tilt_deg: float | None,
    azimuth_deg: float | None,
    system_capacity_w: float,
) -> pd.DataFrame:
    out = issue_frame.sort_values("origin_time").reset_index(drop=True).copy()
    out = add_timestamp_features(
        out,
        time_col="origin_time",
        prefix="origin",
        latitude=latitude,
        longitude=longitude,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        system_capacity_w=system_capacity_w,
    )
    out = _add_issue_history_features(out, config=config, system_capacity_w=system_capacity_w)
    return out


def add_target_features(
    frame: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    tilt_deg: float | None,
    azimuth_deg: float | None,
    system_capacity_w: float,
) -> pd.DataFrame:
    out = frame.copy()
    out = add_timestamp_features(
        out,
        time_col="target_time",
        prefix="target",
        latitude=latitude,
        longitude=longitude,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        system_capacity_w=system_capacity_w,
    )
    delta = out["target_time"] - out["origin_time"]
    out["lead_minutes"] = delta.dt.total_seconds() / 60.0
    out["lead_hours"] = out["lead_minutes"] / 60.0
    out["lead_steps"] = out["lead_minutes"] / 15.0
    out["target_day_offset"] = (
        out["target_time"].dt.normalize() - out["origin_time"].dt.normalize()
    ).dt.days.astype(int)
    out["remaining_day_minutes_at_issue"] = (
        (out["origin_time"].dt.normalize() + pd.Timedelta(days=1)) - out["origin_time"]
    ).dt.total_seconds() / 60.0
    return out


def add_relative_physics_features(
    frame: pd.DataFrame,
    *,
    system_capacity_w: float,
) -> pd.DataFrame:
    out = frame.copy()
    origin_clear_sky = None
    target_clear_sky = None

    if "origin_clear_sky_power_proxy_w" in out.columns:
        origin_norm = _safe_clear_sky_normalizer(out["origin_clear_sky_power_proxy_w"])
        origin_clear_sky = origin_norm
        out["origin_power_clear_sky_ratio"] = out["power_w"] / origin_norm

    if "target_clear_sky_power_proxy_w" in out.columns:
        target_norm = _safe_clear_sky_normalizer(out["target_clear_sky_power_proxy_w"])
        target_clear_sky = target_norm
        if "baseline_previous_day_power_w" in out.columns:
            out["baseline_previous_day_clear_sky_ratio"] = out["baseline_previous_day_power_w"] / target_norm
        if "baseline_previous_week_power_w" in out.columns:
            out["baseline_previous_week_clear_sky_ratio"] = out["baseline_previous_week_power_w"] / target_norm
        if system_capacity_w > 0:
            out["target_clear_sky_capacity_ratio"] = out["target_clear_sky_power_proxy_w"] / system_capacity_w

    if "hist_shortwave_radiation" in out.columns and "origin_solar_cos_zenith" in out.columns:
        solar_norm = np.maximum(out["origin_solar_cos_zenith"].to_numpy(dtype=float), 0.05)
        out["origin_shortwave_cos_ratio"] = out["hist_shortwave_radiation"].to_numpy(dtype=float) / solar_norm

    if "baseline_issue_persistence_w" in out.columns:
        out["feature_baseline_issue_persistence_w"] = out["baseline_issue_persistence_w"].astype(float)
    if "baseline_previous_day_power_w" in out.columns:
        out["feature_baseline_previous_day_power_w"] = out["baseline_previous_day_power_w"].astype(float)
    if "baseline_previous_week_power_w" in out.columns:
        out["feature_baseline_previous_week_power_w"] = out["baseline_previous_week_power_w"].astype(float)

    if origin_clear_sky is not None and target_clear_sky is not None:
        clear_sky_progression_ratio = target_clear_sky / origin_clear_sky
        out["clear_sky_progression_ratio"] = clear_sky_progression_ratio
        if "baseline_issue_persistence_w" in out.columns:
            out["feature_persistence_clear_sky_scaled_w"] = (
                out["baseline_issue_persistence_w"].to_numpy(dtype=float) * clear_sky_progression_ratio
            )
        if "origin_panel_cos_incidence" in out.columns and "target_panel_cos_incidence" in out.columns:
            origin_panel_norm = np.maximum(out["origin_panel_cos_incidence"].to_numpy(dtype=float), 0.02)
            target_panel = out["target_panel_cos_incidence"].to_numpy(dtype=float)
            out["panel_incidence_progression_ratio"] = target_panel / origin_panel_norm

    if target_clear_sky is not None:
        if "baseline_previous_day_clear_sky_ratio" in out.columns:
            out["feature_previous_day_clear_sky_scaled_w"] = (
                out["baseline_previous_day_clear_sky_ratio"].to_numpy(dtype=float) * target_clear_sky
            )
        if "baseline_previous_week_clear_sky_ratio" in out.columns:
            out["feature_previous_week_clear_sky_scaled_w"] = (
                out["baseline_previous_week_clear_sky_ratio"].to_numpy(dtype=float) * target_clear_sky
            )

    if "lead_hours" in out.columns:
        lead_hours = out["lead_hours"].to_numpy(dtype=float)
        out["lead_bucket_code"] = _compute_lead_bucket_codes(lead_hours)
        out["lead_bucket_label"] = pd.Series(out["lead_bucket_code"]).map(LEAD_BUCKET_LABELS).fillna("unknown")
        out["is_short_lead"] = (lead_hours <= LEAD_BUCKET_BOUNDS_HOURS[0]).astype(int)
        out["is_medium_lead"] = (
            (lead_hours > LEAD_BUCKET_BOUNDS_HOURS[0]) & (lead_hours <= LEAD_BUCKET_BOUNDS_HOURS[2])
        ).astype(int)
        out["is_long_lead"] = (lead_hours > LEAD_BUCKET_BOUNDS_HOURS[2]).astype(int)

        persistence_weight = np.clip(1.0 - (lead_hours / 8.0), 0.15, 0.95)
        weekly_weight = np.clip((lead_hours - 2.0) / 14.0, 0.0, 0.30)
        out["intraday_persistence_weight"] = persistence_weight
        out["intraday_weekly_weight"] = weekly_weight

        persistence_scaled = out.get("feature_persistence_clear_sky_scaled_w", out.get("feature_baseline_issue_persistence_w"))
        prev_day_scaled = out.get("feature_previous_day_clear_sky_scaled_w", out.get("feature_baseline_previous_day_power_w"))
        prev_week_scaled = out.get(
            "feature_previous_week_clear_sky_scaled_w",
            out.get("feature_baseline_previous_week_power_w", prev_day_scaled),
        )
        if persistence_scaled is not None and prev_day_scaled is not None:
            persistence_scaled_values = np.asarray(persistence_scaled, dtype=float)
            prev_day_scaled_values = np.asarray(prev_day_scaled, dtype=float)
            prev_week_scaled_values = np.asarray(prev_week_scaled, dtype=float)
            day_blend = (
                persistence_weight * persistence_scaled_values
                + (1.0 - persistence_weight) * prev_day_scaled_values
            )
            out["feature_intraday_baseline_blend_w"] = (
                (1.0 - weekly_weight) * day_blend + weekly_weight * prev_week_scaled_values
            )

    if "target_fcst_panel_radiation_proxy" in out.columns and target_clear_sky is not None:
        out["feature_fcst_panel_clear_sky_ratio"] = (
            out["target_fcst_panel_radiation_proxy"].to_numpy(dtype=float) / target_clear_sky
        )
        if "feature_intraday_baseline_blend_w" in out.columns:
            forecast_factor = np.clip(out["feature_fcst_panel_clear_sky_ratio"].to_numpy(dtype=float), 0.0, 1.6)
            baseline_factor = np.clip(out["target_clear_sky_capacity_ratio"].to_numpy(dtype=float), 0.02, 1.0)
            out["feature_intraday_fcst_adjusted_blend_w"] = (
                out["feature_intraday_baseline_blend_w"].to_numpy(dtype=float)
                * np.where(baseline_factor > 0.0, forecast_factor / baseline_factor, 1.0)
            )

    return out


def add_timestamp_features(
    frame: pd.DataFrame,
    *,
    time_col: str,
    prefix: str,
    latitude: float,
    longitude: float,
    tilt_deg: float | None,
    azimuth_deg: float | None,
    system_capacity_w: float,
) -> pd.DataFrame:
    out = frame.copy()
    ts = out[time_col]
    out[f"{prefix}_hour"] = ts.dt.hour
    out[f"{prefix}_minute"] = ts.dt.minute
    out[f"{prefix}_day_of_week"] = ts.dt.dayofweek
    out[f"{prefix}_day_of_year"] = ts.dt.dayofyear
    out[f"{prefix}_month"] = ts.dt.month
    out[f"{prefix}_week_of_year"] = ts.dt.isocalendar().week.astype(int)
    out[f"{prefix}_is_weekend"] = (out[f"{prefix}_day_of_week"] >= 5).astype(int)

    out[f"{prefix}_sin_hour"] = np.sin(2.0 * math.pi * out[f"{prefix}_hour"] / 24.0)
    out[f"{prefix}_cos_hour"] = np.cos(2.0 * math.pi * out[f"{prefix}_hour"] / 24.0)
    out[f"{prefix}_sin_day_of_year"] = np.sin(
        2.0 * math.pi * out[f"{prefix}_day_of_year"] / 366.0
    )
    out[f"{prefix}_cos_day_of_year"] = np.cos(
        2.0 * math.pi * out[f"{prefix}_day_of_year"] / 366.0
    )

    season_idx = ((out[f"{prefix}_month"] % 12) // 3).astype(int)
    out[f"{prefix}_season_idx"] = season_idx
    out[f"{prefix}_season"] = season_idx.map(
        {
            0: "winter",
            1: "spring",
            2: "summer",
            3: "autumn",
        }
    )

    out = _add_solar_geometry_features(
        out,
        time_col=time_col,
        prefix=prefix,
        latitude=latitude,
        longitude=longitude,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        system_capacity_w=system_capacity_w,
    )
    return out


def _add_issue_history_features(
    df: pd.DataFrame,
    *,
    config: FeatureConfig,
    system_capacity_w: float,
) -> pd.DataFrame:
    out = df.copy()
    for lag in config.lags:
        out[f"lag_power_{lag}"] = out["power_w"].shift(lag)

    shifted = out["power_w"].shift(1)
    for window in config.rolling_windows:
        rolled = shifted.rolling(window=window, min_periods=1)
        out[f"roll_mean_power_{window}"] = rolled.mean()
        out[f"roll_max_power_{window}"] = rolled.max()
        out[f"roll_std_power_{window}"] = rolled.std()
        out[f"roll_sum_power_{window}"] = rolled.sum()

    for ramp_lag in (1, 4, 16):
        lag_col = f"lag_power_{ramp_lag}"
        if lag_col in out.columns:
            out[f"ramp_power_{ramp_lag}"] = out["power_w"] - out[lag_col]

    for weather_col in ("hist_cloud_cover", "hist_shortwave_radiation", "hist_temperature_2m"):
        if weather_col not in out.columns:
            continue
        shifted_weather = out[weather_col].shift(1)
        out[f"{weather_col}_roll_mean_4"] = shifted_weather.rolling(window=4, min_periods=1).mean()
        out[f"{weather_col}_delta_1"] = out[weather_col] - out[weather_col].shift(1)
        out[f"{weather_col}_delta_4"] = out[weather_col] - out[weather_col].shift(4)

    out["power_kw"] = out["power_w"] / 1000.0
    if system_capacity_w > 0:
        out["power_capacity_ratio"] = out["power_w"] / system_capacity_w
    out["is_daylight_proxy"] = (out["power_w"] > 5.0).astype(int)
    return out


def _add_solar_geometry_features(
    df: pd.DataFrame,
    *,
    time_col: str,
    prefix: str,
    latitude: float,
    longitude: float,
    tilt_deg: float | None,
    azimuth_deg: float | None,
    system_capacity_w: float,
) -> pd.DataFrame:
    out = df.copy()
    ts = out[time_col]
    day_of_year = ts.dt.dayofyear.to_numpy(dtype=float)
    hour = ts.dt.hour.to_numpy(dtype=float)
    minute = ts.dt.minute.to_numpy(dtype=float)

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
    true_solar_minutes = np.mod((hour * 60.0) + minute + time_offset, 1440.0)
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
    solar_azimuth_deg = (np.rad2deg(azimuth_rad) + 180.0) % 360.0

    out[f"{prefix}_solar_elevation_deg"] = elevation_deg
    out[f"{prefix}_solar_azimuth_deg"] = solar_azimuth_deg
    out[f"{prefix}_solar_cos_zenith"] = cos_zenith_clipped
    out[f"{prefix}_is_daylight_solar"] = (elevation_deg > 0.0).astype(int)

    panel_tilt = float(tilt_deg or 0.0)
    panel_azimuth = float(azimuth_deg or 180.0)
    elev_rad = np.deg2rad(np.clip(elevation_deg, -90.0, 90.0))
    solar_azimuth_rad = np.deg2rad(solar_azimuth_deg)
    panel_tilt_rad = np.deg2rad(panel_tilt)
    panel_azimuth_rad = np.deg2rad(panel_azimuth)

    cos_incidence = (
        np.sin(elev_rad) * np.cos(panel_tilt_rad)
        + np.cos(elev_rad)
        * np.sin(panel_tilt_rad)
        * np.cos(solar_azimuth_rad - panel_azimuth_rad)
    )
    cos_incidence = np.clip(cos_incidence, 0.0, 1.0)
    out[f"{prefix}_panel_cos_incidence"] = cos_incidence
    out[f"{prefix}_clear_sky_power_proxy_w"] = cos_incidence * max(system_capacity_w, 0.0)
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
        "issue_date",
        "target_date",
        "target_season",
        "origin_season",
        "baseline_previous_day_power_w",
        "baseline_previous_week_power_w",
        "baseline_issue_persistence_w",
        "forecast_valid_hour",
        "forecast_source_time",
        "is_missing_target_power",
        "fold_id",
        "fold_name",
    }
    meta_cols = [
        "origin_time",
        "target_time",
        "issue_date",
        "target_date",
        "target_season",
        "baseline_previous_day_power_w",
        "baseline_previous_week_power_w",
        "baseline_issue_persistence_w",
    ]
    if "forecast_available_at_origin" in work.columns:
        meta_cols.append("forecast_available_at_origin")
    if "lead_bucket_label" in work.columns:
        meta_cols.append("lead_bucket_label")

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


def _safe_clear_sky_normalizer(series: pd.Series) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    return np.maximum(values, MIN_CLEAR_SKY_PROXY_W)


def _compute_lead_bucket_codes(lead_hours: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(lead_hours, dtype=float), bins=np.asarray(LEAD_BUCKET_BOUNDS_HOURS), right=True)
