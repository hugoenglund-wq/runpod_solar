from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import SYSTEM_CAPACITY_W
from src.data_loader import (
    ProjectPaths,
    SystemMetadata,
    default_project_paths,
    load_noaa_gfs_forecast_frame,
    load_previous_runs_wide_frame,
    load_production_frame,
    load_system_metadata,
    load_weather_history_frame,
)
from src.day_features import (
    DEFAULT_FEATURE_CONFIG,
    add_relative_physics_features,
    add_timestamp_features,
    build_issue_feature_frame,
)


DAILY_ISSUE_HOUR = 23
DAILY_ISSUE_MINUTE = 45
FORECAST_BASE_VARS = ("temperature_2m", "cloud_cover", "shortwave_radiation")


@dataclass(frozen=True)
class DayModelSpec:
    name: str
    day_offset: int
    use_forecast_archive: bool
    issue_schedule: str


DEFAULT_DAY_MODEL_SPECS = (
    DayModelSpec(
        name="day_0",
        day_offset=0,
        use_forecast_archive=False,
        issue_schedule="rolling_15min",
    ),
    DayModelSpec(
        name="day_1",
        day_offset=1,
        use_forecast_archive=True,
        issue_schedule="daily_23_45",
    ),
    DayModelSpec(
        name="day_2",
        day_offset=2,
        use_forecast_archive=True,
        issue_schedule="daily_23_45",
    ),
    DayModelSpec(
        name="day_3",
        day_offset=3,
        use_forecast_archive=True,
        issue_schedule="daily_23_45",
    ),
)


def build_day_model_frame(
    spec: DayModelSpec,
    *,
    paths: ProjectPaths | None = None,
    metadata: SystemMetadata | None = None,
    feature_config=DEFAULT_FEATURE_CONFIG,
    daylight_only_targets: bool = True,
) -> pd.DataFrame:
    paths = paths or default_project_paths()
    metadata = metadata or load_system_metadata(paths)
    has_noaa_gfs = paths.weather_noaa_gfs_issue_valid_csv.exists()
    use_noaa_gfs = bool(spec.use_forecast_archive and has_noaa_gfs)
    use_intraday_forecast = bool(spec.day_offset == 0 and has_noaa_gfs)

    production = load_production_frame(paths, complete_index=True)
    weather_history = load_weather_history_frame(paths)

    issue_frame = production.rename(
        columns={
            "timestamp": "origin_time",
            "is_missing_production": "is_missing_origin_power",
        }
    ).copy()
    issue_frame["issue_date"] = issue_frame["origin_time"].dt.normalize()
    issue_frame = issue_frame.merge(
        weather_history,
        left_on="origin_time",
        right_on="timestamp",
        how="left",
    ).drop(columns=["timestamp"])

    if spec.issue_schedule == "daily_23_45":
        issue_frame = issue_frame.loc[
            (issue_frame["origin_time"].dt.hour == DAILY_ISSUE_HOUR)
            & (issue_frame["origin_time"].dt.minute == DAILY_ISSUE_MINUTE)
        ].copy()

    issue_frame = build_issue_feature_frame(
        issue_frame,
        config=feature_config,
        latitude=metadata.latitude,
        longitude=metadata.longitude,
        tilt_deg=metadata.tilt,
        azimuth_deg=metadata.azimuth,
        system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
    )

    target_lookup = production.rename(
        columns={
            "timestamp": "target_time",
            "power_w": "target_power_w",
            "is_missing_production": "is_missing_target_power",
        }
    ).copy()
    target_lookup["target_date"] = target_lookup["target_time"].dt.normalize()
    target_lookup = add_timestamp_features(
        target_lookup,
        time_col="target_time",
        prefix="target",
        latitude=metadata.latitude,
        longitude=metadata.longitude,
        tilt_deg=metadata.tilt,
        azimuth_deg=metadata.azimuth,
        system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
    )
    if daylight_only_targets:
        target_lookup = target_lookup.loc[target_lookup["target_is_daylight_solar"] == 1].copy()

    expanded = _expand_issue_target_pairs(issue_frame, target_lookup, spec)
    if expanded.empty:
        return expanded
    for redundant_col in ("is_missing_production_x", "is_missing_production_y"):
        if redundant_col in expanded.columns:
            expanded = expanded.drop(columns=[redundant_col])

    expanded["lead_minutes"] = (expanded["target_time"] - expanded["origin_time"]).dt.total_seconds() / 60.0
    expanded["lead_hours"] = expanded["lead_minutes"] / 60.0
    expanded["lead_steps"] = expanded["lead_minutes"] / 15.0
    expanded["target_day_offset"] = spec.day_offset
    expanded["remaining_day_minutes_at_issue"] = (
        (expanded["origin_time"].dt.normalize() + pd.Timedelta(days=1)) - expanded["origin_time"]
    ).dt.total_seconds() / 60.0

    if spec.use_forecast_archive or use_intraday_forecast:
        expanded = _merge_target_forecast_features(
            expanded,
            paths=paths,
            day_offset=spec.day_offset,
            use_noaa_gfs=has_noaa_gfs,
        )
        if expanded.empty:
            return expanded
        expanded = _add_daily_forecast_aggregates(expanded)

    expanded = _add_baselines(expanded, production)
    expanded = add_relative_physics_features(
        expanded,
        system_capacity_w=metadata.system_capacity_w or SYSTEM_CAPACITY_W,
    )
    expanded["model_name"] = spec.name
    expanded["issue_schedule"] = spec.issue_schedule
    return expanded.reset_index(drop=True)


def _expand_issue_target_pairs(
    issue_frame: pd.DataFrame,
    target_lookup: pd.DataFrame,
    spec: DayModelSpec,
) -> pd.DataFrame:
    issue_groups = {
        key: frame.reset_index(drop=True)
        for key, frame in issue_frame.groupby("issue_date", sort=True)
    }
    target_groups = {
        key: frame.reset_index(drop=True)
        for key, frame in target_lookup.groupby("target_date", sort=True)
    }

    day_frames: list[pd.DataFrame] = []
    for issue_date, issue_day in issue_groups.items():
        target_date = issue_date + pd.Timedelta(days=spec.day_offset)
        target_day = target_groups.get(target_date)
        if target_day is None or target_day.empty:
            continue

        merged = (
            issue_day.assign(_merge_key=1)
            .merge(target_day.assign(_merge_key=1), on="_merge_key", how="inner")
            .drop(columns=["_merge_key"])
        )
        if spec.day_offset == 0:
            merged = merged.loc[merged["target_time"] > merged["origin_time"]].copy()
        if merged.empty:
            continue
        day_frames.append(merged)

    if not day_frames:
        return pd.DataFrame()
    return pd.concat(day_frames, ignore_index=True)


def _merge_target_forecast_features(
    frame: pd.DataFrame,
    *,
    paths: ProjectPaths,
    day_offset: int,
    use_noaa_gfs: bool,
) -> pd.DataFrame:
    if use_noaa_gfs:
        return _merge_noaa_gfs_target_forecast_features(frame, paths=paths)

    wide = load_previous_runs_wide_frame(paths)
    wide["forecast_valid_hour"] = pd.to_datetime(wide["time"], errors="raise")

    selected_cols = ["forecast_valid_hour"]
    rename_map: dict[str, str] = {}
    for base_var in FORECAST_BASE_VARS:
        source_col = f"{base_var}_previous_day{day_offset}"
        if source_col in wide.columns:
            selected_cols.append(source_col)
            rename_map[source_col] = f"target_fcst_{base_var}"

    if len(selected_cols) == 1:
        return pd.DataFrame()

    forecast = wide.loc[:, selected_cols].rename(columns=rename_map)

    out = frame.copy()
    out["forecast_valid_hour"] = out["target_time"].dt.floor("h")
    out["forecast_source_time"] = out["forecast_valid_hour"] - pd.to_timedelta(day_offset, unit="D")
    out = out.merge(forecast, on="forecast_valid_hour", how="left")
    out["forecast_available_at_origin"] = out["forecast_source_time"] <= out["origin_time"]
    out["forecast_issue_age_hours_at_origin"] = (
        out["origin_time"] - out["forecast_source_time"]
    ).dt.total_seconds() / 3600.0

    required_cols = [col for col in out.columns if col.startswith("target_fcst_")]
    out = out.loc[out["forecast_available_at_origin"]].copy()
    out = out.dropna(axis="index", how="any", subset=required_cols)
    if "target_fcst_shortwave_radiation" in out.columns:
        out["target_fcst_panel_radiation_proxy"] = (
            out["target_fcst_shortwave_radiation"] * out["target_panel_cos_incidence"]
        )
    return out


def _merge_noaa_gfs_target_forecast_features(
    frame: pd.DataFrame,
    *,
    paths: ProjectPaths,
) -> pd.DataFrame:
    noaa = load_noaa_gfs_forecast_frame(paths).copy()
    rename_map = {
        "temperature_2m": "target_fcst_temperature_2m",
        "cloud_cover": "target_fcst_cloud_cover",
        "shortwave_radiation": "target_fcst_shortwave_radiation",
        "lead_hours": "target_fcst_lead_hours",
    }
    noaa = noaa.rename(columns=rename_map)
    noaa["forecast_valid_hour"] = noaa["forecast_valid_time"].dt.floor("h")

    selected_cols = [
        "forecast_issue_time",
        "forecast_valid_hour",
        "forecast_valid_time",
        *rename_map.values(),
    ]
    out = frame.copy()
    out["forecast_valid_hour"] = out["target_time"].dt.floor("h")
    left = out.sort_values(["forecast_valid_hour", "origin_time"]).reset_index(drop=True)
    right = noaa.loc[:, selected_cols].sort_values(["forecast_valid_hour", "forecast_issue_time"]).reset_index(
        drop=True
    )
    merged_parts: list[pd.DataFrame] = []
    right_groups = {
        forecast_valid_hour: part.sort_values("forecast_issue_time").reset_index(drop=True)
        for forecast_valid_hour, part in right.groupby("forecast_valid_hour", sort=False)
    }
    for forecast_valid_hour, left_part in left.groupby("forecast_valid_hour", sort=False):
        right_part = right_groups.get(forecast_valid_hour)
        if right_part is None or right_part.empty:
            left_empty = left_part.copy()
            for col in selected_cols:
                if col not in left_empty.columns:
                    if col in {"forecast_issue_time", "forecast_valid_time"}:
                        left_empty[col] = pd.NaT
                    else:
                        left_empty[col] = pd.NA
            merged_parts.append(left_empty)
            continue

        merged_parts.append(
            pd.merge_asof(
                left_part.sort_values("origin_time").reset_index(drop=True),
                right_part,
                left_on="origin_time",
                right_on="forecast_issue_time",
                direction="backward",
                allow_exact_matches=True,
            )
        )

    if not merged_parts:
        return pd.DataFrame()
    out = pd.concat(merged_parts, ignore_index=True)
    out["forecast_issue_time"] = pd.to_datetime(out["forecast_issue_time"], errors="coerce")
    out["forecast_valid_time"] = pd.to_datetime(out["forecast_valid_time"], errors="coerce")
    out["forecast_available_at_origin"] = out["forecast_valid_time"].notna()
    out["forecast_issue_age_hours_at_origin"] = (
        out["origin_time"] - out["forecast_issue_time"]
    ).dt.total_seconds() / 3600.0
    required_cols = [col for col in rename_map.values() if col in out.columns]
    out = out.loc[out["forecast_available_at_origin"]].copy()
    out = out.dropna(axis="index", how="any", subset=required_cols)
    if "target_fcst_shortwave_radiation" in out.columns:
        out["target_fcst_panel_radiation_proxy"] = (
            out["target_fcst_shortwave_radiation"] * out["target_panel_cos_incidence"]
        )
    return out


def _add_daily_forecast_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    group_cols = ["origin_time", "issue_date", "target_date"]
    aggregations: dict[str, tuple[str, str]] = {}
    if "target_fcst_temperature_2m" in out.columns:
        aggregations["day_fcst_temp_mean"] = ("target_fcst_temperature_2m", "mean")
        aggregations["day_fcst_temp_min"] = ("target_fcst_temperature_2m", "min")
        aggregations["day_fcst_temp_max"] = ("target_fcst_temperature_2m", "max")
    if "target_fcst_cloud_cover" in out.columns:
        aggregations["day_fcst_cloud_mean"] = ("target_fcst_cloud_cover", "mean")
        aggregations["day_fcst_cloud_max"] = ("target_fcst_cloud_cover", "max")
    if "target_fcst_shortwave_radiation" in out.columns:
        aggregations["day_fcst_shortwave_sum"] = ("target_fcst_shortwave_radiation", "sum")
        aggregations["day_fcst_shortwave_mean"] = ("target_fcst_shortwave_radiation", "mean")
        aggregations["day_fcst_shortwave_max"] = ("target_fcst_shortwave_radiation", "max")
    if "target_fcst_panel_radiation_proxy" in out.columns:
        aggregations["day_fcst_panel_proxy_sum"] = ("target_fcst_panel_radiation_proxy", "sum")
        aggregations["day_fcst_panel_proxy_max"] = ("target_fcst_panel_radiation_proxy", "max")

    if not aggregations:
        return out

    grouped = out.groupby(group_cols, as_index=False).agg(**aggregations)
    return out.merge(grouped, on=group_cols, how="left")


def _add_baselines(frame: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    lookup = production.rename(columns={"timestamp": "baseline_time", "power_w": "baseline_power_w"}).copy()

    prev_day_lookup = lookup.rename(columns={"baseline_power_w": "baseline_previous_day_power_w"})
    out["baseline_previous_day_time"] = out["target_time"] - pd.Timedelta(days=1)
    out = out.merge(
        prev_day_lookup,
        left_on="baseline_previous_day_time",
        right_on="baseline_time",
        how="left",
    ).drop(columns=["baseline_time", "baseline_previous_day_time"])

    prev_week_lookup = lookup.rename(columns={"baseline_power_w": "baseline_previous_week_power_w"})
    out["baseline_previous_week_time"] = out["target_time"] - pd.Timedelta(days=7)
    out = out.merge(
        prev_week_lookup,
        left_on="baseline_previous_week_time",
        right_on="baseline_time",
        how="left",
    ).drop(columns=["baseline_time", "baseline_previous_week_time"])

    out["baseline_issue_persistence_w"] = out["power_w"]
    out["baseline_previous_day_power_w"] = out["baseline_previous_day_power_w"].fillna(
        out["baseline_issue_persistence_w"]
    )
    out["baseline_previous_week_power_w"] = out["baseline_previous_week_power_w"].fillna(
        out["baseline_previous_day_power_w"]
    )
    return out
