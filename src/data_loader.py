from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_FREQUENCY,
    FORECAST_HORIZON_STEPS,
    PANEL_AZIMUTH,
    PANEL_LAT,
    PANEL_LON,
    PANEL_TILT,
    SYSTEM_CAPACITY_W,
    TIMEZONE,
    ProjectPaths,
    default_project_paths,
)


@dataclass(frozen=True)
class SystemMetadata:
    latitude: float
    longitude: float
    tilt: float | None
    azimuth: float | None
    system_capacity_w: float | None
    timezone: str


def _read_csv(path: Path, *, parse_dates: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def load_system_metadata(paths: ProjectPaths | None = None) -> SystemMetadata:
    paths = paths or default_project_paths()
    payload = {}
    if paths.metadata_json.exists():
        payload = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    return SystemMetadata(
        latitude=float(payload.get("latitude", PANEL_LAT)),
        longitude=float(payload.get("longitude", PANEL_LON)),
        tilt=_coerce_optional_float(payload.get("tilt", PANEL_TILT)),
        azimuth=_coerce_optional_float(payload.get("azimuth", PANEL_AZIMUTH)),
        system_capacity_w=_coerce_optional_float(
            payload.get("installed_kwp", SYSTEM_CAPACITY_W / 1000.0),
            multiplier=1000.0,
        ),
        timezone=str(payload.get("timezone", TIMEZONE)),
    )


def _coerce_optional_float(value: object, multiplier: float = 1.0) -> float | None:
    if value is None:
        return None
    return float(value) * multiplier


def load_production_frame(
    paths: ProjectPaths | None = None,
    *,
    freq: str = DATA_FREQUENCY,
    complete_index: bool = True,
) -> pd.DataFrame:
    paths = paths or default_project_paths()
    df = _read_csv(paths.production_csv, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp")
    df["power_w"] = pd.to_numeric(df["power_w"], errors="raise")
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    if not complete_index:
        return df

    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=freq)
    full = pd.DataFrame({"timestamp": full_index})
    merged = full.merge(df, on="timestamp", how="left")
    merged["is_missing_production"] = merged["power_w"].isna()
    return merged


def load_weather_history_frame(
    paths: ProjectPaths | None = None,
    *,
    freq: str = DATA_FREQUENCY,
    prefer_15min: bool = True,
) -> pd.DataFrame:
    paths = paths or default_project_paths()
    candidates = []
    if prefer_15min and paths.weather_history_15min_csv.exists():
        candidates.append(("15min", paths.weather_history_15min_csv))
    if paths.weather_archive_hourly_csv.exists():
        candidates.append(("hourly", paths.weather_archive_hourly_csv))
    if not candidates:
        raise FileNotFoundError("No weather history file was found in the organized data directory.")

    source_kind, source_path = candidates[0]
    df = _read_csv(source_path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp")
    weather_columns = [column for column in df.columns if column != "timestamp"]

    if source_kind == "hourly":
        df = _upsample_hourly_weather_to_15min(df, freq=freq)
    else:
        df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    rename_map = {column: f"hist_{column}" for column in weather_columns}
    df = df.rename(columns=rename_map)
    df["weather_history_source"] = source_kind
    return df


def _upsample_hourly_weather_to_15min(df: pd.DataFrame, *, freq: str) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp").sort_index()
    upsampled = df.resample(freq).ffill()
    upsampled = upsampled.reset_index()
    return upsampled


def load_previous_runs_forecast_frame(paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or default_project_paths()
    if paths.weather_previous_runs_csv.exists():
        df = _read_csv(
            paths.weather_previous_runs_csv,
            parse_dates=["forecast_issue_time", "forecast_valid_time"],
        )
        return df.sort_values(["forecast_issue_time", "forecast_valid_time"]).reset_index(drop=True)

    chunks_dir = paths.weather_previous_runs_csv.parent / "chunks"
    chunk_paths = sorted(chunks_dir.glob("*_issue_valid.csv"))
    if not chunk_paths:
        raise FileNotFoundError(
            "No previous-runs master file or chunk files were found in the organized data directory."
        )

    frames = [
        pd.read_csv(chunk_path, parse_dates=["forecast_issue_time", "forecast_valid_time"])
        for chunk_path in chunk_paths
    ]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["forecast_issue_time", "forecast_valid_time", "source_previous_day"]
    )
    df = df.sort_values(["forecast_issue_time", "forecast_valid_time"]).reset_index(drop=True)
    return df


def load_previous_runs_wide_frame(paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or default_project_paths()
    if paths.weather_previous_runs_wide_csv.exists():
        df = _read_csv(paths.weather_previous_runs_wide_csv, parse_dates=["time"])
        return df.sort_values("time").reset_index(drop=True)

    chunks_dir = paths.weather_previous_runs_wide_csv.parent / "chunks"
    chunk_paths = sorted(chunks_dir.glob("*_wide.csv"))
    if not chunk_paths:
        raise FileNotFoundError(
            "No previous-runs wide master file or chunk files were found in the organized data directory."
        )

    frames = [pd.read_csv(chunk_path, parse_dates=["time"]) for chunk_path in chunk_paths]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"])
    return df.sort_values("time").reset_index(drop=True)


def load_noaa_gfs_forecast_frame(paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or default_project_paths()
    if not paths.weather_noaa_gfs_issue_valid_csv.exists():
        raise FileNotFoundError(f"NOAA GFS forecast file was not found: {paths.weather_noaa_gfs_issue_valid_csv}")
    df = _read_csv(
        paths.weather_noaa_gfs_issue_valid_csv,
        parse_dates=["forecast_issue_time", "forecast_valid_time"],
    )
    return df.sort_values(["forecast_issue_time", "forecast_valid_time"]).reset_index(drop=True)


def build_horizon_training_frame(
    horizon_steps: int = FORECAST_HORIZON_STEPS,
    *,
    paths: ProjectPaths | None = None,
    freq: str = DATA_FREQUENCY,
    include_history_weather: bool = True,
    include_leakage_safe_forecast: bool = True,
    drop_missing_target: bool = True,
    drop_missing_forecast: bool = False,
) -> pd.DataFrame:
    paths = paths or default_project_paths()
    production = load_production_frame(paths, freq=freq, complete_index=True)
    frame = production.rename(columns={"timestamp": "origin_time"}).copy()

    horizon_delta = pd.to_timedelta(horizon_steps * 15, unit="minute")
    horizon_hours = horizon_steps / 4.0
    required_previous_day = max(1, math.ceil(horizon_hours / 24.0))

    frame["horizon_steps"] = horizon_steps
    frame["horizon_hours"] = horizon_hours
    frame["required_previous_day"] = required_previous_day
    frame["target_time"] = frame["origin_time"] + horizon_delta

    target_lookup = production.rename(
        columns={
            "timestamp": "target_time",
            "power_w": "target_power_w",
            "is_missing_production": "is_missing_target_power",
        }
    )
    frame = frame.merge(
        target_lookup.loc[:, ["target_time", "target_power_w", "is_missing_target_power"]],
        on="target_time",
        how="left",
    )

    if include_history_weather:
        history_weather = load_weather_history_frame(paths, freq=freq)
        frame = frame.merge(
            history_weather,
            left_on="origin_time",
            right_on="timestamp",
            how="left",
        ).drop(columns=["timestamp"])

    if include_leakage_safe_forecast:
        previous_runs = load_previous_runs_forecast_frame(paths)
        previous_runs = previous_runs.loc[
            previous_runs["source_previous_day"] == required_previous_day
        ].copy()
        previous_runs["forecast_valid_hour"] = previous_runs["forecast_valid_time"]

        base_columns = [
            column
            for column in previous_runs.columns
            if column
            not in {
                "forecast_issue_time",
                "forecast_valid_time",
                "forecast_valid_hour",
                "lead_hours",
                "source_previous_day",
            }
        ]
        rename_map = {
            column: f"fcst_day{required_previous_day}_{column}" for column in base_columns
        }
        previous_runs = previous_runs.rename(columns=rename_map)

        frame["forecast_valid_hour"] = frame["target_time"].dt.floor("h")
        frame = frame.merge(
            previous_runs,
            on="forecast_valid_hour",
            how="left",
        )
        frame["forecast_available_at_origin"] = (
            frame["forecast_issue_time"].notna() & (frame["forecast_issue_time"] <= frame["origin_time"])
        )
        frame["forecast_issue_age_hours_at_origin"] = (
            frame["origin_time"] - frame["forecast_issue_time"]
        ).dt.total_seconds() / 3600.0
    else:
        frame["forecast_available_at_origin"] = False

    if drop_missing_target:
        frame = frame.loc[frame["target_power_w"].notna()].copy()

    if include_leakage_safe_forecast:
        frame = frame.loc[
            frame["forecast_available_at_origin"] | frame["forecast_issue_time"].isna()
        ].copy()
        if drop_missing_forecast:
            frame = frame.loc[frame["forecast_available_at_origin"]].copy()

    return frame.reset_index(drop=True)


def build_multi_horizon_training_frames(
    horizon_steps_list: list[int],
    *,
    paths: ProjectPaths | None = None,
    freq: str = DATA_FREQUENCY,
    include_history_weather: bool = True,
    include_leakage_safe_forecast: bool = True,
    drop_missing_target: bool = True,
    drop_missing_forecast: bool = False,
) -> dict[int, pd.DataFrame]:
    return {
        horizon_steps: build_horizon_training_frame(
            horizon_steps,
            paths=paths,
            freq=freq,
            include_history_weather=include_history_weather,
            include_leakage_safe_forecast=include_leakage_safe_forecast,
            drop_missing_target=drop_missing_target,
            drop_missing_forecast=drop_missing_forecast,
        )
        for horizon_steps in horizon_steps_list
    }


def summarize_training_frame(frame: pd.DataFrame) -> dict[str, object]:
    summary = {
        "rows": int(len(frame)),
        "origin_start": _safe_timestamp(frame, "origin_time", "min"),
        "origin_end": _safe_timestamp(frame, "origin_time", "max"),
        "target_start": _safe_timestamp(frame, "target_time", "min"),
        "target_end": _safe_timestamp(frame, "target_time", "max"),
        "missing_target_power_rows": int(frame["target_power_w"].isna().sum())
        if "target_power_w" in frame.columns
        else 0,
        "forecast_available_rows": int(frame["forecast_available_at_origin"].fillna(False).sum())
        if "forecast_available_at_origin" in frame.columns
        else 0,
    }
    return summary


def _safe_timestamp(frame: pd.DataFrame, column: str, op: str) -> str | None:
    if column not in frame.columns or frame.empty:
        return None
    series = frame[column]
    value = getattr(series, op)()
    return None if pd.isna(value) else str(value)
