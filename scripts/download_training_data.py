#!/usr/bin/env python3
"""
Download and organize the training data needed for home solar forecasting.

The script builds a local data lake with three layers:

1. Production history copied from the existing local CSV
2. Seamless 15-minute weather history from Open-Meteo Historical Forecast API
3. Leakage-aware hourly forecast archive from Open-Meteo Previous Runs API

Important:
- The seamless weather history is useful for feature engineering, but it does not
  contain forecast issue times. Do not treat it as a leakage-safe forecast archive.
- The previous-runs archive is leakage-aware, but Open-Meteo's public coverage for
  most variables begins in January 2024 according to their docs.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PANEL_AZIMUTH,
    PANEL_LAT,
    PANEL_LON,
    PANEL_TILT,
    SYSTEM_CAPACITY_W,
    TIMEZONE,
)


ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_DOCS = {
    "archive": "https://open-meteo.com/en/docs/historical-weather-api",
    "historical_forecast": "https://open-meteo.com/en/docs/historical-forecast-api",
    "previous_runs": "https://open-meteo.com/en/docs/previous-runs-api",
}

DEFAULT_ARCHIVE_HOURLY_VARS = [
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
]

DEFAULT_15MIN_WEATHER_VARS = [
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
]

DEFAULT_PREVIOUS_RUN_BASE_VARS = [
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
]


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date

    @property
    def label(self) -> str:
        return f"{self.start.isoformat()}_{self.end.isoformat()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and organize solar production and weather training data."
    )
    parser.add_argument(
        "--production-csv",
        default="solar_dataset_2021_2024.csv",
        help="Path to the local solar production CSV.",
    )
    parser.add_argument("--latitude", type=float, default=PANEL_LAT, help="Home latitude.")
    parser.add_argument("--longitude", type=float, default=PANEL_LON, help="Home longitude.")
    parser.add_argument(
        "--timezone",
        default=TIMEZONE,
        help="IANA timezone used for both production and weather data.",
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date in YYYY-MM-DD. Defaults to production data start.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date in YYYY-MM-DD. Defaults to production data end.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Root folder where all organized data will be written.",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=PANEL_TILT,
        help="Optional panel tilt in degrees. Stored as metadata only in this script.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=PANEL_AZIMUTH,
        help="Optional panel azimuth in degrees. Stored as metadata only in this script.",
    )
    parser.add_argument(
        "--installed-kwp",
        type=float,
        default=SYSTEM_CAPACITY_W / 1000.0,
        help="Optional installed solar capacity in kWp. Stored as metadata only.",
    )
    parser.add_argument(
        "--seamless-chunk-days",
        type=int,
        default=30,
        help="Chunk size for historical forecast backfill.",
    )
    parser.add_argument(
        "--previous-runs-chunk-days",
        type=int,
        default=1,
        help="Chunk size for previous-runs backfill. Keep this small because the API is slower.",
    )
    parser.add_argument(
        "--previous-runs-start-date",
        default="2024-01-01",
        help="Earliest date to request from the previous-runs archive.",
    )
    parser.add_argument(
        "--previous-run-max-day",
        type=int,
        default=5,
        help="Maximum previous_dayN columns to request from Open-Meteo Previous Runs API.",
    )
    parser.add_argument(
        "--weather-15min-vars",
        default=",".join(DEFAULT_15MIN_WEATHER_VARS),
        help="Comma-separated minutely_15 variables for historical forecast backfill.",
    )
    parser.add_argument(
        "--archive-hourly-vars",
        default=",".join(DEFAULT_ARCHIVE_HOURLY_VARS),
        help="Comma-separated hourly variables for stable historical weather backfill.",
    )
    parser.add_argument(
        "--forecast-base-vars",
        default=",".join(DEFAULT_PREVIOUS_RUN_BASE_VARS),
        help="Comma-separated base hourly variables for previous-runs backfill.",
    )
    parser.add_argument(
        "--skip-seamless-history",
        action="store_true",
        help="Skip the 15-minute historical-forecast weather backfill.",
    )
    parser.add_argument(
        "--skip-forecast-archive",
        action="store_true",
        help="Skip the previous-runs forecast archive backfill.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download chunk files even if they already exist.",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def chunk_dates(start: date, end: date, chunk_days: int) -> Iterable[DateWindow]:
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield DateWindow(start=cursor, end=chunk_end)
        cursor = chunk_end + timedelta(days=1)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, object],
    *,
    timeout: int = 120,
    retries: int = 4,
    backoff_seconds: float = 2.0,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(payload["reason"])
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries:
                break
            sleep_seconds = backoff_seconds * attempt
            print(
                f"[retry {attempt}/{retries}] {url} failed with {exc}. Sleeping {sleep_seconds:.1f}s...",
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed request for {url}: {last_error}") from last_error


def read_and_normalize_production_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    expected = {"datetime", "power_w"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required production columns: {sorted(missing)}")
    df = df.loc[:, ["datetime", "power_w"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    df["power_w"] = pd.to_numeric(df["power_w"], errors="raise")
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_json(payload: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def flatten_time_block(payload: dict, block_name: str) -> pd.DataFrame:
    if block_name not in payload:
        raise RuntimeError(f"Response did not contain expected block '{block_name}'")
    block = payload[block_name]
    if "time" not in block:
        raise RuntimeError(f"Block '{block_name}' does not contain a time column")
    return pd.DataFrame(block)


def download_historical_forecast_15min(
    session: requests.Session,
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    start: date,
    end: date,
    variables: list[str],
    chunk_days: int,
    output_dir: Path,
    force: bool,
) -> dict:
    chunks_dir = ensure_dir(output_dir / "raw" / "weather" / "historical_forecast_15min" / "chunks")
    frames: list[pd.DataFrame] = []
    chunk_files: list[str] = []
    for window in chunk_dates(start, end, chunk_days):
        chunk_path = chunks_dir / f"{window.label}.csv"
        chunk_files.append(str(chunk_path))
        if chunk_path.exists() and not force:
            frame = pd.read_csv(chunk_path, parse_dates=["datetime"])
            frames.append(frame)
            print(f"[skip] historical forecast chunk exists: {chunk_path.name}", flush=True)
            continue

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "start_date": window.start.isoformat(),
            "end_date": window.end.isoformat(),
            "minutely_15": ",".join(variables),
        }
        print(
            f"[download] historical forecast 15-min {window.start.isoformat()} -> {window.end.isoformat()}",
            flush=True,
        )
        payload = fetch_json(session, HISTORICAL_FORECAST_URL, params)
        frame = flatten_time_block(payload, "minutely_15").rename(columns={"time": "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        write_csv(frame, chunk_path)
        frames.append(frame)

    if not frames:
        raise RuntimeError("No historical forecast weather data was downloaded.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    master_path = output_dir / "raw" / "weather" / "historical_forecast_15min" / "weather_15min_seamless.csv"
    write_csv(combined, master_path)
    return {
        "source": "Open-Meteo Historical Forecast API",
        "docs": OPEN_METEO_DOCS["historical_forecast"],
        "coverage_start": str(combined["datetime"].min()),
        "coverage_end": str(combined["datetime"].max()),
        "rows": int(len(combined)),
        "variables": variables,
        "master_file": str(master_path),
        "chunk_files": chunk_files,
        "warning": (
            "This file is forecast-derived seamless history. It does not expose forecast_issue_time "
            "and must not be treated as a leakage-safe forecast archive."
        ),
    }


def download_archive_hourly_weather(
    session: requests.Session,
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    start: date,
    end: date,
    variables: list[str],
    chunk_days: int,
    output_dir: Path,
    force: bool,
) -> dict:
    chunks_dir = ensure_dir(output_dir / "raw" / "weather" / "archive_hourly" / "chunks")
    frames: list[pd.DataFrame] = []
    chunk_files: list[str] = []
    for window in chunk_dates(start, end, chunk_days):
        chunk_path = chunks_dir / f"{window.label}.csv"
        chunk_files.append(str(chunk_path))
        if chunk_path.exists() and not force:
            frame = pd.read_csv(chunk_path, parse_dates=["datetime"])
            frames.append(frame)
            print(f"[skip] archive weather chunk exists: {chunk_path.name}", flush=True)
            continue

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "start_date": window.start.isoformat(),
            "end_date": window.end.isoformat(),
            "hourly": ",".join(variables),
        }
        print(
            f"[download] archive hourly weather {window.start.isoformat()} -> {window.end.isoformat()}",
            flush=True,
        )
        payload = fetch_json(session, ARCHIVE_API_URL, params)
        frame = flatten_time_block(payload, "hourly").rename(columns={"time": "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        write_csv(frame, chunk_path)
        frames.append(frame)

    if not frames:
        raise RuntimeError("No archive weather data was downloaded.")

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    master_path = output_dir / "raw" / "weather" / "archive_hourly" / "weather_hourly_archive.csv"
    write_csv(combined, master_path)
    return {
        "source": "Open-Meteo Archive API",
        "docs": OPEN_METEO_DOCS["archive"],
        "coverage_start": str(combined["datetime"].min()),
        "coverage_end": str(combined["datetime"].max()),
        "rows": int(len(combined)),
        "variables": variables,
        "master_file": str(master_path),
        "chunk_files": chunk_files,
        "note": "Stable historical weather layer for backfills and QA.",
    }


def build_previous_run_hourly_variables(base_vars: list[str], max_previous_day: int) -> list[str]:
    variables: list[str] = []
    for base_var in base_vars:
        variables.append(base_var)
        for day in range(1, max_previous_day + 1):
            variables.append(f"{base_var}_previous_day{day}")
    return variables


def transform_previous_runs_to_issue_valid(
    wide_df: pd.DataFrame,
    base_vars: list[str],
    max_previous_day: int,
) -> pd.DataFrame:
    wide_df = wide_df.copy()
    wide_df["forecast_valid_time"] = pd.to_datetime(wide_df["time"], errors="raise")
    frames: list[pd.DataFrame] = []
    for day in range(0, max_previous_day + 1):
        source_columns = []
        rename_map = {}
        for base_var in base_vars:
            source_column = base_var if day == 0 else f"{base_var}_previous_day{day}"
            if source_column in wide_df.columns:
                source_columns.append(source_column)
                rename_map[source_column] = base_var
        if not source_columns:
            continue
        frame = wide_df.loc[:, ["forecast_valid_time", *source_columns]].rename(columns=rename_map)
        frame["forecast_issue_time"] = frame["forecast_valid_time"] - pd.to_timedelta(day, unit="D")
        frame["lead_hours"] = day * 24
        frame["source_previous_day"] = day
        weather_cols = [col for col in frame.columns if col not in {"forecast_valid_time", "forecast_issue_time", "lead_hours", "source_previous_day"}]
        frame = frame.dropna(axis="index", how="all", subset=weather_cols)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "forecast_issue_time",
                "forecast_valid_time",
                "lead_hours",
                "source_previous_day",
                *base_vars,
            ]
        )
    tidy = pd.concat(frames, ignore_index=True)
    tidy = tidy.sort_values(["forecast_issue_time", "forecast_valid_time"]).reset_index(drop=True)
    return tidy


def _download_previous_runs_window_by_variable(
    session: requests.Session,
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    window: DateWindow,
    base_vars: list[str],
    max_previous_day: int,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for base_var in base_vars:
        variable_group = [base_var, *[f"{base_var}_previous_day{day}" for day in range(1, max_previous_day + 1)]]
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "start_date": window.start.isoformat(),
            "end_date": window.end.isoformat(),
            "hourly": ",".join(variable_group),
        }
        payload = fetch_json(session, PREVIOUS_RUNS_URL, params, timeout=300, retries=6)
        frame = flatten_time_block(payload, "hourly")
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="time", how="outer")
    if merged is None:
        raise RuntimeError(f"No previous-runs data returned for {window.label}")
    return merged


def download_previous_runs_archive(
    session: requests.Session,
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    start: date,
    end: date,
    base_vars: list[str],
    max_previous_day: int,
    chunk_days: int,
    output_dir: Path,
    force: bool,
) -> dict:
    chunks_dir = ensure_dir(output_dir / "raw" / "weather" / "previous_runs_hourly" / "chunks")
    raw_frames: list[pd.DataFrame] = []
    tidy_frames: list[pd.DataFrame] = []
    chunk_files: list[str] = []
    variables = build_previous_run_hourly_variables(base_vars, max_previous_day)
    failed_windows: list[str] = []

    for window in chunk_dates(start, end, chunk_days):
        raw_chunk_path = chunks_dir / f"{window.label}_wide.csv"
        tidy_chunk_path = chunks_dir / f"{window.label}_issue_valid.csv"
        chunk_files.extend([str(raw_chunk_path), str(tidy_chunk_path)])

        if raw_chunk_path.exists() and tidy_chunk_path.exists() and not force:
            raw_frames.append(pd.read_csv(raw_chunk_path))
            tidy_frames.append(
                pd.read_csv(
                    tidy_chunk_path,
                    parse_dates=["forecast_issue_time", "forecast_valid_time"],
                )
            )
            print(f"[skip] previous runs chunk exists: {window.label}", flush=True)
            continue

        print(
            f"[download] previous runs hourly {window.start.isoformat()} -> {window.end.isoformat()}",
            flush=True,
        )
        try:
            raw_frame = _download_previous_runs_window_by_variable(
                session,
                latitude=latitude,
                longitude=longitude,
                timezone=timezone,
                window=window,
                base_vars=base_vars,
                max_previous_day=max_previous_day,
            )
            tidy_frame = transform_previous_runs_to_issue_valid(raw_frame, base_vars, max_previous_day)
            write_csv(raw_frame, raw_chunk_path)
            write_csv(tidy_frame, tidy_chunk_path)
            raw_frames.append(raw_frame)
            tidy_frames.append(tidy_frame)
        except RuntimeError as exc:
            failed_windows.append(window.label)
            print(f"[warning] previous-runs window failed: {window.label}: {exc}", flush=True)
            continue

    if not raw_frames or not tidy_frames:
        raise RuntimeError("Previous-runs backfill did not produce any successful chunks.")

    raw_combined = (
        pd.concat(raw_frames, ignore_index=True)
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    tidy_combined = (
        pd.concat(tidy_frames, ignore_index=True)
        .drop_duplicates(subset=["forecast_issue_time", "forecast_valid_time", "source_previous_day"])
        .sort_values(["forecast_issue_time", "forecast_valid_time"])
        .reset_index(drop=True)
    )
    raw_master_path = output_dir / "raw" / "weather" / "previous_runs_hourly" / "weather_previous_runs_wide.csv"
    tidy_master_path = (
        output_dir / "raw" / "weather" / "previous_runs_hourly" / "weather_previous_runs_issue_valid.csv"
    )
    write_csv(raw_combined, raw_master_path)
    write_csv(tidy_combined, tidy_master_path)
    return {
        "source": "Open-Meteo Previous Runs API",
        "docs": OPEN_METEO_DOCS["previous_runs"],
        "coverage_start": str(start),
        "coverage_end": str(end),
        "raw_rows": int(len(raw_combined)),
        "issue_valid_rows": int(len(tidy_combined)),
        "variables": variables,
        "base_variables": base_vars,
        "max_previous_day": max_previous_day,
        "raw_master_file": str(raw_master_path),
        "issue_valid_master_file": str(tidy_master_path),
        "chunk_files": chunk_files,
        "failed_windows": failed_windows,
        "warning": (
            "Open-Meteo documents that most previous-runs coverage begins in January 2024. "
            "This dataset is leakage-aware, but coverage before that date may be missing."
        ),
    }


def build_manifest(
    *,
    args: argparse.Namespace,
    production_summary: dict,
    archive_summary: dict,
    seamless_summary: dict | None,
    previous_runs_summary: dict | None,
    output_dir: Path,
) -> dict:
    manifest = {
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "project": "solar_google_collab",
        "data_root": str(output_dir.resolve()),
        "metadata": {
            "latitude": args.latitude,
            "longitude": args.longitude,
            "timezone": args.timezone,
            "tilt": args.tilt,
            "azimuth": args.azimuth,
            "installed_kwp": args.installed_kwp,
        },
        "production": production_summary,
        "weather": {
            "archive_hourly": archive_summary,
            "historical_forecast_15min": seamless_summary,
            "previous_runs_hourly": previous_runs_summary,
        },
        "notes": [
            "Use weather_previous_runs_issue_valid.csv for leakage-aware feature building where coverage exists.",
            "Use weather_hourly_archive.csv as the stable historical weather baseline.",
            "Use weather_15min_seamless.csv for dense weather features, but not as a substitute for true historical forecast issue archives.",
            "Production timestamps are assumed to already be expressed in the provided timezone.",
        ],
    }
    return manifest


def summarize_production(df: pd.DataFrame, output_path: Path, original_path: Path) -> dict:
    deltas = df["datetime"].sort_values().diff().dropna()
    top_delta = None
    if not deltas.empty:
        counts = deltas.value_counts()
        top_delta = str(counts.index[0])
    return {
        "source_file": str(original_path),
        "organized_file": str(output_path),
        "rows": int(len(df)),
        "coverage_start": str(df["datetime"].min()),
        "coverage_end": str(df["datetime"].max()),
        "top_time_step": top_delta,
        "duplicate_timestamps": int(df["datetime"].duplicated().sum()),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    production_csv_path = Path(args.production_csv)

    if not production_csv_path.exists():
        raise FileNotFoundError(f"Production CSV was not found: {production_csv_path}")

    production_df = read_and_normalize_production_csv(production_csv_path)
    requested_start = parse_date(args.start_date) if args.start_date else production_df["datetime"].min().date()
    requested_end = parse_date(args.end_date) if args.end_date else production_df["datetime"].max().date()

    production_df = production_df[
        (production_df["datetime"].dt.date >= requested_start)
        & (production_df["datetime"].dt.date <= requested_end)
    ].reset_index(drop=True)
    if production_df.empty:
        raise RuntimeError("No production rows remain after applying the requested date range.")

    metadata_dir = ensure_dir(output_dir / "metadata")
    production_dir = ensure_dir(output_dir / "raw" / "production")
    organized_production_path = production_dir / "solar_power_15min.csv"
    write_csv(production_df, organized_production_path)
    if production_csv_path.resolve() != organized_production_path.resolve():
        original_copy_path = production_dir / "original_input.csv"
        shutil.copyfile(production_csv_path, original_copy_path)

    production_summary = summarize_production(production_df, organized_production_path, production_csv_path)
    write_json(
        {
            "latitude": args.latitude,
            "longitude": args.longitude,
            "timezone": args.timezone,
            "tilt": args.tilt,
            "azimuth": args.azimuth,
            "installed_kwp": args.installed_kwp,
            "coverage_start": requested_start.isoformat(),
            "coverage_end": requested_end.isoformat(),
        },
        metadata_dir / "system_metadata.json",
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "solar-google-collab-data-loader/1.0"})

    archive_summary = download_archive_hourly_weather(
        session,
        latitude=args.latitude,
        longitude=args.longitude,
        timezone=args.timezone,
        start=requested_start,
        end=requested_end,
        variables=csv_list(args.archive_hourly_vars),
        chunk_days=args.seamless_chunk_days,
        output_dir=output_dir,
        force=args.force,
    )

    seamless_summary = None
    if not args.skip_seamless_history:
        try:
            seamless_summary = download_historical_forecast_15min(
                session,
                latitude=args.latitude,
                longitude=args.longitude,
                timezone=args.timezone,
                start=requested_start,
                end=requested_end,
                variables=csv_list(args.weather_15min_vars),
                chunk_days=args.seamless_chunk_days,
                output_dir=output_dir,
                force=args.force,
            )
        except RuntimeError as exc:
            seamless_summary = {
                "failed": True,
                "reason": str(exc),
                "warning": "Historical forecast 15-minute backfill failed, but the rest of the dataset was still organized.",
            }
            print(f"[warning] 15-min historical forecast backfill skipped after failure: {exc}", flush=True)

    previous_runs_summary = None
    if not args.skip_forecast_archive:
        previous_runs_start = max(requested_start, parse_date(args.previous_runs_start_date))
        if previous_runs_start <= requested_end:
            try:
                previous_runs_summary = download_previous_runs_archive(
                    session,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone=args.timezone,
                    start=previous_runs_start,
                    end=requested_end,
                    base_vars=csv_list(args.forecast_base_vars),
                    max_previous_day=args.previous_run_max_day,
                    chunk_days=args.previous_runs_chunk_days,
                    output_dir=output_dir,
                    force=args.force,
                )
            except RuntimeError as exc:
                previous_runs_summary = {
                    "failed": True,
                    "reason": str(exc),
                    "warning": "Previous-runs forecast archive backfill failed, so leakage-safe forecast features are incomplete.",
                }
                print(f"[warning] previous-runs backfill skipped after failure: {exc}", flush=True)
        else:
            previous_runs_summary = {
                "skipped": True,
                "reason": "Requested date range ends before previous-runs coverage start.",
                "coverage_start": args.previous_runs_start_date,
            }

    manifest = build_manifest(
        args=args,
        production_summary=production_summary,
        archive_summary=archive_summary,
        seamless_summary=seamless_summary,
        previous_runs_summary=previous_runs_summary,
        output_dir=output_dir,
    )
    manifest_path = metadata_dir / "data_manifest.json"
    write_json(manifest, manifest_path)

    print("\nData download complete.", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Production rows: {production_summary['rows']}", flush=True)
    print(f"Archive weather rows: {archive_summary['rows']}", flush=True)
    if seamless_summary:
        if seamless_summary.get("failed"):
            print("15-min weather rows: failed to download", flush=True)
        else:
            print(f"15-min weather rows: {seamless_summary['rows']}", flush=True)
    if previous_runs_summary and not previous_runs_summary.get("skipped") and not previous_runs_summary.get("failed"):
        print(f"Forecast archive rows: {previous_runs_summary['issue_valid_rows']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
