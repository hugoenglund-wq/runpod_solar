#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PANEL_LAT, PANEL_LON, TIMEZONE
from src.noaa_gfs import (
    build_daily_issue_times,
    discover_catalog_files_for_issue,
    extract_issue_point_forecast,
    required_leads_for_local_day_offsets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download leakage-safe NOAA GFS point forecasts for d1-d3 training."
    )
    parser.add_argument("--production-csv", default="solar_dataset_2021_2024.csv")
    parser.add_argument("--latitude", type=float, default=PANEL_LAT)
    parser.add_argument("--longitude", type=float, default=PANEL_LON)
    parser.add_argument("--timezone", default=TIMEZONE)
    parser.add_argument("--start-date", help="YYYY-MM-DD. Defaults to production start.")
    parser.add_argument("--end-date", help="YYYY-MM-DD. Defaults to production end.")
    parser.add_argument("--issue-cycle-hour-utc", type=int, default=18)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_and_normalize_production_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    expected = {"datetime", "power_w"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required production columns: {sorted(missing)}")
    df = df.loc[:, ["datetime", "power_w"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    df["power_w"] = pd.to_numeric(df["power_w"], errors="raise")
    return df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    production_path = Path(args.production_csv)
    if not production_path.exists():
        raise FileNotFoundError(f"Production CSV was not found: {production_path}")

    production = read_and_normalize_production_csv(production_path)
    start_date = parse_date(args.start_date) if args.start_date else production["datetime"].min().date()
    end_date = parse_date(args.end_date) if args.end_date else production["datetime"].max().date()

    output_dir = Path(args.output_dir)
    weather_dir = ensure_dir(output_dir / "raw" / "weather" / "noaa_gfs_hourly")
    output_csv = weather_dir / "weather_noaa_gfs_issue_valid.csv"
    metadata_json = weather_dir / "weather_noaa_gfs_metadata.json"
    if output_csv.exists() and not args.force:
        print(f"[skip] NOAA GFS archive already exists: {output_csv}")
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": "runpod-solar-noaa-gfs-downloader/1.0"})

    frames: list[pd.DataFrame] = []
    missing_issue_dates: list[str] = []
    for issue_time_utc in build_daily_issue_times(
        start_date=start_date,
        end_date=end_date,
        issue_cycle_hour_utc=args.issue_cycle_hour_utc,
    ):
        discovered = discover_catalog_files_for_issue(session, issue_time_utc=issue_time_utc)
        if not discovered:
            missing_issue_dates.append(issue_time_utc.date().isoformat())
            continue

        available_leads = [item.lead_hours for item in discovered]
        required_leads = required_leads_for_local_day_offsets(
            issue_time_utc=issue_time_utc,
            timezone=args.timezone,
            day_offsets=(1, 2, 3),
        )
        selected = [item for item in discovered if item.lead_hours in required_leads]
        if not selected:
            missing_issue_dates.append(issue_time_utc.date().isoformat())
            continue

        print(
            f"[download] NOAA GFS issue {issue_time_utc.isoformat()} "
            f"available_leads={min(available_leads)}-{max(available_leads)} "
            f"selected={len(selected)}",
            flush=True,
        )
        issue_frame = extract_issue_point_forecast(
            session,
            catalog_files=selected,
            latitude=args.latitude,
            longitude=args.longitude,
            timezone=args.timezone,
        )
        if not issue_frame.empty:
            frames.append(issue_frame)

    if not frames:
        raise RuntimeError(
            "NOAA GFS download did not produce any point forecasts. "
            "Check archive coverage and ecCodes availability."
        )

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["forecast_issue_time", "forecast_valid_time"]
    )
    combined = combined.sort_values(["forecast_issue_time", "forecast_valid_time"]).reset_index(drop=True)
    combined.to_csv(output_csv, index=False)

    metadata = {
        "source": "NOAA NCEI GFS archive",
        "latitude": args.latitude,
        "longitude": args.longitude,
        "timezone": args.timezone,
        "issue_cycle_hour_utc": args.issue_cycle_hour_utc,
        "coverage_start": str(combined["forecast_issue_time"].min()),
        "coverage_end": str(combined["forecast_valid_time"].max()),
        "rows": int(len(combined)),
        "missing_issue_dates": missing_issue_dates,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[ok] wrote {len(combined)} NOAA GFS rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
