#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 4)),
        help="Parallel NOAA issue-day workers. Default: min(8, cpu_count).",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_name_for_issue(issue_time_utc: datetime) -> str:
    return f"{issue_time_utc.strftime('%Y%m%dT%H%MZ')}.csv"


def build_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "runpod-solar-noaa-gfs-downloader/1.0"})
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=32,
        pool_maxsize=32,
        max_retries=2,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


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


def process_issue_time(
    *,
    issue_time_utc: datetime,
    chunk_path: Path,
    latitude: float,
    longitude: float,
    timezone: str,
) -> dict[str, object]:
    session = build_http_session()
    try:
        discovered = discover_catalog_files_for_issue(session, issue_time_utc=issue_time_utc)
        if not discovered:
            return {"status": "missing", "issue_date": issue_time_utc.date().isoformat()}

        available_leads = [item.lead_hours for item in discovered]
        required_leads = required_leads_for_local_day_offsets(
            issue_time_utc=issue_time_utc,
            timezone=timezone,
            day_offsets=(1, 2, 3),
        )
        selected = [item for item in discovered if item.lead_hours in required_leads]
        if not selected:
            return {"status": "missing", "issue_date": issue_time_utc.date().isoformat()}

        print(
            f"[download] NOAA GFS issue {issue_time_utc.isoformat()} "
            f"available_leads={min(available_leads)}-{max(available_leads)} "
            f"selected={len(selected)}",
            flush=True,
        )
        issue_frame = extract_issue_point_forecast(
            session,
            catalog_files=selected,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
        )
        if issue_frame.empty:
            return {"status": "empty", "issue_date": issue_time_utc.date().isoformat()}

        tmp_path = chunk_path.with_suffix(".tmp")
        issue_frame.to_csv(tmp_path, index=False)
        tmp_path.replace(chunk_path)
        return {
            "status": "saved",
            "chunk_name": chunk_path.name,
            "rows": int(len(issue_frame)),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "issue_date": issue_time_utc.date().isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        session.close()


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
    chunks_dir = ensure_dir(weather_dir / "chunks")
    output_csv = weather_dir / "weather_noaa_gfs_issue_valid.csv"
    metadata_json = weather_dir / "weather_noaa_gfs_metadata.json"
    if output_csv.exists() and not args.force:
        print(f"[skip] NOAA GFS archive already exists: {output_csv}")
        return 0

    issue_times = build_daily_issue_times(
        start_date=start_date,
        end_date=end_date,
        issue_cycle_hour_utc=args.issue_cycle_hour_utc,
    )
    selected_chunk_paths = [chunks_dir / chunk_name_for_issue(issue_time_utc) for issue_time_utc in issue_times]
    if args.force:
        for chunk_path in selected_chunk_paths:
            if chunk_path.exists():
                chunk_path.unlink()
        if output_csv.exists():
            output_csv.unlink()
        if metadata_json.exists():
            metadata_json.unlink()

    missing_issue_dates: list[str] = []
    error_issue_dates: list[dict[str, str]] = []
    pending_jobs: list[tuple[datetime, Path]] = []
    for issue_time_utc in issue_times:
        chunk_path = chunks_dir / chunk_name_for_issue(issue_time_utc)
        if chunk_path.exists() and not args.force:
            print(f"[resume] NOAA GFS issue {issue_time_utc.isoformat()} chunk exists, skipping", flush=True)
            continue
        pending_jobs.append((issue_time_utc, chunk_path))

    if pending_jobs:
        print(f"[start] NOAA GFS backfill workers={args.workers} pending_issues={len(pending_jobs)}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {
                executor.submit(
                    process_issue_time,
                    issue_time_utc=issue_time_utc,
                    chunk_path=chunk_path,
                    latitude=args.latitude,
                    longitude=args.longitude,
                    timezone=args.timezone,
                ): issue_time_utc
                for issue_time_utc, chunk_path in pending_jobs
            }
            for future in as_completed(future_map):
                result = future.result()
                status = result["status"]
                if status in {"missing", "empty"}:
                    missing_issue_dates.append(str(result["issue_date"]))
                    continue
                if status == "error":
                    error_issue_dates.append(
                        {"issue_date": str(result["issue_date"]), "error": str(result["error"])}
                    )
                    print(
                        f"[error] NOAA GFS issue {result['issue_date']} failed: {result['error']}",
                        flush=True,
                    )
                    continue
                if status == "saved":
                    print(f"[saved] {result['chunk_name']} rows={result['rows']}", flush=True)

    chunk_paths = [path for path in selected_chunk_paths if path.exists()]
    if not chunk_paths:
        raise RuntimeError(
            "NOAA GFS download did not produce any point forecasts. "
            "Check archive coverage and ecCodes availability."
        )

    frames = [pd.read_csv(path, parse_dates=["forecast_issue_time", "forecast_valid_time"]) for path in chunk_paths]
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
        "chunk_files": int(len(chunk_paths)),
        "missing_issue_dates": missing_issue_dates,
        "error_issue_dates": error_issue_dates,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[ok] wrote {len(combined)} NOAA GFS rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
