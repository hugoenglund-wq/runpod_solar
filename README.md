# runpod_solar

Solar forecasting project for `d0`, `d1`, `d2`, `d3` home PV forecasts on RunPod.

## Models

- `day_0`: same-day daylight forecast for the rest of today
- `day_1`: tomorrow
- `day_2`: day after tomorrow
- `day_3`: three days ahead

Evaluation is daylight-only and uses time-causal backtesting over roughly `20%` of issue days.

## RunPod Setup

Clone the repo and run:

```bash
bash scripts/setup_runpod_noaa.sh
```

That installs:

- Python dependencies from [requirements.txt](C:/Users/hugoe/Desktop/solar_google_collab/requirements.txt)
- native `ecCodes` runtime for NOAA GFS GRIB decoding
- a verification step via [verify_noaa_runtime.py](C:/Users/hugoe/Desktop/solar_google_collab/scripts/verify_noaa_runtime.py)

## NOAA GFS Backfill

To build leakage-safe NOAA GFS training data for `d1-d3`:

```bash
python scripts/download_noaa_gfs_point_archive.py --output-dir data
```

This writes:

- `data/raw/weather/noaa_gfs_hourly/weather_noaa_gfs_issue_valid.csv`

When that file exists, the training pipeline automatically prefers NOAA GFS for `d1-d3`.

## Training

Run the full training job with:

```bash
python scripts/train_all_models.py --artifacts-dir artifacts
```

Artifacts are written under:

- `artifacts/models`
- `artifacts/metrics`
- `artifacts/predictions`
- `artifacts/datasets`

## Smoke Test

For a local quick check:

```bash
python scripts/smoke_test_one_week.py
```

## Notes

- `d0` can train over the full production history.
- `d1-d3` require historical forecasts, not future observed weather.
- NOAA GFS backfill needs GRIB decoding support, which is why the RunPod setup installs `libeccodes-dev`.
