from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from netCDF4 import Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_OUTPUT_ROOT = Path(
    "data/work"
)
DEFAULT_TRIMS_DIR = Path(
    "data/training"
)

TIMES = ["0130", "1030", "1330", "2230"]
HIST_START = pd.Timestamp("2003-01-01")
HIST_END = pd.Timestamp("2014-12-31")
HIST_YEARS = list(range(2003, 2015))
EXPECTED_ROWS_PER_DAY = 97371

QUANTILES = np.round(np.arange(0.01, 1.00, 0.01), 2).astype("float64")
MIN_HIST_SAMPLES = 300
GRID_CHUNK = 2048
MAX_DAYS_PER_CALENDAR_MONTH = 400


def progress(iterable: Iterable, **kwargs):
    if tqdm is not None:
        return tqdm(iterable, **kwargs)
    desc = kwargs.get("desc", "progress")
    total = kwargs.get("total")
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    def _generator():
        count = 0
        last_reported = 0
        for item in iterable:
            count += 1
            if total and (
                count == 1
                or count == total
                or count - last_reported >= max(1, total // 20)
            ):
                print(f"{desc}: {count}/{total}", flush=True)
                last_reported = count
            yield item
        if total is None:
            print(f"{desc}: {count}", flush=True)

    return _generator()


def memory_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024 / 1024


def log(message: str) -> None:
    print(f"{message} | maxrss={memory_gb():.2f} GB", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Formal Code3: fit GCM-specific historical QDM from saved 2003-2014 "
            "Pre-QDM LST and TRIMS."
        )
    )
    parser.add_argument("--gcm", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trims-dir", type=Path, default=DEFAULT_TRIMS_DIR)
    parser.add_argument("--times", nargs="+", choices=TIMES, default=TIMES)
    parser.add_argument("--grid-chunk", type=int, default=GRID_CHUNK)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def month_window(target_month: int) -> list[int]:
    return [
        12 if target_month == 1 else target_month - 1,
        target_month,
        1 if target_month == 12 else target_month + 1,
    ]


def discover_gcms(output_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        gcms = requested
    else:
        gcms = sorted(
            p.name
            for p in output_root.iterdir()
            if p.is_dir() and (p / "02_历史时期" / "Pre-QDM").is_dir()
        )
    if not gcms:
        raise RuntimeError(f"No GCM Pre-QDM directories found under {output_root}")
    missing = [
        gcm
        for gcm in gcms
        if not (output_root / gcm / "02_历史时期" / "Pre-QDM").is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Pre-QDM folder for GCM(s): {missing}")
    return gcms


def pre_qdm_paths(output_root: Path, gcm: str) -> list[Path]:
    root = output_root / gcm / "02_历史时期" / "Pre-QDM"
    paths = [
        root / f"{gcm}_historical_Pre-QDM_{year}.parquet"
        for year in HIST_YEARS
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {gcm} Pre-QDM historical parquet files: {missing}"
        )
    return paths


def trims_paths(trims_dir: Path) -> list[Path]:
    paths = [trims_dir / f"{year}.parquet" for year in HIST_YEARS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing TRIMS historical parquet files: {missing}")
    return paths


def read_reference_grid(first_path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(first_path)
    if pf.num_row_groups < 1:
        raise RuntimeError(f"Empty parquet: {first_path}")
    frame = pf.read_row_group(0, columns=["lon", "lat"]).to_pandas()
    if len(frame) != EXPECTED_ROWS_PER_DAY:
        raise RuntimeError(
            f"Reference first day expected {EXPECTED_ROWS_PER_DAY:,} grid rows, "
            f"got {len(frame):,}"
        )
    frame["lon"] = frame["lon"].round(6)
    frame["lat"] = frame["lat"].round(6)
    if frame.duplicated(["lon", "lat"]).any():
        raise RuntimeError("Duplicate lon/lat in reference grid")
    frame = frame[["lon", "lat"]].reset_index(drop=True)
    frame["grid_id"] = np.arange(len(frame), dtype="int32")
    return frame


def assert_grid_order(frame: pd.DataFrame, grid: pd.DataFrame, context: str) -> None:
    if len(frame) != len(grid):
        raise RuntimeError(
            f"{context}: grid size mismatch: {len(frame):,} vs reference {len(grid):,}"
        )
    lon = frame["lon"].to_numpy(dtype="float64").round(6)
    lat = frame["lat"].to_numpy(dtype="float64").round(6)
    if not np.array_equal(
        lon, grid["lon"].to_numpy(dtype="float64")
    ) or not np.array_equal(
        lat, grid["lat"].to_numpy(dtype="float64")
    ):
        raise RuntimeError(
            f"{context}: lon/lat grid order differs from the reference grid."
        )


def split_row_group_by_date(
    frame: pd.DataFrame,
) -> Iterable[tuple[pd.Timestamp, pd.DataFrame]]:
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.copy()
    frame["__date_norm"] = dates
    for date_value, group in frame.groupby("__date_norm", sort=False):
        yield pd.Timestamp(date_value), group.drop(columns="__date_norm")


def cache_path(
    cache_dir: Path,
    kind: str,
    time_key: str,
    month: int,
) -> Path:
    return cache_dir / f"{kind}_{time_key}_month_{month:02d}.f32"


def create_month_memmaps(
    cache_dir: Path,
    kind: str,
    time_key: str,
    n_grid: int,
) -> dict[int, np.memmap]:
    stores = {}
    for month in range(1, 13):
        stores[month] = np.memmap(
            cache_path(cache_dir, kind, time_key, month),
            dtype="float32",
            mode="w+",
            shape=(MAX_DAYS_PER_CALENDAR_MONTH, n_grid),
        )
    return stores


def open_month_memmap(
    cache_dir: Path,
    kind: str,
    time_key: str,
    month: int,
    n_grid: int,
) -> np.memmap:
    return np.memmap(
        cache_path(cache_dir, kind, time_key, month),
        dtype="float32",
        mode="r",
        shape=(MAX_DAYS_PER_CALENDAR_MONTH, n_grid),
    )


def close_memmaps(stores: dict[int, np.memmap]) -> None:
    for mmap in stores.values():
        mmap.flush()
    stores.clear()
    gc.collect()


def build_month_cache(
    paths: list[Path],
    time_key: str,
    grid: pd.DataFrame,
    cache_dir: Path,
    kind: str,
) -> dict[int, int]:
    n_grid = len(grid)
    target = f"LST_{time_key}"
    stores = create_month_memmaps(cache_dir, kind, time_key, n_grid)
    counts = {month: 0 for month in range(1, 13)}
    seen_dates: set[pd.Timestamp] = set()

    try:
        for path in progress(
            paths,
            desc=f"{kind} cache {time_key}",
            unit="year",
        ):
            pf = pq.ParquetFile(path)
            required = ["date", "lon", "lat", target]
            missing = [c for c in required if c not in pf.schema_arrow.names]
            if missing:
                raise RuntimeError(f"{path}: missing columns {missing}")

            for rg in range(pf.num_row_groups):
                frame = pf.read_row_group(rg, columns=required).to_pandas()
                for date_value, day in split_row_group_by_date(frame):
                    if not (HIST_START <= date_value <= HIST_END):
                        continue
                    if date_value in seen_dates:
                        raise RuntimeError(
                            f"Duplicate {kind} date {date_value.date()}"
                        )
                    seen_dates.add(date_value)
                    assert_grid_order(
                        day,
                        grid,
                        f"{kind} {date_value.date()} {path.name}:row_group={rg + 1}",
                    )
                    month = int(date_value.month)
                    row = counts[month]
                    if row >= MAX_DAYS_PER_CALENDAR_MONTH:
                        raise RuntimeError(
                            f"Too many {kind} days for calendar month {month}"
                        )
                    stores[month][row, :] = day[target].to_numpy(dtype="float32")
                    counts[month] += 1
                del frame
            gc.collect()
    finally:
        close_memmaps(stores)

    years = {date.year for date in seen_dates}
    missing_years = sorted(set(HIST_YEARS) - years)
    if missing_years:
        raise RuntimeError(f"{kind} historical input missing years: {missing_years}")

    log(f"{kind} historical cache complete for {time_key}: days={len(seen_dates):,}")
    return counts


def nanquantile_linear(values: np.ndarray, q: np.ndarray, axis: int) -> np.ndarray:
    try:
        return np.nanquantile(values, q, axis=axis, method="linear")
    except TypeError:
        return np.nanquantile(values, q, axis=axis, interpolation="linear")


def qdm_output_path(output_root: Path, gcm: str, time_key: str) -> Path:
    return output_root / gcm / "01_QDM模型" / f"qdm_{time_key}.nc"


def create_qdm_netcdf(
    path: Path,
    gcm: str,
    time_key: str,
    grid: pd.DataFrame,
) -> Dataset:
    n_grid = len(grid)
    dataset = Dataset(path, "w", format="NETCDF4")
    dataset.createDimension("month", 12)
    dataset.createDimension("quantile", len(QUANTILES))
    dataset.createDimension("grid", n_grid)

    dataset.createVariable("month", "i1", ("month",))[:] = np.arange(
        1, 13, dtype="int8"
    )
    dataset.createVariable("quantile", "f4", ("quantile",))[:] = QUANTILES.astype(
        "float32"
    )
    dataset.createVariable("grid_id", "i4", ("grid",))[:] = np.arange(
        n_grid, dtype="int32"
    )
    dataset.createVariable("lon", "f8", ("grid",))[:] = grid["lon"].to_numpy(
        dtype="float64"
    )
    dataset.createVariable("lat", "f8", ("grid",))[:] = grid["lat"].to_numpy(
        dtype="float64"
    )

    q_chunk = (1, len(QUANTILES), min(4096, n_grid))
    g_chunk = (1, min(8192, n_grid))

    for name in ["q_pred_hist", "q_obs_hist", "historical_bias"]:
        dataset.createVariable(
            name,
            "f4",
            ("month", "quantile", "grid"),
            zlib=True,
            complevel=4,
            chunksizes=q_chunk,
            fill_value=np.float32(np.nan),
        )

    for name in ["n_valid_pred", "n_valid_obs", "n_valid"]:
        dataset.createVariable(
            name,
            "i4",
            ("month", "grid"),
            zlib=True,
            complevel=4,
            chunksizes=g_chunk,
        )

    dataset.createVariable(
        "qdm_valid",
        "i1",
        ("month", "grid"),
        zlib=True,
        complevel=4,
        chunksizes=g_chunk,
    )

    dataset.code = "Formal-Code3"
    dataset.source_algorithm = "Code4-01"
    dataset.gcm = gcm
    dataset.time_of_day = time_key
    dataset.historical_reference_period = "2003-01-01/2014-12-31"
    dataset.observation_reference = "TRIMS"
    dataset.prediction_reference = "saved_Pre-QDM_GCM_historical_LST"
    dataset.method = "gridwise_seasonal_additive_QDM"
    dataset.seasonal_window = "centered_3_month_calendar_window"
    dataset.quantiles = "P1-P99, 1% step"
    dataset.interpolation = "linear"
    dataset.tail_policy_for_future_application = "clip_to_P1_P99"
    dataset.min_historical_samples = MIN_HIST_SAMPLES
    dataset.bias_definition = "historical_bias = q_obs_hist - q_pred_hist"
    dataset.daily_pairing_used = (
        "false; climatological distributions are compared"
    )
    dataset.created_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return dataset


def fit_qdm_for_time(
    cache_dir: Path,
    output_root: Path,
    gcm: str,
    time_key: str,
    grid: pd.DataFrame,
    pred_counts: dict[int, int],
    obs_counts: dict[int, int],
    grid_chunk: int,
    months_to_fit: list[int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:

    selected_months = list(range(1, 13)) if months_to_fit is None else list(months_to_fit)
    if not selected_months or len(set(selected_months)) != len(selected_months) or any(m < 1 or m > 12 for m in selected_months):
        raise ValueError("Invalid QDM calendar month selection")
    n_grid = len(grid)
    out_path = qdm_output_path(output_root, gcm, time_key)
    coverage_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    pred_maps = {
        month: open_month_memmap(cache_dir, "pred", time_key, month, n_grid)
        for month in range(1, 13)
    }
    obs_maps = {
        month: open_month_memmap(cache_dir, "obs", time_key, month, n_grid)
        for month in range(1, 13)
    }

    dataset = create_qdm_netcdf(out_path, gcm, time_key, grid)
    q_pred_var = dataset.variables["q_pred_hist"]
    q_obs_var = dataset.variables["q_obs_hist"]
    bias_var = dataset.variables["historical_bias"]
    n_pred_var = dataset.variables["n_valid_pred"]
    n_obs_var = dataset.variables["n_valid_obs"]
    n_valid_var = dataset.variables["n_valid"]
    qdm_valid_var = dataset.variables["qdm_valid"]

    try:
        for target_month in progress(
            selected_months,
            desc=f"fit QDM {gcm} {time_key}",
            unit="month",
        ):
            months = month_window(target_month)
            n_valid_grids = 0
            sum_q_pred = np.zeros(len(QUANTILES), dtype="float64")
            sum_q_obs = np.zeros(len(QUANTILES), dtype="float64")
            sum_bias = np.zeros(len(QUANTILES), dtype="float64")
            summary_count = np.zeros(len(QUANTILES), dtype="int64")

            for start in range(0, n_grid, grid_chunk):
                end = min(start + grid_chunk, n_grid)

                pred_block = np.concatenate(
                    [
                        np.asarray(pred_maps[m][:pred_counts[m], start:end])
                        for m in months
                    ],
                    axis=0,
                )
                obs_block = np.concatenate(
                    [
                        np.asarray(obs_maps[m][:obs_counts[m], start:end])
                        for m in months
                    ],
                    axis=0,
                )

                n_pred = np.sum(np.isfinite(pred_block), axis=0).astype("int32")
                n_obs = np.sum(np.isfinite(obs_block), axis=0).astype("int32")
                n_valid = np.minimum(n_pred, n_obs).astype("int32")
                valid_grid = (
                    (n_pred >= MIN_HIST_SAMPLES)
                    & (n_obs >= MIN_HIST_SAMPLES)
                )
                n_valid_grids += int(np.sum(valid_grid))

                width = end - start
                q_pred = np.full(
                    (len(QUANTILES), width),
                    np.nan,
                    dtype="float32",
                )
                q_obs = np.full(
                    (len(QUANTILES), width),
                    np.nan,
                    dtype="float32",
                )

                if np.any(valid_grid):
                    q_pred[:, valid_grid] = nanquantile_linear(
                        pred_block[:, valid_grid],
                        QUANTILES,
                        axis=0,
                    ).astype("float32")
                    q_obs[:, valid_grid] = nanquantile_linear(
                        obs_block[:, valid_grid],
                        QUANTILES,
                        axis=0,
                    ).astype("float32")

                historical_bias = q_obs - q_pred

                q_pred_var[target_month - 1, :, start:end] = q_pred
                q_obs_var[target_month - 1, :, start:end] = q_obs
                bias_var[target_month - 1, :, start:end] = historical_bias
                n_pred_var[target_month - 1, start:end] = n_pred
                n_obs_var[target_month - 1, start:end] = n_obs
                n_valid_var[target_month - 1, start:end] = n_valid
                qdm_valid_var[target_month - 1, start:end] = valid_grid.astype(
                    "int8"
                )

                finite = np.isfinite(historical_bias)
                sum_q_pred += np.nansum(q_pred, axis=1)
                sum_q_obs += np.nansum(q_obs, axis=1)
                sum_bias += np.nansum(historical_bias, axis=1)
                summary_count += np.sum(finite, axis=1).astype("int64")

                del (
                    pred_block,
                    obs_block,
                    n_pred,
                    n_obs,
                    n_valid,
                    valid_grid,
                    q_pred,
                    q_obs,
                    historical_bias,
                    finite,
                )
                gc.collect()

            coverage_rows.append(
                {
                    "gcm": gcm,
                    "time": time_key,
                    "month": target_month,
                    "seasonal_months": "+".join(f"{m:02d}" for m in months),
                    "n_total_grids": n_grid,
                    "n_valid_grids": n_valid_grids,
                    "valid_fraction": (
                        n_valid_grids / n_grid if n_grid else np.nan
                    ),
                    "n_pred_days_in_window": int(
                        sum(pred_counts[m] for m in months)
                    ),
                    "n_obs_days_in_window": int(
                        sum(obs_counts[m] for m in months)
                    ),
                }
            )

            with np.errstate(invalid="ignore", divide="ignore"):
                mean_pred = sum_q_pred / summary_count
                mean_obs = sum_q_obs / summary_count
                mean_bias = sum_bias / summary_count

            for q, pred_mean, obs_mean, bias_mean, n in zip(
                QUANTILES,
                mean_pred,
                mean_obs,
                mean_bias,
                summary_count,
            ):
                summary_rows.append(
                    {
                        "gcm": gcm,
                        "time": time_key,
                        "month": target_month,
                        "quantile": float(q),
                        "n_valid_grids": int(n),
                        "spatial_mean_q_pred_hist": (
                            float(pred_mean) if np.isfinite(pred_mean) else np.nan
                        ),
                        "spatial_mean_q_obs_hist": (
                            float(obs_mean) if np.isfinite(obs_mean) else np.nan
                        ),
                        "spatial_mean_historical_bias": (
                            float(bias_mean) if np.isfinite(bias_mean) else np.nan
                        ),
                    }
                )

            dataset.sync()
            log(
                f"QDM complete: {gcm} {time_key} target_month={target_month:02d} "
                f"valid={n_valid_grids:,}/{n_grid:,}"
            )
    finally:
        dataset.close()
        pred_maps.clear()
        obs_maps.clear()
        gc.collect()

    log(f"saved historical QDM model: {out_path}")
    return coverage_rows, summary_rows


def qdm_outputs_are_usable(
    output_root: Path,
    gcm: str,
    times: list[str],
) -> bool:
    for time_key in times:
        path = qdm_output_path(output_root, gcm, time_key)
        if not path.exists():
            return False
        try:
            with Dataset(path, "r") as dataset:
                if str(getattr(dataset, "gcm", "")) != gcm:
                    return False
                if tuple(dataset.variables["historical_bias"].shape) != (
                    12,
                    len(QUANTILES),
                    EXPECTED_ROWS_PER_DAY,
                ):
                    return False
                if tuple(dataset.variables["qdm_valid"].shape) != (
                    12,
                    EXPECTED_ROWS_PER_DAY,
                ):
                    return False
        except (KeyError, OSError, RuntimeError, ValueError):
            return False
    return True


def remove_time_cache(cache_dir: Path, kind: str, time_key: str) -> None:
    for month in range(1, 13):
        path = cache_path(cache_dir, kind, time_key, month)
        if path.exists():
            path.unlink()


def write_config(
    output_root: Path,
    gcm: str,
    pre_qdm: list[Path],
    trims: list[Path],
    times: list[str],
    pred_counts_all: dict[str, dict[int, int]],
) -> None:
    qdm_dir = output_root / gcm / "01_QDM模型"
    config = {
        "code": "Formal-Code3",
        "source_algorithm": "Code4-01",
        "purpose": "GCM-specific historical QDM for historical and future CMB-LST",
        "gcm": gcm,
        "historical_reference_period": "2003-01-01/2014-12-31",
        "historical_prediction_source": "Formal Code2 saved Pre-QDM LST",
        "historical_reference_lst": "TRIMS",
        "times": times,
        "qdm_type": "gridwise_seasonal_additive_QDM",
        "historical_seasonal_window": "centered_3_month_calendar_window",
        "quantile_min": 0.01,
        "quantile_max": 0.99,
        "quantile_step": 0.01,
        "quantile_count": 99,
        "interpolation": "linear",
        "tail_policy_for_future_application": "clip_to_P1_P99",
        "min_historical_samples_pred": MIN_HIST_SAMPLES,
        "min_historical_samples_obs": MIN_HIST_SAMPLES,
        "bias_definition": "historical_bias = Q_TRIMS_hist - Q_GCMpred_hist",
        "daily_GCM_vs_TRIMS_metrics": False,
        "future_or_2015plus_data_used": False,
        "HiCPC_input_second_bias_correction": False,
        "grid_policy": (
            "must exactly match TRIMS/Code1 grid size, coordinates, and order; "
            "no silent remapping"
        ),
        "pre_qdm_sources": [str(p) for p in pre_qdm],
        "trims_reference_files": [str(p) for p in trims],
        "prediction_month_counts": pred_counts_all,
        "output_files": [f"qdm_{time_key}.nc" for time_key in times],
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    (qdm_dir / "qdm_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_gcm(
    output_root: Path,
    trims_dir: Path,
    gcm: str,
    times: list[str],
    grid_chunk: int,
    overwrite: bool,
    keep_cache: bool,
    dry_run: bool,
) -> None:
    pre_qdm = pre_qdm_paths(output_root, gcm)
    trims = trims_paths(trims_dir)
    grid = read_reference_grid(trims[0])

    qdm_dir = output_root / gcm / "01_QDM模型"
    qdm_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite and qdm_outputs_are_usable(output_root, gcm, times):
        log(f"Reuse existing formal QDM set: {gcm}")
        return

    print("\n" + "=" * 100, flush=True)
    print(f"Formal Code3 QDM setup: GCM={gcm}", flush=True)
    print(
        f"Pre-QDM input : {output_root / gcm / '02_历史时期' / 'Pre-QDM'}",
        flush=True,
    )
    print(f"TRIMS input   : {trims_dir}", flush=True)
    print(f"QDM output    : {qdm_dir}", flush=True)
    print(f"Times         : {times}", flush=True)
    print(f"Period        : {HIST_START.date()} / {HIST_END.date()}", flush=True)
    print("=" * 100, flush=True)

    if dry_run:
        return

    cache_dir = qdm_dir / "_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    coverage_all: list[dict[str, object]] = []
    summary_all: list[dict[str, object]] = []
    pred_counts_all: dict[str, dict[int, int]] = {}

    try:
        for time_key in times:
            pred_counts = build_month_cache(
                pre_qdm,
                time_key,
                grid,
                cache_dir,
                "pred",
            )
            pred_counts_all[time_key] = pred_counts

            obs_counts = build_month_cache(
                trims,
                time_key,
                grid,
                cache_dir,
                "obs",
            )

            coverage_rows, summary_rows = fit_qdm_for_time(
                cache_dir=cache_dir,
                output_root=output_root,
                gcm=gcm,
                time_key=time_key,
                grid=grid,
                pred_counts=pred_counts,
                obs_counts=obs_counts,
                grid_chunk=grid_chunk,
            )
            coverage_all.extend(coverage_rows)
            summary_all.extend(summary_rows)

            remove_time_cache(cache_dir, "pred", time_key)
            remove_time_cache(cache_dir, "obs", time_key)
            gc.collect()

        pd.DataFrame(coverage_all).to_csv(
            qdm_dir / "qdm_coverage.csv",
            index=False,
        )
        pd.DataFrame(summary_all).to_csv(
            qdm_dir / "qdm_quantile_summary.csv",
            index=False,
        )
        write_config(
            output_root,
            gcm,
            pre_qdm,
            trims,
            times,
            pred_counts_all,
        )
    finally:
        if not keep_cache and cache_dir.exists():
            shutil.rmtree(cache_dir)

    log(f"Formal Code3 complete for GCM={gcm}. Output: {qdm_dir}")


def main() -> None:
    args = parse_args()
    if args.grid_chunk <= 0:
        raise ValueError("--grid-chunk must be positive")

    gcms = discover_gcms(args.output_root, args.gcm)
    for gcm in gcms:
        process_gcm(
            output_root=args.output_root,
            trims_dir=args.trims_dir,
            gcm=gcm,
            times=args.times,
            grid_chunk=args.grid_chunk,
            overwrite=args.overwrite,
            keep_cache=args.keep_cache,
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("\nDry-run passed. No QDM model was fitted.", flush=True)


if __name__ == "__main__":
    main()
