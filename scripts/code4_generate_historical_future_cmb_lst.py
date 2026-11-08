from __future__ import annotations

import argparse
import calendar
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys
import os
import json
import re
import resource
import shutil
from datetime import datetime, timezone
from pathlib import Path


from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from netCDF4 import Dataset
from historical_quantile_correction import apply_qdm_to_target_year_parallel
from release_netcdf_writer import (
    NetCDFTableSink,
    published_year_path,
    validate_release_year,
    write_future_release_year,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_MODEL_DIR = Path("data/models")
DEFAULT_QDM_DIR = Path("data/work")
DEFAULT_HICPC_ROOT = Path("data/hicpc")
DEFAULT_OUTPUT_ROOT = Path("data/work")
DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_DIAG_ROOT = Path("data/diagnostics")
DEFAULT_RELEASE_ROOT = Path("data/release")
DEFAULT_MANIFEST_ROOT = Path("data/manifests")

TIMES = ["0130", "1030", "1330", "2230"]
TIME_INDEX = {time_key: i for i, time_key in enumerate(TIMES)}

HIST_START_YEAR = 2003
HIST_END_YEAR = 2014
FUTURE_START_YEAR = 2015
REQUESTED_END_YEAR_DEFAULT = 2099
FUTURE_WINDOW_YEARS = 12
EXPECTED_ROWS_PER_DAY = 97371
MAX_DAYS_PER_YEAR = 366
MAX_HIST_DAYS_PER_CALENDAR_MONTH = 31 * (HIST_END_YEAR - HIST_START_YEAR + 1)
QDM_GRID_CHUNK = 2048
PREDICTION_BATCH_DAYS = 5
QUANTILES = np.round(np.arange(0.01, 1.00, 0.01), 2).astype("float64")

SSP_LABELS = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}


FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "date_sin": ("date_sin",),
    "date_cos": ("date_cos",),
    "lon": ("lon",),
    "lat": ("lat",),
    "tasmax_cmfd_daily": ("tasmax_cmfd_daily", "tasmax", "tasmax_hicpc_daily", "tasmax_daily"),
    "tasmin_cmfd_daily": ("tasmin_cmfd_daily", "tasmin", "tasmin_hicpc_daily", "tasmin_daily"),
    "tas_cmfd_daily": ("tas_cmfd_daily", "tas", "tas_hicpc_daily", "tas_daily"),
    "pr": ("pr", "pr_hicpc_daily", "pr_daily"),
    "DEM": ("DEM", "dem"),
    "slope": ("slope",),
    "aspect": ("aspect",),
}

FINAL_COLUMNS = ["date", "lon", "lat", *[f"LST_{t}" for t in TIMES]]


def progress(iterable: Iterable, **kwargs):
    if tqdm is not None:
        kwargs.setdefault("mininterval", 1.0)
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("leave", True)
        return tqdm(iterable, **kwargs)

    desc = kwargs.get("desc", "progress")
    total = kwargs.get("total")
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    def fallback():
        count = 0
        last = 0
        for item in iterable:
            count += 1
            if total and (
                count == 1
                or count == total
                or count - last >= max(1, total // 20)
            ):
                print(f"{desc}: {count}/{total}", flush=True)
                last = count
            yield item
        if total is None:
            print(f"{desc}: {count}", flush=True)

    return fallback()


def memory_gb() -> float:

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024 / 1024


def log(message: str) -> None:
    print(f"{message} | maxrss={memory_gb():.2f} GB", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def month_window(target_month: int) -> list[int]:
    return [
        12 if target_month == 1 else target_month - 1,
        target_month,
        1 if target_month == 12 else target_month + 1,
    ]


def normalize_ssp(value: str) -> str:
    raw = value.strip().lower().replace("_", "-").replace(" ", "")
    direct = {
        "ssp126": "ssp126",
        "126": "ssp126",
        "ssp1-2.6": "ssp126",
        "ssp12.6": "ssp126",
        "ssp1-26": "ssp126",
        "ssp245": "ssp245",
        "245": "ssp245",
        "ssp2-4.5": "ssp245",
        "ssp24.5": "ssp245",
        "ssp2-45": "ssp245",
        "ssp370": "ssp370",
        "370": "ssp370",
        "ssp3-7.0": "ssp370",
        "ssp37.0": "ssp370",
        "ssp3-70": "ssp370",
        "ssp585": "ssp585",
        "585": "ssp585",
        "ssp5-8.5": "ssp585",
        "ssp58.5": "ssp585",
        "ssp5-85": "ssp585",
    }
    if raw in direct:
        return direct[raw]
    raise ValueError(f"Unsupported SSP value: {value!r}")


def scenario_output_dir(output_root: Path, gcm: str, ssp: str) -> Path:
    return output_root / gcm / "03_未来时期" / SSP_LABELS[ssp]


def scenario_output_path(output_root: Path, gcm: str, ssp: str, year: int) -> Path:
    label = SSP_LABELS[ssp]
    return scenario_output_dir(output_root, gcm, ssp) / f"{gcm}_{label}_{year}.parquet"


def final_scenario_output_path(args: argparse.Namespace, gcm: str, ssp: str, year: int) -> Path:
    if args.final_format == "netcdf":
        return published_year_path(args.release_root, gcm, SSP_LABELS[ssp], year)
    return scenario_output_path(args.output_root, gcm, ssp, year)


def final_scenario_output_dir(args: argparse.Namespace, gcm: str, ssp: str) -> Path:
    if args.final_format == "netcdf":
        return published_year_path(args.release_root, gcm, SSP_LABELS[ssp], 2015).parent
    return scenario_output_dir(args.output_root, gcm, ssp)


def final_historical_output_path(args: argparse.Namespace, gcm: str, year: int) -> Path:
    if args.final_format == "netcdf":
        return published_year_path(args.release_root, gcm, "historical", year)
    return historical_output_path(args.output_root, gcm, year)


def valid_final_output(path: Path, n_grid: int, year: int) -> bool:
    if path.suffix == ".nc":
        return validate_release_year(path, n_grid, year)
    return validate_existing_final_parquet(path, n_grid)


def scenario_cache_dir(cache_root: Path, gcm: str, ssp: str) -> Path:
    label = SSP_LABELS[ssp]
    return cache_root / gcm / "future" / label / f"cache_{label}"


def scenario_diag_dir(diag_root: Path, gcm: str, ssp: str) -> Path:
    return diag_root / gcm / "future" / SSP_LABELS[ssp]


def prediction_cache_path(cache_dir: Path, year: int) -> Path:
    return cache_dir / f"pred_{year}.f32"


def presence_cache_path(cache_dir: Path, year: int) -> Path:
    return cache_dir / f"present_{year}.u1"


def corrected_year_cache_path(cache_dir: Path, year: int) -> Path:
    return cache_dir / f"corr_{year}.f32"


def prediction_cache_marker_path(cache_dir: Path) -> Path:
    return cache_dir / "prediction_cache_complete.json"


def expected_prediction_cache_bytes(n_grid: int) -> int:
    return MAX_DAYS_PER_YEAR * len(TIMES) * n_grid * np.dtype("float32").itemsize


def expected_presence_bytes() -> int:
    return MAX_DAYS_PER_YEAR * np.dtype("uint8").itemsize


def file_signature(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CMOD LST: GCM meteorological inputs, LightGBM prediction, historical-quantile "
            "additive bias correction and annual NetCDF output (Parquet optional)"
        )
    )
    parser.add_argument(
        "--gcm",
        default="all",
        help="GCM name. Use 'all' to process every GCM with a complete formal QDM set.",
    )
    parser.add_argument(
        "--ssp",
        default="all",
        help="ssp126/ssp245/ssp370/ssp585 (or SSP1-2.6 etc.). Use 'all' for all four SSPs.",
    )
    parser.add_argument(
        "--hicpc-feature",
        type=Path,
        default=None,
        help=(
            "Exact HiCPC feature parquet file or directory. Only allowed when one GCM and one SSP are selected. "
            "Otherwise feature Parquet files are discovered under --hicpc-root."
        ),
    )
    parser.add_argument("--hicpc-root", type=Path, default=DEFAULT_HICPC_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--qdm-dir", type=Path, default=DEFAULT_QDM_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Internal model/QDM/Pre-QDM/diagnostic production root.")
    parser.add_argument("--final-format", choices=["netcdf", "parquet"], default="netcdf",
                        help="Final annual LST product. NetCDF writes public release directly (default).")
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT,
                        help="Public release product root; used with --final-format netcdf.")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT,
                        help="Separate directory for SHA256 and annual NetCDF release manifests.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--diagnostics-root", type=Path, default=DEFAULT_DIAG_ROOT)
    parser.add_argument("--start-year", type=int, default=FUTURE_START_YEAR)
    parser.add_argument("--end-year", type=int, default=REQUESTED_END_YEAR_DEFAULT)
    parser.add_argument(
        "--require-end-year",
        action="store_true",
        help="If requested --end-year is unavailable in HiCPC features, stop instead of clamping to actual coverage.",
    )
    parser.add_argument("--grid-chunk", type=int, default=QDM_GRID_CHUNK)
    parser.add_argument(
        "--qdm-year-workers", type=int,
        default=1,
        help="Number of concurrent processing years (default: 1).",
    )
    parser.add_argument(
        "--qdm-threads-per-year", type=int, default=4,
        help="Number of threads for independent time/grid tasks within each year.",
    )
    parser.add_argument(
        "--qdm-parallel-grid-chunk", type=int, default=1024,
        help="Number of grid cells per correction task.",
    )
    parser.add_argument("--batch-days", type=int, default=PREDICTION_BATCH_DAYS)
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Recompute and atomically replace already-existing annual final products.",
    )
    parser.add_argument(
        "--rebuild-prediction-cache",
        action="store_true",
        help="Delete and rebuild the raw future LightGBM prediction cache for each selected scenario.",
    )
    parser.add_argument(
        "--keep-prediction-cache",
        action="store_true",
        help="Keep the large raw prediction cache after a scenario finishes successfully. Default: delete after success.",
    )
    parser.add_argument(
        "--skip-missing-scenarios",
        action="store_true",
        help="When looping multiple GCM/SSP combinations, log missing inputs and continue instead of stopping.",
    )
    parser.add_argument(
        "--check-grid-every-day",
        action="store_true",
        help="Strict but slower: compare lon/lat order with historical QDM grid for every future day.",
    )
    parser.add_argument(
        "--skip-historical",
        action="store_true",
        help="Skip 2003-2014 historical QDM correction.",
    )
    parser.add_argument(
        "--skip-future",
        action="store_true",
        help="Skip 2015+ future projection.",
    )
    parser.add_argument(
        "--overwrite-historical",
        action="store_true",
        help="Recompute and atomically replace corrected historical annual products.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve models/QDM/HiCPC inputs and planned historical/future outputs without generating data.",
    )
    return parser.parse_args()


def load_code1_models(model_dir: Path) -> tuple[dict[str, object], list[str], dict[str, object]]:


    info_path = model_dir / "model_info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)

    info = json.loads(info_path.read_text(encoding="utf-8"))

    training_period = str(
        info.get(
            "final_evaluation_model_training_period",
            info.get("training_period", ""),
        )
    )
    if training_period != "2003-2014":
        raise RuntimeError(
            "This workflow requires the full LightGBM models "
            "trained on 2003-2014; "
            f"model_info.json reports training_period={training_period!r}."
        )

    features = list(info.get("feature_names", []))
    if not features:
        raise RuntimeError(
            "model_info.json does not contain feature_names"
        )

    models: dict[str, object] = {}
    for time_key in TIMES:
        path = model_dir / f"q50_{time_key}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)

        payload = joblib.load(path)
        if not isinstance(payload, dict) or "model" not in payload:
            raise RuntimeError(f"Invalid LightGBM model payload: {path}")

        payload_features = list(payload.get("features", features))
        if payload_features != features:
            raise RuntimeError(
                f"Feature order mismatch between {path.name} and model_info.json"
            )

        payload_period = str(payload.get("training_period", training_period))
        if payload_period != "2003-2014":
            raise RuntimeError(
                f"{path.name} does not represent the full 2003-2014 model; "
                f"payload training_period={payload_period!r}"
            )

        models[time_key] = payload["model"]

    return models, features, info


def discover_gcms(qdm_dir: Path) -> list[str]:
    if not qdm_dir.exists():
        raise FileNotFoundError(qdm_dir)
    gcms: list[str] = []
    for gcm_dir in sorted(p for p in qdm_dir.iterdir() if p.is_dir()):
        qdm_model_dir = gcm_dir / "01_QDM模型"
        if all((qdm_model_dir / f"qdm_{t}.nc").exists() for t in TIMES):
            gcms.append(gcm_dir.name)
    if not gcms:
        raise FileNotFoundError(
            f"No complete formal QDM set found under {qdm_dir}; expected "
            "<GCM>/01_QDM模型/qdm_0130/1030/1330/2230.nc"
        )
    return gcms


def resolve_hicpc_sources(
    explicit: Path | None,
    root: Path,
    gcm: str,
    ssp: str,
) -> list[Path]:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        if explicit.is_file():
            if explicit.suffix.lower() != ".parquet":
                raise RuntimeError(f"--hicpc-feature must be parquet: {explicit}")
            return [explicit]
        paths = sorted(explicit.rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet found under {explicit}")
        return paths

    scenario_dir = root / gcm / "future" / SSP_LABELS[ssp]
    if not scenario_dir.is_dir():
        raise FileNotFoundError(
            f"Future HiCPC input directory not found for {gcm} {SSP_LABELS[ssp]}: "
            f"{scenario_dir}"
        )
    paths = sorted(scenario_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No future HiCPC parquet found for {gcm} {SSP_LABELS[ssp]} under "
            f"{scenario_dir}"
        )
    return paths

def resolve_feature_map(schema_names: list[str], model_features: list[str]) -> dict[str, str]:
    available = set(schema_names)
    mapping: dict[str, str] = {}
    for feature in model_features:
        aliases = FEATURE_ALIASES.get(feature, (feature,))
        found = next((name for name in aliases if name in available), None)
        if found is None:
            raise RuntimeError(
                f"HiCPC feature parquet cannot supply model feature {feature!r}. "
                f"Allowed aliases={aliases}; available columns={schema_names}"
            )
        mapping[feature] = found
    return mapping


def source_columns_for_prediction(feature_map: dict[str, str]) -> list[str]:
    return list(dict.fromkeys(["date", "lon", "lat", *feature_map.values()]))


def build_model_frame(
    frame: pd.DataFrame,
    model_features: list[str],
    feature_map: dict[str, str],
) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for feature in model_features:
        source = feature_map[feature]
        values = frame[source]
        if feature in {"lon", "lat"}:
            values = values.round(6)
        data[feature] = values.astype("float32", copy=False)
    out = pd.DataFrame(data, columns=model_features)
    bad = out.isna().any(axis=1)
    if bad.any():
        raise RuntimeError(f"HiCPC model input contains {int(bad.sum()):,} rows with NaN predictor values")
    return out


def qdm_path(qdm_dir: Path, gcm: str, time_key: str) -> Path:
    return qdm_dir / gcm / "01_QDM模型" / f"qdm_{time_key}.nc"


def open_and_validate_qdm_set(qdm_dir: Path, gcm: str) -> tuple[dict[str, Dataset], np.ndarray, np.ndarray]:
    handles: dict[str, Dataset] = {}
    ref_lon: np.ndarray | None = None
    ref_lat: np.ndarray | None = None

    try:
        for time_key in TIMES:
            path = qdm_path(qdm_dir, gcm, time_key)
            if not path.exists():
                raise FileNotFoundError(path)
            ds = Dataset(path, "r")
            handles[time_key] = ds

            for variable in ["quantile", "lon", "lat", "historical_bias", "qdm_valid"]:
                if variable not in ds.variables:
                    raise RuntimeError(f"{path.name} missing variable {variable!r}")

            gcm_attr = str(getattr(ds, "gcm", gcm))
            if gcm_attr != gcm:
                raise RuntimeError(f"{path.name}: gcm attribute={gcm_attr!r}, expected {gcm!r}")

            code_attr = str(getattr(ds, "code", ""))
            if code_attr and code_attr not in {"Code4-01", "Formal-Code3"}:
                raise RuntimeError(
                    f"{path.name}: code attribute={code_attr!r}; "
                    "expected historical QDM generated by Code4-01 or Formal-Code3"
                )

            period = str(getattr(ds, "historical_reference_period", ""))
            if period and period != "2003-01-01/2014-12-31":
                raise RuntimeError(f"{path.name}: unexpected historical_reference_period={period!r}")

            q = np.asarray(ds.variables["quantile"][:], dtype="float64")
            if q.shape != QUANTILES.shape or not np.allclose(q, QUANTILES, atol=1e-6):
                raise RuntimeError(f"{path.name}: quantile nodes are not P1-P99 at 1% intervals")

            lon = np.asarray(ds.variables["lon"][:], dtype="float64").round(6)
            lat = np.asarray(ds.variables["lat"][:], dtype="float64").round(6)
            if ref_lon is None:
                ref_lon = lon
                ref_lat = lat
            else:
                if not np.array_equal(lon, ref_lon) or not np.array_equal(lat, ref_lat):
                    raise RuntimeError(f"{path.name}: QDM grid differs from the other time-of-day QDM files")

            if ds.variables["historical_bias"].shape != (12, len(QUANTILES), len(lon)):
                raise RuntimeError(
                    f"{path.name}: historical_bias shape={ds.variables['historical_bias'].shape}, "
                    f"expected (12, {len(QUANTILES)}, {len(lon)})"
                )
            if ds.variables["qdm_valid"].shape != (12, len(lon)):
                raise RuntimeError(f"{path.name}: unexpected qdm_valid shape={ds.variables['qdm_valid'].shape}")

        assert ref_lon is not None and ref_lat is not None
        if len(ref_lon) != EXPECTED_ROWS_PER_DAY:
            raise RuntimeError(
                f"historical QDM grid contains {len(ref_lon):,} cells; expected project grid {EXPECTED_ROWS_PER_DAY:,}."
            )
        return handles, ref_lon, ref_lat
    except Exception:
        for ds in handles.values():
            ds.close()
        raise


def close_qdm_set(handles: dict[str, Dataset]) -> None:
    for ds in handles.values():
        ds.close()
    handles.clear()


def assert_day_grid(
    frame: pd.DataFrame,
    qdm_lon: np.ndarray,
    qdm_lat: np.ndarray,
    context: str,
) -> None:
    if len(frame) != len(qdm_lon):
        raise RuntimeError(
            f"{context}: grid size mismatch: {len(frame):,} vs Code4-01 QDM {len(qdm_lon):,}"
        )
    lon = frame["lon"].to_numpy(dtype="float64").round(6)
    lat = frame["lat"].to_numpy(dtype="float64").round(6)
    if not np.array_equal(lon, qdm_lon) or not np.array_equal(lat, qdm_lat):
        raise RuntimeError(
            f"{context}: lon/lat grid order differs from historical QDM/Code1 target grid. "
            "Formal Code4 does not silently reorder or nearest-neighbour remap."
        )


def split_row_group_by_date(frame: pd.DataFrame) -> Iterable[tuple[pd.Timestamp, pd.DataFrame]]:
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    work = frame.copy()
    work["__date_norm"] = dates
    for date_value, group in work.groupby("__date_norm", sort=False):
        yield pd.Timestamp(date_value), group.drop(columns="__date_norm")


def parquet_row_group_date_bounds(pf: pq.ParquetFile, row_group: int) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    names = pf.schema_arrow.names
    if "date" not in names:
        return None, None
    col_idx = names.index("date")
    stats = pf.metadata.row_group(row_group).column(col_idx).statistics
    if stats is None or not stats.has_min_max:
        return None, None
    try:
        lower = pd.Timestamp(stats.min).normalize()
        upper = pd.Timestamp(stats.max).normalize()
        return lower, upper
    except Exception:
        return None, None


def iter_future_days(
    sources: list[Path],
    columns: list[str],
    min_year: int = FUTURE_START_YEAR,
    max_year: int = REQUESTED_END_YEAR_DEFAULT,
) -> Iterable[tuple[pd.Timestamp, pd.DataFrame, Path, int]]:
    min_date = pd.Timestamp(min_year, 1, 1)
    max_date = pd.Timestamp(max_year, 12, 31)

    for path in sources:
        pf = pq.ParquetFile(path)
        missing = [c for c in columns if c not in pf.schema_arrow.names]
        if missing:
            raise RuntimeError(f"{path}: missing columns {missing}")

        for rg in range(pf.num_row_groups):
            lower, upper = parquet_row_group_date_bounds(pf, rg)
            if upper is not None and upper < min_date:
                continue
            if lower is not None and lower > max_date:
                continue


            if lower is None or upper is None:
                date_only = pf.read_row_group(rg, columns=["date"]).to_pandas()["date"]
                dates = pd.to_datetime(date_only, errors="raise")
                if dates.max() < min_date or dates.min() > max_date:
                    del date_only, dates
                    continue
                del date_only, dates

            frame = pf.read_row_group(rg, columns=columns).to_pandas()
            for date_value, day in split_row_group_by_date(frame):
                if min_date <= date_value <= max_date:
                    yield date_value, day, path, rg
            del frame


def ensure_presence_map(cache_dir: Path, year: int) -> np.memmap:
    path = presence_cache_path(cache_dir, year)
    if path.exists():
        if path.stat().st_size != expected_presence_bytes():
            raise RuntimeError(f"Presence cache size mismatch: {path}")
        return np.memmap(path, dtype="uint8", mode="r+", shape=(MAX_DAYS_PER_YEAR,))

    mmap = np.memmap(path, dtype="uint8", mode="w+", shape=(MAX_DAYS_PER_YEAR,))
    mmap[:] = 0
    mmap.flush()
    return mmap


def ensure_prediction_map(cache_dir: Path, year: int, n_grid: int, mode: str = "r+") -> np.memmap:
    path = prediction_cache_path(cache_dir, year)
    expected = expected_prediction_cache_bytes(n_grid)
    if path.exists():
        if path.stat().st_size != expected:
            raise RuntimeError(
                f"Prediction cache size mismatch: {path}, got {path.stat().st_size:,}, expected {expected:,}"
            )
        return np.memmap(path, dtype="float32", mode=mode, shape=(MAX_DAYS_PER_YEAR, len(TIMES), n_grid))

    if mode == "r":
        raise FileNotFoundError(path)
    return np.memmap(
        path,
        dtype="float32",
        mode="w+",
        shape=(MAX_DAYS_PER_YEAR, len(TIMES), n_grid),
    )


def open_corrected_map(cache_dir: Path, year: int, n_grid: int) -> np.memmap:
    path = corrected_year_cache_path(cache_dir, year)
    if path.exists():
        path.unlink()
    return np.memmap(
        path,
        dtype="float32",
        mode="w+",
        shape=(MAX_DAYS_PER_YEAR, len(TIMES), n_grid),
    )


def model_signatures(model_dir: Path) -> dict[str, dict[str, int]]:
    paths = [model_dir / "model_info.json", *[model_dir / f"q50_{t}.pkl" for t in TIMES]]
    return {path.name: file_signature(path) for path in paths}


def cache_marker_is_usable(
    marker: dict[str, object],
    gcm: str,
    ssp: str,
    n_grid: int,
    model_dir: Path,
) -> bool:
    if marker.get("gcm") != gcm or marker.get("ssp") != ssp:
        return False
    if int(marker.get("n_grid", -1)) != n_grid:
        return False
    if marker.get("times") != TIMES:
        return False
    if marker.get("model_signatures") != model_signatures(model_dir):
        return False
    return True


def flush_prediction_batch(
    batch_days: list[pd.DataFrame],
    batch_meta: list[tuple[pd.Timestamp, Path, int]],
    models: dict[str, object],
    model_features: list[str],
    feature_map: dict[str, str],
    cache_dir: Path,
    presence_maps: dict[int, np.memmap],
    n_grid: int,
) -> int:
    if not batch_days:
        return 0

    combined = pd.concat(batch_days, ignore_index=True)
    x = build_model_frame(combined, model_features, feature_map)

    predictions: dict[str, np.ndarray] = {}
    for time_key in TIMES:
        values = np.asarray(models[time_key].predict(x, num_threads=512), dtype="float32")
        if len(values) != len(combined):
            raise RuntimeError(f"Prediction length mismatch for {time_key}")
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite LightGBM predictions produced for {time_key}")
        predictions[time_key] = values

    written = 0
    year_maps: dict[int, np.memmap] = {}
    try:
        offset = 0
        for date_value, _path, _rg in batch_meta:
            end = offset + n_grid
            year = date_value.year
            doy0 = int(date_value.dayofyear) - 1
            if year not in presence_maps:
                presence_maps[year] = ensure_presence_map(cache_dir, year)
            present = presence_maps[year]

            if present[doy0] == 1:

                offset = end
                continue

            if year not in year_maps:
                year_maps[year] = ensure_prediction_map(cache_dir, year, n_grid, mode="r+")
            pred_map = year_maps[year]
            for time_key in TIMES:
                pred_map[doy0, TIME_INDEX[time_key], :] = predictions[time_key][offset:end]
            pred_map.flush()
            present[doy0] = 1
            present.flush()
            written += 1
            offset = end
    finally:
        for mmap in year_maps.values():
            mmap.flush()
        year_maps.clear()
        del combined, x
        predictions.clear()
        gc.collect()

    return written


def validate_complete_year_presence(year: int, present: np.ndarray) -> tuple[bool, int]:
    count = int(np.sum(present == 1))
    expected = 366 if calendar.isleap(year) else 365
    if count == expected:
        return True, count

    if calendar.isleap(year) and count == 365:
        return True, count
    return False, count


def cached_complete_years(cache_dir: Path) -> tuple[list[int], dict[int, int]]:
    years: list[int] = []
    counts: dict[int, int] = {}
    for path in sorted(cache_dir.glob("present_*.u1")):
        match = re.match(r"present_(\d{4})\.u1$", path.name)
        if not match:
            continue
        year = int(match.group(1))
        present = np.memmap(path, dtype="uint8", mode="r", shape=(MAX_DAYS_PER_YEAR,))
        complete, count = validate_complete_year_presence(year, present)
        counts[year] = count
        del present
        if complete:
            years.append(year)
    return sorted(years), counts


def cached_complete_prediction_years(cache_dir: Path, n_grid: int) -> tuple[list[int], dict[int, int]]:
    complete, counts = cached_complete_years(cache_dir)
    expected = expected_prediction_cache_bytes(n_grid)
    usable = []
    for year in complete:
        path = prediction_cache_path(cache_dir, year)
        if path.exists() and path.stat().st_size == expected:
            usable.append(year)
    return usable, counts


def contiguous_future_years(complete_years: list[int]) -> list[int]:
    years_set = set(complete_years)
    if FUTURE_START_YEAR not in years_set:
        raise RuntimeError(
            f"Future cache does not contain a complete {FUTURE_START_YEAR} year; cannot start the SSP sequence."
        )
    result = []
    year = FUTURE_START_YEAR
    while year in years_set:
        result.append(year)
        year += 1
    return result


def first_year_to_scan_for_prediction_cache(cache_dir: Path, n_grid: int, max_year: int) -> int:
    complete, _counts = cached_complete_prediction_years(cache_dir, n_grid)
    if not complete:
        return FUTURE_START_YEAR
    try:
        contiguous = contiguous_future_years(complete)
    except RuntimeError:
        return FUTURE_START_YEAR
    return min(contiguous[-1] + 1, max_year + 1)
def existing_valid_output_years(
    args: argparse.Namespace,
    gcm: str,
    ssp: str,
    start_year: int,
    end_year: int,
    n_grid: int,
) -> list[int]:
    years: list[int] = []
    for year in range(start_year, end_year + 1):
        path = final_scenario_output_path(args, gcm, ssp, year)
        if path.exists() and valid_final_output(path, n_grid, year):
            years.append(year)
    return years


def source_year_bounds(sources: list[Path]) -> tuple[int, int] | None:
    years: list[int] = []
    for path in sources:
        years.extend(int(value) for value in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", path.stem))
    if not years:
        return None
    return min(years), max(years)


def build_future_prediction_cache(
    sources: list[Path],
    models: dict[str, object],
    model_features: list[str],
    feature_map: dict[str, str],
    qdm_lon: np.ndarray,
    qdm_lat: np.ndarray,
    cache_dir: Path,
    gcm: str,
    ssp: str,
    batch_days_n: int,
    model_dir: Path,
    check_grid_every_day: bool,
) -> dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker_path = prediction_cache_marker_path(cache_dir)

    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not cache_marker_is_usable(marker, gcm, ssp, len(qdm_lon), model_dir):
            raise RuntimeError(
                f"Existing prediction cache marker is incompatible with the current models/grid: {marker_path}. "
                "Rerun with --rebuild-prediction-cache."
            )

        years = list(map(int, marker.get("complete_years", [])))
        if years and all(prediction_cache_path(cache_dir, y).exists() for y in years):
            if max(years) >= REQUESTED_END_YEAR_DEFAULT:
                log(
                    f"Reuse completed prediction cache: {gcm} {ssp}, "
                    f"years={min(years)}-{max(years)}"
                )
                return marker
            log(
                f"Existing prediction cache marker is partial: {gcm} {SSP_LABELS[ssp]}, "
                f"years={min(years)}-{max(years)}; continue filling cache"
            )
        else:
            log(
                f"Existing prediction cache marker is incomplete: {gcm} {SSP_LABELS[ssp]}; "
                "rechecking day-level cache files"
            )

    columns = source_columns_for_prediction(feature_map)
    presence_maps: dict[int, np.memmap] = {}
    checked_sources: set[Path] = set()
    seen_this_run: set[pd.Timestamp] = set()
    batch_days: list[pd.DataFrame] = []
    batch_meta: list[tuple[pd.Timestamp, Path, int]] = []
    n_new_days = 0
    n_reused_days = 0
    n_seen_days = 0
    scan_start_year = first_year_to_scan_for_prediction_cache(
        cache_dir, len(qdm_lon), REQUESTED_END_YEAR_DEFAULT
    )
    if scan_start_year > REQUESTED_END_YEAR_DEFAULT:
        log(
            f"Prediction cache already covers through {scan_start_year - 1}: "
            f"{gcm} {SSP_LABELS[ssp]}"
        )
    elif scan_start_year > FUTURE_START_YEAR:
        log(
            f"Resume prediction cache from {scan_start_year}: "
            f"{gcm} {SSP_LABELS[ssp]}"
        )

    scan_total_days = (
        (
            pd.Timestamp(REQUESTED_END_YEAR_DEFAULT, 12, 31)
            - pd.Timestamp(scan_start_year, 1, 1)
        ).days
        + 1
        if scan_start_year <= REQUESTED_END_YEAR_DEFAULT
        else 0
    )

    def flush() -> None:
        nonlocal n_new_days, batch_days, batch_meta
        if not batch_days:
            return
        n_new_days += flush_prediction_batch(
            batch_days,
            batch_meta,
            models,
            model_features,
            feature_map,
            cache_dir,
            presence_maps,
            len(qdm_lon),
        )
        batch_days = []
        batch_meta = []

    iterator = iter_future_days(
        sources,
        columns,
        min_year=scan_start_year,
        max_year=REQUESTED_END_YEAR_DEFAULT,
    )

    for date_value, day, source_path, rg in progress(
        iterator,
        desc=f"future LightGBM prediction {gcm} {ssp}",
        unit="day",
        total=scan_total_days,
    ):
        n_seen_days += 1
        print(
            f"[进度] {gcm} {SSP_LABELS[ssp]}: "
            f"第 {n_seen_days}/{scan_total_days} 天，日期={date_value.date()}",
            flush=True,
        )
        if date_value in seen_this_run:
            raise RuntimeError(
                f"Duplicate HiCPC future date {date_value.date()} found while reading {source_path}. "
                "Check whether duplicate feature parquet files were supplied."
            )
        seen_this_run.add(date_value)

        if len(day) != len(qdm_lon):
            raise RuntimeError(
                f"{source_path.name}: row_group={rg + 1}, date={date_value.date()} has {len(day):,} rows; "
                f"expected {len(qdm_lon):,}."
            )

        if check_grid_every_day or source_path not in checked_sources:
            assert_day_grid(
                day,
                qdm_lon,
                qdm_lat,
                f"HiCPC {gcm} {ssp} {date_value.date()} {source_path.name}:row_group={rg + 1}",
            )
            checked_sources.add(source_path)

        year = date_value.year
        doy0 = int(date_value.dayofyear) - 1
        if year not in presence_maps:
            presence_maps[year] = ensure_presence_map(cache_dir, year)
        if presence_maps[year][doy0] == 1:
            n_reused_days += 1
            del day
            continue

        batch_days.append(day)
        batch_meta.append((date_value, source_path, rg))
        if len(batch_days) >= batch_days_n:
            flush()

    flush()
    for mmap in presence_maps.values():
        mmap.flush()
    presence_maps.clear()
    gc.collect()

    complete, counts = cached_complete_prediction_years(cache_dir, len(qdm_lon))
    contiguous = contiguous_future_years(complete)
    if len(contiguous) < FUTURE_WINDOW_YEARS:
        raise RuntimeError(
            f"Need at least {FUTURE_WINDOW_YEARS} contiguous prepared future years; "
            f"found {len(contiguous)} ({contiguous[0] if contiguous else 'none'}-"
            f"{contiguous[-1] if contiguous else 'none'})."
        )


    for year in range(FUTURE_START_YEAR, contiguous[-1] + 1):
        if year not in complete:
            raise RuntimeError(f"Incomplete/missing future year inside continuous SSP sequence: {year}")

    marker = {
        "code": "Code4-02-prediction-cache",
        "gcm": gcm,
        "ssp": ssp,
        "ssp_label": SSP_LABELS[ssp],
        "n_grid": len(qdm_lon),
        "times": TIMES,
        "complete_years": contiguous,
        "calendar_day_counts": {str(y): counts.get(y, 0) for y in contiguous},
        "actual_future_start": contiguous[0],
        "actual_future_end": contiguous[-1],
        "model_dir": str(model_dir),
        "model_signatures": model_signatures(model_dir),
        "feature_sources": [str(p) for p in sources],
        "new_days_predicted_this_run": n_new_days,
        "cached_days_reused_this_run": n_reused_days,
        "created_utc": utc_now(),
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"Prediction cache complete: {gcm} {ssp}, actual future coverage "
        f"{contiguous[0]}-{contiguous[-1]}, new_days={n_new_days}, reused_days={n_reused_days}"
    )
    return marker


def year_date_info(cache_dir: Path, year: int) -> dict[str, object]:
    path = presence_cache_path(cache_dir, year)
    if not path.exists():
        raise FileNotFoundError(path)
    present = np.memmap(path, dtype="uint8", mode="r", shape=(MAX_DAYS_PER_YEAR,))
    rows = np.flatnonzero(np.asarray(present) == 1).astype("int16")
    del present

    if len(rows) == 0:
        raise RuntimeError(f"No cached dates for {year}")

    start = pd.Timestamp(year, 1, 1)
    dates = pd.DatetimeIndex([start + pd.Timedelta(days=int(row)) for row in rows])
    if np.any(dates.year != year):
        raise RuntimeError(f"Presence cache contains an invalid day-of-year position for {year}")

    month_rows: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        month_rows[month] = rows[dates.month.to_numpy() == month].astype("int16")

    return {
        "rows": rows,
        "dates": dates,
        "month_rows": month_rows,
        "n_days": len(rows),
    }


def interpolation_from_percentile(
    p: np.ndarray,
    historical_bias_grid_quantile: np.ndarray,
) -> np.ndarray:


    p_clip = np.clip(p, float(QUANTILES[0]), float(QUANTILES[-1]))
    u = (p_clip - float(QUANTILES[0])) / 0.01
    lo_idx = np.floor(u + 1e-12).astype("int16")
    lo_idx = np.clip(lo_idx, 0, len(QUANTILES) - 1)
    hi_idx = np.minimum(lo_idx + 1, len(QUANTILES) - 1)
    frac = u - lo_idx

    d_lo = np.take_along_axis(historical_bias_grid_quantile, lo_idx.T, axis=1).T
    d_hi = np.take_along_axis(historical_bias_grid_quantile, hi_idx.T, axis=1).T
    return d_lo + frac * (d_hi - d_lo)


def apply_qdm_to_target_year(
    cache_dir: Path,
    qdm_handles: dict[str, Dataset],
    year_infos: dict[int, dict[str, object]],
    target_year: int,
    first_future_year: int,
    last_future_year: int,
    n_grid: int,
    grid_chunk: int,
):

    return apply_qdm_to_target_year_parallel(
        sys.modules[__name__],
        cache_dir=cache_dir,
        qdm_handles=qdm_handles,
        year_infos=year_infos,
        target_year=target_year,
        first_future_year=first_future_year,
        last_future_year=last_future_year,
        n_grid=n_grid,
        grid_chunk=grid_chunk,
        max_workers=4,
    )


def _code4_parallel_qdm_job(job: dict) -> dict:
    from pathlib import Path
    gcm,ssp,year=job["gcm"],job["ssp"],int(job["year"])
    cache_dir=Path(job["cache_dir"])
    handles,lon,lat=open_and_validate_qdm_set(Path(job["qdm_dir"]),gcm)
    corrected_cache=None
    try:
        if job["final_format"]=="netcdf":
            out=published_year_path(Path(job["release_root"]),gcm,SSP_LABELS[ssp],year)
        else:
            out=scenario_output_path(Path(job["output_root"]),gcm,ssp,year)

        completed_marker = Path(job["diagnostics_root"])/"completed"/gcm/SSP_LABELS[ssp]/f"{year}.json"
        if out.exists() and completed_marker.is_file():
            marker=json.loads(completed_marker.read_text())
            if (marker.get("method")=="historical_quantile_additive"
                    and marker.get("file_size")==out.stat().st_size
                    and marker.get("release_path")==str(out)
                    and valid_final_output(out,len(lon),year)):
                return {"year":year,"status":"already_validated","annual":[],"coverage":[]}
        if out.exists() and not job["overwrite_output"]:
            if valid_final_output(out,len(lon),year):
                return {"year":year,"status":"already_valid","annual":[],"coverage":[]}
            raise RuntimeError("Incomplete pre-existing output: "+str(out))
        if out.exists() and not valid_final_output(out,len(lon),year):
            raise RuntimeError("Refusing to replace invalid or unrecognized previous release "+str(out))
        infos={y:year_date_info(cache_dir,y) for y in range(job["first"],job["last"]+1)}
        corrected_cache,annual,coverage=apply_qdm_to_target_year_parallel(
            sys.modules[__name__],
            cache_dir=cache_dir,qdm_handles=handles,year_infos=infos,
            target_year=year,first_future_year=job["first"],last_future_year=job["last"],
            n_grid=len(lon),grid_chunk=job["grid_chunk"],max_workers=job["threads"],
        )

        if job["final_format"]=="netcdf":
            write_future_release_year(
                output=out,corrected_cache=corrected_cache,year_info=infos[year],
                year=year,lon=lon,lat=lat,gcm=gcm,scenario=SSP_LABELS[ssp],
                manifest_root=Path(job["manifest_root"]),overwrite=bool(job["overwrite_output"]),
            )
        else:
            write_final_year_parquet(
                output_path=out,corrected_cache=corrected_cache,cache_dir=cache_dir,
                year_info=infos[year],year=year,lon=lon,lat=lat,
            )
        if not valid_final_output(out,len(lon),year):
            raise RuntimeError("Annual release validation failed: "+str(out))
        completed_marker.parent.mkdir(parents=True,exist_ok=True)
        record={
            "method":"historical_quantile_additive",
            "year":year,"gcm":gcm,"ssp":SSP_LABELS[ssp],
            "release_path":str(out),"file_size":out.stat().st_size,
            "run_status":"COMMITTED_BITWISE_VERIFIED",
            "historical_cdf": "2003-2014 GCM Pre-QDM q_pred_hist",
            "historical_bias": "2003-2014 TRIMS minus Pre-QDM",
            "tail":"constant historical_bias(P01/P99), no LST clipping",
            "completed_utc":utc_now()
        }
        tmp=completed_marker.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
        tmp.replace(completed_marker)
        return {"year":year,"status":"B_complete","annual":annual,"coverage":coverage}
    finally:
        if corrected_cache is not None and corrected_cache.exists():
            corrected_cache.unlink()
        close_qdm_set(handles)
        gc.collect()


def _code4_parallel_qdm_batch(args,gcm,ssp,cache_dir,years,first,last,diag_dir) -> None:
    if not years:
        return
    max_workers=min(len(years),args.qdm_year_workers)
    tasks=[dict(
        gcm=gcm,ssp=ssp,year=year,cache_dir=str(cache_dir),
        qdm_dir=str(args.qdm_dir),output_root=str(args.output_root),
        release_root=str(args.release_root),manifest_root=str(args.manifest_root),
        first=first,last=last,grid_chunk=args.qdm_parallel_grid_chunk,
        threads=args.qdm_threads_per_year,final_format=args.final_format,
        overwrite_output=args.overwrite_output,diagnostics_root=str(args.diagnostics_root),
    ) for year in years]
    log(f"Start QDM year-parallel {gcm} {SSP_LABELS[ssp]}: {len(tasks)} years, "
        f"{max_workers} year processes x {args.qdm_threads_per_year} threads")

    keys=("CMB_LST_DISABLE_THREAD_TUNING","OMP_NUM_THREADS",
          "OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")
    previous={key:os.environ.get(key) for key in keys}
    try:
        os.environ.update({
            "CMB_LST_DISABLE_THREAD_TUNING":"1",
            "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
            "MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1",
        })
        with ProcessPoolExecutor(max_workers=max_workers,mp_context=mp.get_context("spawn")) as pool:
            mapping={pool.submit(_code4_parallel_qdm_job,task):task["year"] for task in tasks}
            for future in as_completed(mapping):
                year=mapping[future]
                result=future.result()
                if result["annual"]:
                    annual_df=merge_diagnostic_csv(
                        diag_dir/"annual_diagnostics.csv",result["annual"],
                        key_columns=["year","time"],
                    )
                    merge_diagnostic_csv(
                        diag_dir/"qdm_coverage_by_month.csv",result["coverage"],
                        key_columns=["year","time","month"],
                    )
                    write_trend_diagnostics(annual_df,diag_dir/"trend_diagnostics.csv")
                log(f"QDM_PARALLEL_YEAR_DONE {gcm} {SSP_LABELS[ssp]} {year} {result['status']}")
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    log(f"QDM parallel completed all {len(tasks)} selected years of {gcm} {SSP_LABELS[ssp]}")


def validate_existing_final_parquet(path: Path, n_grid: int) -> bool:
    if not path.exists():
        return False
    try:
        pf = pq.ParquetFile(path)
        if pf.schema_arrow.names != FINAL_COLUMNS:
            return False
        if pf.num_row_groups not in {365, 366}:
            return False
        for rg in range(pf.num_row_groups):
            if pf.metadata.row_group(rg).num_rows != n_grid:
                return False
        return True
    except Exception:
        return False


def write_final_year_parquet(
    output_path: Path,
    corrected_cache: Path,
    cache_dir: Path,
    year_info: dict[str, object],
    year: int,
    lon: np.ndarray,
    lat: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_grid = len(lon)
    corr = np.memmap(
        corrected_cache,
        dtype="float32",
        mode="r",
        shape=(MAX_DAYS_PER_YEAR, len(TIMES), n_grid),
    )

    rows = np.asarray(year_info["rows"], dtype="int16")
    dates = pd.DatetimeIndex(year_info["dates"])
    if len(rows) != len(dates):
        raise RuntimeError(f"Internal year index mismatch for {year}")

    schema = pa.schema(
        [
            ("date", pa.timestamp("ns")),
            ("lon", pa.float64()),
            ("lat", pa.float64()),
            *[(f"LST_{time_key}", pa.float32()) for time_key in TIMES],
        ]
    )

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = pq.ParquetWriter(tmp_path, schema=schema, compression="zstd")
    try:
        for row, date_value in zip(rows, dates):
            lst_columns = {}
            for time_key in TIMES:
                values = np.asarray(corr[int(row), TIME_INDEX[time_key], :], dtype="float32")
                if not np.all(np.isfinite(values)):
                    bad = int(np.sum(~np.isfinite(values)))
                    raise RuntimeError(
                        f"{year} {date_value.date()} {time_key}: final corrected field contains {bad:,} non-finite cells"
                    )
                lst_columns[f"LST_{time_key}"] = pa.array(values, type=pa.float32())

            date_array = np.full(
                n_grid,
                np.datetime64(date_value.to_datetime64(), "ns"),
                dtype="datetime64[ns]",
            )
            table = pa.Table.from_arrays(
                [
                    pa.array(date_array, type=pa.timestamp("ns")),
                    pa.array(lon, type=pa.float64()),
                    pa.array(lat, type=pa.float64()),
                    *[lst_columns[f"LST_{t}"] for t in TIMES],
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=n_grid)
            del table, lst_columns, date_array
    finally:
        writer.close()
        del corr
        gc.collect()


    tmp_path.replace(output_path)
    log(f"saved final release parquet: {output_path}")


def historical_pre_qdm_dir(output_root: Path, gcm: str) -> Path:
    return output_root / gcm / "02_历史时期" / "Pre-QDM"


def historical_output_dir(output_root: Path, gcm: str) -> Path:
    return output_root / gcm / "02_历史时期" / "CMB-LST"


def historical_pre_qdm_path(output_root: Path, gcm: str, year: int) -> Path:
    return historical_pre_qdm_dir(output_root, gcm) / f"{gcm}_historical_Pre-QDM_{year}.parquet"


def historical_output_path(output_root: Path, gcm: str, year: int) -> Path:
    return historical_output_dir(output_root, gcm) / f"{gcm}_historical_CMB-LST_{year}.parquet"


def _legacy_qpred_cache_path(cache_dir: Path, time_key: str, month: int) -> Path:
    return cache_dir / f"pred_{time_key}_month_{month:02d}.f32"


def _q_pred_hist_ready(ds: Dataset) -> bool:
    if "q_pred_hist" not in ds.variables:
        return False
    if ds.variables["q_pred_hist"].shape != (12, len(QUANTILES), len(ds.variables["lon"])):
        return False
    if str(getattr(ds, "code", "")) == "Formal-Code3":
        return True
    return int(getattr(ds, "q_pred_hist_complete", 0)) == 1


def _nanquantile_linear(values: np.ndarray, q: np.ndarray, axis: int) -> np.ndarray:
    try:
        return np.nanquantile(values, q, axis=axis, method="linear")
    except TypeError:
        return np.nanquantile(values, q, axis=axis, interpolation="linear")


def _build_legacy_qpred_month_cache(
    input_paths: list[Path],
    time_key: str,
    qdm_lon: np.ndarray,
    qdm_lat: np.ndarray,
    cache_dir: Path,
) -> dict[int, int]:
    n_grid = len(qdm_lon)
    target = f"LST_{time_key}"
    stores = {
        month: np.memmap(
            _legacy_qpred_cache_path(cache_dir, time_key, month),
            dtype="float32",
            mode="w+",
            shape=(MAX_HIST_DAYS_PER_CALENDAR_MONTH, n_grid),
        )
        for month in range(1, 13)
    }
    counts = {month: 0 for month in range(1, 13)}
    seen_dates: set[pd.Timestamp] = set()

    try:
        for input_path in input_paths:
            pf = pq.ParquetFile(input_path)
            required = ["date", "lon", "lat", target]
            missing = [c for c in required if c not in pf.schema_arrow.names]
            if missing:
                raise RuntimeError(f"{input_path}: missing columns {missing}")
            for rg in range(pf.num_row_groups):
                frame = pf.read_row_group(rg, columns=required).to_pandas()
                for date_value, day in split_row_group_by_date(frame):
                    if not (HIST_START_YEAR <= date_value.year <= HIST_END_YEAR):
                        continue
                    if date_value in seen_dates:
                        raise RuntimeError(f"Duplicate historical Pre-QDM date {date_value.date()}")
                    seen_dates.add(date_value)
                    assert_day_grid(
                        day,
                        qdm_lon,
                        qdm_lat,
                        f"legacy q_pred_hist {time_key} {date_value.date()}",
                    )
                    month = int(date_value.month)
                    row = counts[month]
                    if row >= MAX_HIST_DAYS_PER_CALENDAR_MONTH:
                        raise RuntimeError(f"Too many historical days for month {month}")
                    stores[month][row, :] = day[target].to_numpy(dtype="float32")
                    counts[month] += 1
                del frame
            gc.collect()
    finally:
        for mmap in stores.values():
            mmap.flush()
        stores.clear()
        gc.collect()

    years = {d.year for d in seen_dates}
    missing_years = sorted(set(range(HIST_START_YEAR, HIST_END_YEAR + 1)) - years)
    if missing_years:
        raise RuntimeError(f"Historical Pre-QDM input missing years: {missing_years}")
    return counts


def ensure_legacy_q_pred_hist(
    args: argparse.Namespace,
    gcm: str,
    qdm_lon: np.ndarray,
    qdm_lat: np.ndarray,
) -> None:
    missing_times: list[str] = []
    for time_key in TIMES:
        with Dataset(qdm_path(args.qdm_dir, gcm, time_key), "r") as ds:
            if not _q_pred_hist_ready(ds):
                missing_times.append(time_key)
    if not missing_times:
        return

    input_paths = [
        historical_pre_qdm_path(args.output_root, gcm, year)
        for year in range(HIST_START_YEAR, HIST_END_YEAR + 1)
    ]
    missing_inputs = [str(path) for path in input_paths if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing historical Pre-QDM files: {missing_inputs}")

    cache_dir = args.cache_root / gcm / "historical" / "_legacy_qpred_upgrade"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"{gcm}: legacy Code4-01 QDM lacks q_pred_hist for {missing_times}; "
        "reconstructing it from 2003-2014 Pre-QDM without changing historical_bias"
    )

    try:
        for time_key in missing_times:
            counts = _build_legacy_qpred_month_cache(
                input_paths, time_key, qdm_lon, qdm_lat, cache_dir
            )
            month_maps = {
                month: np.memmap(
                    _legacy_qpred_cache_path(cache_dir, time_key, month),
                    dtype="float32",
                    mode="r",
                    shape=(MAX_HIST_DAYS_PER_CALENDAR_MONTH, len(qdm_lon)),
                )
                for month in range(1, 13)
            }
            qdm_file = qdm_path(args.qdm_dir, gcm, time_key)
            try:
                with Dataset(qdm_file, "r+") as ds:
                    if "q_pred_hist" not in ds.variables:
                        q_pred_var = ds.createVariable(
                            "q_pred_hist",
                            "f4",
                            ("month", "quantile", "grid"),
                            zlib=True,
                            complevel=4,
                            chunksizes=(1, len(QUANTILES), min(4096, len(qdm_lon))),
                            fill_value=np.float32(np.nan),
                        )
                    else:
                        q_pred_var = ds.variables["q_pred_hist"]

                    for target_month in progress(
                        range(1, 13),
                        desc=f"rebuild q_pred_hist {gcm} {time_key}",
                        unit="month",
                    ):
                        months = month_window(target_month)
                        for g0 in range(0, len(qdm_lon), args.grid_chunk):
                            g1 = min(g0 + args.grid_chunk, len(qdm_lon))
                            pred_block = np.concatenate(
                                [
                                    np.asarray(
                                        month_maps[m][:counts[m], g0:g1],
                                        dtype="float32",
                                    )
                                    for m in months
                                ],
                                axis=0,
                            )
                            valid = (
                                np.asarray(
                                    ds.variables["qdm_valid"][target_month - 1, g0:g1],
                                    dtype="int8",
                                )
                                == 1
                            )
                            q_pred = np.full(
                                (len(QUANTILES), g1 - g0),
                                np.nan,
                                dtype="float32",
                            )
                            if np.any(valid):
                                q_pred[:, valid] = _nanquantile_linear(
                                    pred_block[:, valid],
                                    QUANTILES,
                                    axis=0,
                                ).astype("float32")
                            q_pred_var[target_month - 1, :, g0:g1] = q_pred
                            del pred_block, valid, q_pred
                        ds.sync()

                    ds.q_pred_hist_source = (
                        "reconstructed by Formal Code4 from saved 2003-2014 Pre-QDM"
                    )
                    ds.q_pred_hist_complete = 1
                    ds.sync()
            finally:
                month_maps.clear()
                gc.collect()
                for month in range(1, 13):
                    path = _legacy_qpred_cache_path(cache_dir, time_key, month)
                    if path.exists():
                        path.unlink()
            log(f"{gcm} {time_key}: q_pred_hist compatibility upgrade complete")
    finally:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)


def percentile_from_historical_quantiles(
    target_pred: np.ndarray,
    q_pred_grid_quantile: np.ndarray,
) -> np.ndarray:


    x = np.asarray(target_pred, dtype="float64")
    qg = np.asarray(q_pred_grid_quantile, dtype="float64")
    if qg.ndim != 2 or qg.shape[1] != len(QUANTILES):
        raise RuntimeError("Unexpected q_pred_hist quantile shape")

    count_le = np.sum(qg[None, :, :] <= x[:, :, None], axis=2)
    below = x <= qg[None, :, 0]
    above = x >= qg[None, :, -1]

    lo_idx = np.clip(count_le - 1, 0, len(QUANTILES) - 2).astype("int16")
    hi_idx = lo_idx + 1

    q_lo = np.take_along_axis(qg, lo_idx.T, axis=1).T
    q_hi = np.take_along_axis(qg, hi_idx.T, axis=1).T
    p_lo = QUANTILES[lo_idx]
    p_hi = QUANTILES[hi_idx]

    denom = q_hi - q_lo
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (x - q_lo) / denom
    duplicate = (~np.isfinite(frac)) | (denom <= 0)
    frac[duplicate] = 0.5
    frac = np.clip(frac, 0.0, 1.0)

    p = p_lo + frac * (p_hi - p_lo)
    p[below] = float(QUANTILES[0])
    p[above] = float(QUANTILES[-1])
    p[~np.isfinite(x)] = np.nan
    return p


def _write_corrected_historical_month(
    writer: pq.ParquetWriter,
    month_dates: list[pd.Timestamp],
    month_values: dict[str, list[np.ndarray]],
    month: int,
    qdm_handles: dict[str, Dataset],
    lon: np.ndarray,
    lat: np.ndarray,
    grid_chunk: int,
) -> None:
    if not month_dates:
        return
    n_days = len(month_dates)
    n_grid = len(lon)

    corrected_by_time: dict[str, np.ndarray] = {}
    for time_key in TIMES:
        pred = np.stack(month_values[time_key], axis=0).astype("float32", copy=False)
        if pred.shape != (n_days, n_grid):
            raise RuntimeError(
                f"Historical month buffer shape mismatch for {time_key}: {pred.shape}"
            )
        corrected = pred.copy()
        ds = qdm_handles[time_key]

        for g0 in range(0, n_grid, grid_chunk):
            g1 = min(g0 + grid_chunk, n_grid)
            target = pred[:, g0:g1].astype("float64", copy=False)
            hist_valid = np.asarray(
                ds.variables["qdm_valid"][month - 1, g0:g1],
                dtype="int8",
            ) == 1
            q_pred_grid = np.asarray(
                ds.variables["q_pred_hist"][month - 1, :, g0:g1],
                dtype="float32",
            ).T
            bias_grid = np.asarray(
                ds.variables["historical_bias"][month - 1, :, g0:g1],
                dtype="float32",
            ).T.astype("float64", copy=False)

            p = percentile_from_historical_quantiles(target, q_pred_grid)
            delta = interpolation_from_percentile(p, bias_grid)
            applied = (
                hist_valid[None, :]
                & np.isfinite(target)
                & np.isfinite(p)
                & np.isfinite(delta)
            )
            block = target.astype("float32", copy=True)
            block[applied] = (target[applied] + delta[applied]).astype("float32")
            corrected[:, g0:g1] = block

        corrected_by_time[time_key] = corrected

    for day_idx, date_value in enumerate(month_dates):
        arrays = [
            pa.array(
                np.full(
                    n_grid,
                    np.datetime64(date_value.to_datetime64(), "ns"),
                    dtype="datetime64[ns]",
                ),
                type=pa.timestamp("ns"),
            ),
            pa.array(lon, type=pa.float64()),
            pa.array(lat, type=pa.float64()),
        ]
        for time_key in TIMES:
            values = corrected_by_time[time_key][day_idx]
            if not np.all(np.isfinite(values)):
                bad = int(np.sum(~np.isfinite(values)))
                raise RuntimeError(
                    f"{date_value.date()} {time_key}: corrected historical field "
                    f"contains {bad:,} non-finite cells"
                )
            arrays.append(pa.array(values, type=pa.float32()))

        schema = pa.schema(
            [
                ("date", pa.timestamp("ns")),
                ("lon", pa.float64()),
                ("lat", pa.float64()),
                *[(f"LST_{time_key}", pa.float32()) for time_key in TIMES],
            ]
        )
        writer.write_table(pa.Table.from_arrays(arrays, schema=schema), row_group_size=n_grid)


def correct_historical_year(
    input_path: Path,
    output_path: Path,
    qdm_handles: dict[str, Dataset],
    qdm_lon: np.ndarray,
    qdm_lat: np.ndarray,
    grid_chunk: int,
    gcm: str = "",
    manifest_root: Path | None = None,
    overwrite: bool = False,
) -> None:
    pf = pq.ParquetFile(input_path)
    required = ["date", "lon", "lat", *[f"LST_{t}" for t in TIMES]]
    missing = [c for c in required if c not in pf.schema_arrow.names]
    if missing:
        raise RuntimeError(f"{input_path}: missing columns {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("date", pa.timestamp("ns")),
            ("lon", pa.float64()),
            ("lat", pa.float64()),
            *[(f"LST_{time_key}", pa.float32()) for time_key in TIMES],
        ]
    )
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    is_netcdf = output_path.suffix.lower() == ".nc"
    if is_netcdf:
        writer = NetCDFTableSink(
            output_path, gcm=gcm, year=int(output_path.stem.split("_")[-1]),
            lon=qdm_lon, lat=qdm_lat, manifest_root=manifest_root,
            source_files=[input_path.name], overwrite=overwrite,
        )
    else:
        if tmp_path.exists():
            tmp_path.unlink()
        writer = pq.ParquetWriter(tmp_path, schema=schema, compression="zstd")

    current_month: int | None = None
    month_dates: list[pd.Timestamp] = []
    month_values: dict[str, list[np.ndarray]] = {t: [] for t in TIMES}
    seen_dates: set[pd.Timestamp] = set()

    def flush_month() -> None:
        nonlocal month_dates, month_values, current_month
        if current_month is None or not month_dates:
            return
        _write_corrected_historical_month(
            writer=writer,
            month_dates=month_dates,
            month_values=month_values,
            month=current_month,
            qdm_handles=qdm_handles,
            lon=qdm_lon,
            lat=qdm_lat,
            grid_chunk=grid_chunk,
        )
        month_dates = []
        month_values = {t: [] for t in TIMES}

    try:
        for rg in range(pf.num_row_groups):
            frame = pf.read_row_group(rg, columns=required).to_pandas()
            for date_value, day in split_row_group_by_date(frame):
                if date_value in seen_dates:
                    raise RuntimeError(f"Duplicate historical Pre-QDM date {date_value.date()}")
                seen_dates.add(date_value)
                assert_day_grid(
                    day,
                    qdm_lon,
                    qdm_lat,
                    f"Historical Pre-QDM {date_value.date()} {input_path.name}:row_group={rg + 1}",
                )
                month = int(date_value.month)
                if current_month is None:
                    current_month = month
                elif month != current_month:
                    flush_month()
                    current_month = month
                month_dates.append(date_value)
                for time_key in TIMES:
                    month_values[time_key].append(
                        day[f"LST_{time_key}"].to_numpy(dtype="float32")
                    )
            del frame
        flush_month()
        if not seen_dates:
            raise RuntimeError(f"No historical Pre-QDM dates found in {input_path}")
        writer.close()
    except BaseException:
        if is_netcdf:
            writer.abort()
        else:
            writer.close()
            tmp_path.unlink(missing_ok=True)
        raise

    if not is_netcdf:
        tmp_path.replace(output_path)
    log(f"saved corrected historical CMB-LST: {output_path}")


def process_historical_gcm(
    args: argparse.Namespace,
    gcm: str,
) -> None:
    qdm_handles, qdm_lon, qdm_lat = open_and_validate_qdm_set(args.qdm_dir, gcm)
    if not args.dry_run and any(not _q_pred_hist_ready(ds) for ds in qdm_handles.values()):
        close_qdm_set(qdm_handles)
        ensure_legacy_q_pred_hist(args, gcm, qdm_lon, qdm_lat)
        qdm_handles, qdm_lon, qdm_lat = open_and_validate_qdm_set(args.qdm_dir, gcm)
    try:
        input_dir = historical_pre_qdm_dir(args.output_root, gcm)
        output_dir = (published_year_path(args.release_root, gcm, "historical", HIST_START_YEAR).parent
                      if args.final_format == "netcdf" else historical_output_dir(args.output_root, gcm))
        print(
            f"Historical correction {gcm}: {input_dir} -> {output_dir}",
            flush=True,
        )
        for year in range(HIST_START_YEAR, HIST_END_YEAR + 1):
            input_path = historical_pre_qdm_path(args.output_root, gcm, year)
            output_path = final_historical_output_path(args, gcm, year)
            if not input_path.exists():
                raise FileNotFoundError(input_path)
            if output_path.exists() and not args.overwrite_historical:
                if valid_final_output(output_path, len(qdm_lon), year):
                    log(f"skip existing valid historical CMB-LST: {output_path}")
                    continue
                raise RuntimeError(
                    f"Existing historical output is incomplete: {output_path}. "
                    "Use --overwrite-historical to replace it."
                )
            if args.dry_run:
                continue
            correct_historical_year(
                input_path=input_path,
                output_path=output_path,
                qdm_handles=qdm_handles,
                qdm_lon=qdm_lon,
                qdm_lat=qdm_lat,
                grid_chunk=args.grid_chunk,
                gcm=gcm,
                manifest_root=args.manifest_root,
                overwrite=args.overwrite_historical,
            )
    finally:
        close_qdm_set(qdm_handles)


def merge_diagnostic_csv(path: Path, new_rows: list[dict[str, object]], key_columns: list[str]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if path.exists():
        old_df = pd.read_csv(path, dtype={"time": str})
    else:
        old_df = pd.DataFrame()

    if old_df.empty:
        merged = new_df.copy()
    elif new_df.empty:
        merged = old_df.copy()
    else:
        old_key = old_df[key_columns].astype(str).agg("|".join, axis=1)
        new_key = set(new_df[key_columns].astype(str).agg("|".join, axis=1))
        old_df = old_df.loc[~old_key.isin(new_key)]
        merged = pd.concat([old_df, new_df], ignore_index=True)

    if not merged.empty:
        merged = merged.sort_values(key_columns).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def write_trend_diagnostics(annual_df: pd.DataFrame, path: Path) -> None:
    rows = []
    for time_key in TIMES:
        frame = annual_df[annual_df["time"] == time_key].sort_values("year")
        if len(frame) < 2:
            continue
        years = frame["year"].to_numpy(dtype="float64")
        pred = frame["annual_mean_LST_pred"].to_numpy(dtype="float64")
        corr = frame["annual_mean_LST_corr"].to_numpy(dtype="float64")
        valid = np.isfinite(years) & np.isfinite(pred) & np.isfinite(corr)
        if np.sum(valid) < 2:
            continue
        years = years[valid]
        pred = pred[valid]
        corr = corr[valid]
        pred_slope = float(np.polyfit(years, pred, 1)[0] * 10.0)
        corr_slope = float(np.polyfit(years, corr, 1)[0] * 10.0)
        correlation = float(np.corrcoef(pred, corr)[0, 1]) if len(pred) > 1 else np.nan
        rows.append(
            {
                "time": time_key,
                "first_year": int(years.min()),
                "last_year": int(years.max()),
                "n_years": len(years),
                "pred_trend_C_per_decade": pred_slope,
                "corr_trend_C_per_decade": corr_slope,
                "trend_difference_C_per_decade": corr_slope - pred_slope,
                "annual_series_correlation": correlation,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_run_config(
    diag_dir: Path,
    gcm: str,
    ssp: str,
    sources: list[Path],
    model_dir: Path,
    qdm_dir: Path,
    output_root: Path,
    final_format: str,
    release_root: Path,
    requested_start: int,
    requested_end: int,
    actual_first: int,
    actual_last: int,
    target_start: int,
    target_end: int,
    feature_map: dict[str, str],
    model_features: list[str],
) -> None:
    config = {
        "code": "Formal-Code4",
        "gcm": gcm,
        "ssp": ssp,
        "ssp_label": SSP_LABELS[ssp],
        "base_model": "full 2003-2014 LightGBM q50",
        "base_model_training_period": "2003-2014",
        "historical_quantile_reference": "GCM-specific 2003-2014 historical predicted LST and TRIMS LST",
        "historical_bias_definition": "Q_TRIMS_hist(p) - Q_GCMpred_hist(p)",
        "future_percentile_definition": "GCM historical pre-correction LST distribution (2003-2014)",
        "historical_seasonal_window": "centered 3 calendar months pooled across 2003-2014",
        "minimum_contiguous_prepared_future_years": FUTURE_WINDOW_YEARS,
        "quantiles": "P1-P99, 1% step",
        "tail": "fixed historical P1/P99 additive bias; projected LST is not clipped",
        "fallback": "LST_corr = LST_pred where historical correction is unavailable",
        "future_TRIMS_used": False,
        "HiCPC_input_second_bias_correction": False,
        "times": TIMES,
        "model_features": model_features,
        "hicpc_to_model_feature_map": feature_map,
        "requested_output_period": f"{requested_start}-{requested_end}",
        "actual_HiCPC_complete_future_period": f"{actual_first}-{actual_last}",
        "actual_output_period_this_run": f"{target_start}-{target_end}",
        "model_dir": str(model_dir),
        "qdm_dir": str(qdm_dir),
        "feature_sources": [str(p) for p in sources],
        "output_root": str(output_root),
        "release_root": str(release_root),
        "final_format": final_format,
        "output_directory": (str(published_year_path(release_root, gcm, SSP_LABELS[ssp], target_start).parent)
                             if final_format == "netcdf" else str(scenario_output_dir(output_root, gcm, ssp))),
        "output_filename_rule": (f"{gcm}_{ssp}_<YEAR>.nc" if final_format == "netcdf"
                                 else f"{gcm}_{SSP_LABELS[ssp]}_<YEAR>.parquet"),
        "output_columns": (["time", "lon", "lat", *[f"LST_{t}" for t in TIMES]]
                           if final_format == "netcdf" else FINAL_COLUMNS),
        "created_utc": utc_now(),
    }
    (diag_dir / "code4_02_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def dry_run_first_future_day(
    sources: list[Path],
    feature_map: dict[str, str],
    qdm_lon: np.ndarray,
    qdm_lat: np.ndarray,
) -> tuple[pd.Timestamp, Path, int]:
    columns = source_columns_for_prediction(feature_map)
    iterator = iter_future_days(sources, columns, FUTURE_START_YEAR, REQUESTED_END_YEAR_DEFAULT)
    try:
        date_value, day, path, rg = next(iter(iterator))
    except StopIteration as exc:
        raise RuntimeError("No HiCPC future record found from 2015 onward") from exc
    assert_day_grid(day, qdm_lon, qdm_lat, f"dry-run {path.name}:row_group={rg + 1}")
    return date_value, path, rg


def process_scenario(
    args: argparse.Namespace,
    gcm: str,
    ssp: str,
    models: dict[str, object],
    model_features: list[str],
    model_info: dict[str, object],
) -> None:
    del model_info


    release_end = min(args.end_year, 2099)
    if (args.final_format == "netcdf" and not args.overwrite_output
            and not args.dry_run
            and FUTURE_START_YEAR <= args.start_year <= release_end
            and (args.end_year <= 2099 or
                 (args.end_year == 2100 and not args.require_end_year))):
        completed_release = all(
            validate_release_year(
                final_scenario_output_path(args, gcm, ssp, year),
                EXPECTED_ROWS_PER_DAY, year)
            for year in range(args.start_year, release_end + 1)
        )
        if completed_release:
            log(
                f"skip complete released NetCDF for {gcm} {SSP_LABELS[ssp]} "
                f"{args.start_year}-{release_end}; HiCPC intermediate input not required"
            )
            return

    sources = resolve_hicpc_sources(args.hicpc_feature, args.hicpc_root, gcm, ssp)
    first_pf = pq.ParquetFile(sources[0])
    feature_map = resolve_feature_map(first_pf.schema_arrow.names, model_features)

    qdm_handles, qdm_lon, qdm_lat = open_and_validate_qdm_set(args.qdm_dir, gcm)
    try:
        date_value, source_path, rg = dry_run_first_future_day(
            sources, feature_map, qdm_lon, qdm_lat
        )
        output_example = final_scenario_output_path(args, gcm, ssp, args.start_year)
        log(
            f"Resolved {gcm} {ssp} ({SSP_LABELS[ssp]}): first future day={date_value.date()}, "
            f"source={source_path.name}:row_group={rg + 1}, output example={output_example}"
        )
        if args.dry_run:
            return


        output_dir = final_scenario_output_dir(args, gcm, ssp)


        actual_input_bounds = source_year_bounds(sources)
        if actual_input_bounds is None:

            scan_start = args.start_year
            scan_end = args.end_year
        else:
            input_first, input_last = actual_input_bounds

            scan_start = max(args.start_year, input_first)
            scan_end = min(args.end_year, input_last)
            if scan_start > input_last or scan_end < input_first:
                raise RuntimeError(
                    f"Requested output years {args.start_year}-{args.end_year} do not overlap "
                    f"with input data coverage {input_first}-{input_last}."
                )
            log(
                f"Input feature parquet coverage: {input_first}-{input_last}; "
                f"will check output years {scan_start}-{scan_end}"
            )

        existing_valid_years = existing_valid_output_years(
            args, gcm, ssp, scan_start, scan_end, len(qdm_lon)
        )
        requested_years = set(range(scan_start, scan_end + 1))
        if existing_valid_years:
            log(
                f"Found {len(existing_valid_years)} existing valid final parquet years for "
                f"{gcm} {SSP_LABELS[ssp]}: {min(existing_valid_years)}-{max(existing_valid_years)}"
            )


        if not args.overwrite_output and requested_years.issubset(set(existing_valid_years)):
            log(
                f"All requested output years already exist for {gcm} {SSP_LABELS[ssp]}; "
                "skipping prediction cache build and QDM processing"
            )
            cache_dir = scenario_cache_dir(args.cache_root, gcm, ssp)
            if cache_dir.exists() and not args.keep_prediction_cache:
                shutil.rmtree(cache_dir)
                log(f"deleted unused prediction cache: {cache_dir}")
            log(
                f"Formal Code4 future scenario complete: {gcm} {SSP_LABELS[ssp]} -> "
                f"{output_dir}"
            )
            return

        if args.rebuild_prediction_cache:
            cache_dir = scenario_cache_dir(args.cache_root, gcm, ssp)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)

        cache_dir = scenario_cache_dir(args.cache_root, gcm, ssp)
        cache_dir.mkdir(parents=True, exist_ok=True)
        marker = build_future_prediction_cache(
            sources=sources,
            models=models,
            model_features=model_features,
            feature_map=feature_map,
            qdm_lon=qdm_lon,
            qdm_lat=qdm_lat,
            cache_dir=cache_dir,
            gcm=gcm,
            ssp=ssp,
            batch_days_n=args.batch_days,
            model_dir=args.model_dir,
            check_grid_every_day=args.check_grid_every_day,
        )

        complete_years = list(map(int, marker["complete_years"]))
        actual_first = min(complete_years)
        actual_last = max(complete_years)

        if args.start_year < actual_first:
            raise RuntimeError(
                f"Requested start year {args.start_year} precedes actual future coverage {actual_first}."
            )
        if args.start_year > actual_last:
            raise RuntimeError(
                f"Requested start year {args.start_year} exceeds actual future coverage ending {actual_last}."
            )

        target_start = max(args.start_year, actual_first)
        if args.end_year > actual_last:
            message = (
                f"Requested end year={args.end_year}, but complete HiCPC feature coverage ends at {actual_last}. "
                f"Input coverage does not include {actual_last + 1}-{args.end_year}; actual output ends at {actual_last}."
            )
            if args.require_end_year:
                raise RuntimeError(message)
            print(f"WARNING: {message}", flush=True)
            target_end = actual_last
        else:
            target_end = args.end_year

        if target_end < target_start:
            raise RuntimeError(f"Invalid actual output period: {target_start}-{target_end}")


        year_infos = {
            year: year_date_info(cache_dir, year)
            for year in range(actual_first, actual_last + 1)
        }

        diag_dir = scenario_diag_dir(args.diagnostics_root, gcm, ssp)
        diag_dir.mkdir(parents=True, exist_ok=True)
        write_run_config(
            diag_dir,
            gcm,
            ssp,
            sources,
            args.model_dir,
            args.qdm_dir,
            args.output_root,
            args.final_format,
            args.release_root,
            args.start_year,
            args.end_year,
            actual_first,
            actual_last,
            target_start,
            target_end,
            feature_map,
            model_features,
        )

        annual_rows: list[dict[str, object]] = []
        coverage_rows: list[dict[str, object]] = []
        years_to_process: list[int] = []

        for year in range(target_start, target_end + 1):
            output_path = final_scenario_output_path(args, gcm, ssp, year)
            if output_path.exists() and not args.overwrite_output:
                if valid_final_output(output_path, len(qdm_lon), year):
                    log(f"skip existing valid final product: {output_path}")
                    continue
                raise RuntimeError(
                    f"Existing output is incomplete or has the wrong schema: {output_path}. "
                    "Use --overwrite-output to replace it."
                )

            years_to_process.append(year)

        if not years_to_process:
            log(f"No years need processing for {gcm} {SSP_LABELS[ssp]}; existing outputs are already complete")
            if not args.keep_prediction_cache:
                shutil.rmtree(cache_dir)
                log(f"deleted large prediction cache after successful scenario: {cache_dir}")
            log(
                f"Formal Code4 future scenario complete: {gcm} {SSP_LABELS[ssp]} -> "
                f"{final_scenario_output_dir(args, gcm, ssp)}"
            )
            return

        if args.qdm_year_workers > 1:
            _code4_parallel_qdm_batch(
                args,gcm,ssp,cache_dir,years_to_process,actual_first,actual_last,diag_dir,
            )
            years_to_process=[]

        for year in years_to_process:
            output_path = final_scenario_output_path(args, gcm, ssp, year)

            corrected_cache, year_annual, year_coverage = apply_qdm_to_target_year(
                cache_dir=cache_dir,
                qdm_handles=qdm_handles,
                year_infos=year_infos,
                target_year=year,
                first_future_year=actual_first,
                last_future_year=actual_last,
                n_grid=len(qdm_lon),
                grid_chunk=args.grid_chunk,
            )

            if args.final_format == "netcdf":
                write_future_release_year(
                    output=output_path,
                    corrected_cache=corrected_cache,
                    year_info=year_infos[year],
                    year=year,
                    lon=qdm_lon,
                    lat=qdm_lat,
                    gcm=gcm,
                    scenario=SSP_LABELS[ssp],
                    manifest_root=args.manifest_root,
                    overwrite=args.overwrite_output,
                )
            else:
                write_final_year_parquet(
                    output_path=output_path,
                    corrected_cache=corrected_cache,
                    cache_dir=cache_dir,
                    year_info=year_infos[year],
                    year=year,
                    lon=qdm_lon,
                    lat=qdm_lat,
                )
            annual_rows.extend(year_annual)
            coverage_rows.extend(year_coverage)

            if corrected_cache.exists():
                corrected_cache.unlink()
            gc.collect()


            annual_df = merge_diagnostic_csv(
                diag_dir / "annual_diagnostics.csv",
                annual_rows,
                key_columns=["year", "time"],
            )
            merge_diagnostic_csv(
                diag_dir / "qdm_coverage_by_month.csv",
                coverage_rows,
                key_columns=["year", "time", "month"],
            )
            write_trend_diagnostics(annual_df, diag_dir / "trend_diagnostics.csv")
            annual_rows = []
            coverage_rows = []


        annual_path = diag_dir / "annual_diagnostics.csv"
        if annual_path.exists():
            annual_df = pd.read_csv(annual_path, dtype={"time": str})
            write_trend_diagnostics(annual_df, diag_dir / "trend_diagnostics.csv")

        success_manifest = {
            "code": "CMOD_LST_generation",
            "method": "historical_quantile_additive",
            "status": "completed",
            "gcm": gcm,
            "ssp": ssp,
            "ssp_label": SSP_LABELS[ssp],
            "actual_input_period": f"{actual_first}-{actual_last}",
            "requested_output_period": f"{args.start_year}-{args.end_year}",
            "actual_output_period": f"{target_start}-{target_end}",
            "output_directory": str(final_scenario_output_dir(args, gcm, ssp)),
            "final_format": args.final_format,
            "completed_utc": utc_now(),
        }
        (diag_dir / "completed.json").write_text(
            json.dumps(success_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if not args.keep_prediction_cache:
            shutil.rmtree(cache_dir)
            log(f"deleted large prediction cache after successful scenario: {cache_dir}")
        log(
            f"Formal Code4 future scenario complete: {gcm} {SSP_LABELS[ssp]} -> "
            f"{final_scenario_output_dir(args, gcm, ssp)}"
        )
    finally:
        close_qdm_set(qdm_handles)


def main() -> None:
    args = parse_args()
    if args.grid_chunk <= 0:
        raise ValueError("--grid-chunk must be positive")
    if not 1<=args.qdm_year_workers<=32:
        raise ValueError("--qdm-year-workers must be 1..32")
    if not 1<=args.qdm_threads_per_year<=16:
        raise ValueError("--qdm-threads-per-year must be 1..16")
    if args.qdm_parallel_grid_chunk<=0:
        raise ValueError("--qdm-parallel-grid-chunk must be positive")

    if args.batch_days <= 0:
        raise ValueError("--batch-days must be positive")
    if args.start_year < FUTURE_START_YEAR:
        raise ValueError(f"--start-year cannot be earlier than {FUTURE_START_YEAR}")
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be >= --start-year")
    if args.skip_historical and args.skip_future:
        raise ValueError("Cannot use --skip-historical and --skip-future together")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.diagnostics_root.mkdir(parents=True, exist_ok=True)

    models, model_features, model_info = load_code1_models(args.model_dir)

    if args.gcm.strip().lower() == "all":
        gcms = discover_gcms(args.qdm_dir)
    else:
        gcms = [args.gcm.strip()]

    if args.ssp.strip().lower() == "all":
        ssps = list(SSP_LABELS.keys())
    else:
        ssps = [normalize_ssp(args.ssp)]

    if args.hicpc_feature is not None and (len(gcms) != 1 or len(ssps) != 1):
        raise ValueError(
            "--hicpc-feature can only be used when exactly one GCM and one SSP are selected"
        )

    print("=" * 100, flush=True)
    print("Formal Code4: historical + future CMB-LST generation", flush=True)
    print(f"GCMs: {gcms}", flush=True)
    print(f"Historical: {'skip' if args.skip_historical else '2003-2014 correction'}", flush=True)
    print(
        f"Future: {'skip' if args.skip_future else [SSP_LABELS[s] for s in ssps]}",
        flush=True,
    )
    print(f"QDM root: {args.qdm_dir}", flush=True)
    print(f"HiCPC future input root: {args.hicpc_root}", flush=True)
    print(f"Internal output root: {args.output_root}", flush=True)
    print(f"Final publication format: {args.final_format}", flush=True)
    if args.final_format == "netcdf":
        print(f"Final release root: {args.release_root}", flush=True)
    print("=" * 100, flush=True)

    if not args.skip_historical:
        for gcm in gcms:
            process_historical_gcm(args, gcm)

    if args.skip_future:
        log("Formal Code4 historical processing finished")
        return

    failures: list[dict[str, str]] = []
    for gcm in gcms:
        for ssp in ssps:
            try:
                process_scenario(
                    args,
                    gcm,
                    ssp,
                    models,
                    model_features,
                    model_info,
                )
            except (FileNotFoundError, RuntimeError) as exc:
                if args.skip_missing_scenarios and len(gcms) * len(ssps) > 1:
                    message = f"{type(exc).__name__}: {exc}"
                    failures.append(
                        {"gcm": gcm, "ssp": ssp, "error": message}
                    )
                    print(
                        f"WARNING: skip scenario {gcm} {SSP_LABELS[ssp]} because: {message}",
                        flush=True,
                    )
                    continue
                raise

    if failures:
        failure_path = (
            args.diagnostics_root
            / "code4_skipped_scenarios.csv"
        )
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"Skipped-scenario report: {failure_path}", flush=True)

    log("Formal Code4 all requested historical/future processing finished")


if __name__ == "__main__":
    main()
