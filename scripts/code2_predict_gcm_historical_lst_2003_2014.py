from __future__ import annotations

import argparse
import gc
import json
import resource
import tempfile
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_HICPC_INPUT_ROOT = Path(
    "data/hicpc"
)
DEFAULT_MODEL_DIR = Path(
    "data/models"
)
DEFAULT_REFERENCE_PARQUET = Path(
    "data/training/2003.parquet"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/work"
)

TIMES = ["0130", "1030", "1330", "2230"]
HIST_START = pd.Timestamp("2003-01-01")
HIST_END = pd.Timestamp("2014-12-31")
HIST_YEARS = list(range(2003, 2015))
EXPECTED_ROWS_PER_DAY = 97371

FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "date_sin": ("date_sin",),
    "date_cos": ("date_cos",),
    "lon": ("lon",),
    "lat": ("lat",),
    "tasmax_cmfd_daily": (
        "tasmax_cmfd_daily", "tasmax", "tasmax_hicpc_daily", "tasmax_daily"
    ),
    "tasmin_cmfd_daily": (
        "tasmin_cmfd_daily", "tasmin", "tasmin_hicpc_daily", "tasmin_daily"
    ),
    "tas_cmfd_daily": (
        "tas_cmfd_daily", "tas", "tas_hicpc_daily", "tas_daily"
    ),
    "pr": ("pr", "pr_hicpc_daily", "pr_daily"),
    "DEM": ("DEM", "dem"),
    "slope": ("slope",),
    "aspect": ("aspect",),
}


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
            "Formal Code2: predict 2003-2014 Pre-QDM historical LST once per GCM "
            "from prepared HiCPC historical predictor parquet files."
        )
    )
    parser.add_argument(
        "--gcm",
        nargs="+",
        default=None,
        help=(
            "One or more GCM names. If omitted, all GCM directories containing "
            "a historical folder under --hicpc-input-root are processed."
        ),
    )
    parser.add_argument("--hicpc-input-root", type=Path, default=DEFAULT_HICPC_INPUT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--reference-parquet", type=Path, default=DEFAULT_REFERENCE_PARQUET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--times", nargs="+", choices=TIMES, default=TIMES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_gcms(input_root: Path, requested: list[str] | None) -> list[str]:
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if requested:
        gcms = requested
    else:
        gcms = sorted(
            p.name
            for p in input_root.iterdir()
            if p.is_dir() and (p / "historical").is_dir()
        )
    if not gcms:
        raise RuntimeError(f"No GCM historical input directories found under {input_root}")
    missing = [
        gcm for gcm in gcms if not (input_root / gcm / "historical").is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"Missing historical input folder for GCM(s): {missing}")
    return gcms


def historical_sources(input_root: Path, gcm: str) -> list[Path]:
    hist_dir = input_root / gcm / "historical"
    paths = sorted(hist_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No historical parquet files found for GCM={gcm}: {hist_dir}"
        )
    return paths


def load_models(
    model_dir: Path,
    times: list[str],
) -> tuple[dict[str, object], list[str], dict[str, object]]:
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
            "Formal Code2 requires the full LightGBM models trained on 2003-2014; "
            f"model_info.json reports training_period={training_period!r}."
        )

    features = list(info.get("feature_names", []))
    if not features:
        raise RuntimeError("model_info.json does not contain feature_names")

    models: dict[str, object] = {}
    for time_key in times:
        path = model_dir / f"q50_{time_key}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "model" not in payload:
            raise RuntimeError(f"Invalid formal Code1 model payload: {path}")
        payload_features = list(payload.get("features", features))
        if payload_features != features:
            raise RuntimeError(
                f"Feature order mismatch between {path.name} and model_info.json"
            )
        if str(payload.get("training_period", training_period)) != "2003-2014":
            raise RuntimeError(f"{path.name} is not the full 2003-2014 production model")
        models[time_key] = payload["model"]

    return models, features, info


def resolve_feature_map(
    schema_names: list[str],
    model_features: list[str],
) -> dict[str, str]:
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


def read_reference_grid(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    pf = pq.ParquetFile(path)
    if pf.num_row_groups < 1:
        raise RuntimeError(f"Empty parquet: {path}")
    frame = pf.read_row_group(0, columns=["lon", "lat"]).to_pandas()
    if len(frame) != EXPECTED_ROWS_PER_DAY:
        raise RuntimeError(
            f"Reference first day expected {EXPECTED_ROWS_PER_DAY:,} grid rows, "
            f"got {len(frame):,}"
        )
    frame["lon"] = frame["lon"].round(6)
    frame["lat"] = frame["lat"].round(6)
    if frame.duplicated(["lon", "lat"]).any():
        raise RuntimeError("Duplicate lon/lat in historical reference grid")
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
            f"{context}: lon/lat grid order differs from the Code1/TRIMS target grid. "
            "Formal Code2 does not silently reorder or nearest-neighbour remap."
        )


def split_row_group_by_date(
    frame: pd.DataFrame,
) -> Iterable[tuple[pd.Timestamp, pd.DataFrame]]:
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.copy()
    frame["__date_norm"] = dates
    for date_value, group in frame.groupby("__date_norm", sort=False):
        yield pd.Timestamp(date_value), group.drop(columns="__date_norm")


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
        raise RuntimeError(
            f"HiCPC model input contains {int(bad.sum()):,} rows with NaN features"
        )
    return out


def inspect_sources_and_feature_map(
    sources: list[Path],
    model_features: list[str],
) -> dict[str, str]:
    first_schema = pq.ParquetFile(sources[0]).schema_arrow.names
    feature_map = resolve_feature_map(first_schema, model_features)
    required_columns = source_columns_for_prediction(feature_map)
    for path in sources:
        names = pq.ParquetFile(path).schema_arrow.names
        missing = [c for c in required_columns if c not in names]
        if missing:
            raise RuntimeError(
                f"HiCPC historical partition {path} missing columns: {missing}"
            )
    return feature_map


def collect_days(
    sources: list[Path],
    columns: list[str],
) -> Iterable[tuple[pd.Timestamp, pd.DataFrame, str]]:
    for path in sources:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            frame = pf.read_row_group(rg, columns=columns).to_pandas()
            for date_value, day in split_row_group_by_date(frame):
                if HIST_START <= date_value <= HIST_END:
                    yield (
                        date_value,
                        day,
                        f"{path.name}:row_group={rg + 1}",
                    )
            del frame


def output_path(output_root: Path, gcm: str, year: int) -> Path:
    return (
        output_root
        / gcm
        / "02_历史时期"
        / "Pre-QDM"
        / f"{gcm}_historical_Pre-QDM_{year}.parquet"
    )


def validate_existing_output(path: Path, year: int, times: list[str]) -> bool:
    if not path.exists():
        return False
    pf = pq.ParquetFile(path)
    required = {
        "date", "lon", "lat", *[f"LST_{time_key}" for time_key in times]
    }
    if not required.issubset(pf.schema_arrow.names):
        return False
    years = set()
    for rg in range(pf.num_row_groups):
        values = pf.read_row_group(rg, columns=["date"]).to_pandas()["date"]
        unique_years = pd.to_datetime(values, errors="raise").dt.year.unique()
        years.update(int(v) for v in unique_years)
    return years == {year}


def predict_one_gcm(
    gcm: str,
    sources: list[Path],
    models: dict[str, object],
    model_features: list[str],
    feature_map: dict[str, str],
    grid: pd.DataFrame,
    output_root: Path,
    times: list[str],
    overwrite: bool,
) -> None:
    columns = source_columns_for_prediction(feature_map)
    seen_dates: set[pd.Timestamp] = set()
    seen_years: set[int] = set()
    writers: dict[int, pq.ParquetWriter] = {}
    temp_paths: dict[int, Path] = {}
    temp_dirs: dict[int, tempfile.TemporaryDirectory] = {}
    skipped_years: set[int] = set()

    out_dir = output_root / gcm / "02_历史时期" / "Pre-QDM"
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in HIST_YEARS:
        final = output_path(output_root, gcm, year)
        if not overwrite and validate_existing_output(final, year, times):
            skipped_years.add(year)
            log(f"{gcm} {year}: reuse existing Pre-QDM output {final}")

    try:
        iterator = collect_days(sources, columns)
        for date_value, day, context in progress(
            iterator,
            desc=f"{gcm} historical LST prediction",
            unit="day",
        ):
            if date_value in seen_dates:
                raise RuntimeError(
                    f"Duplicate HiCPC historical date {date_value.date()} encountered for {gcm}"
                )
            seen_dates.add(date_value)
            year = int(date_value.year)
            seen_years.add(year)

            if year in skipped_years:
                continue

            assert_grid_order(
                day, grid, f"HiCPC {gcm} {date_value.date()} {context}"
            )
            x = build_model_frame(day, model_features, feature_map)

            out = pd.DataFrame(
                {
                    "date": [date_value.strftime("%Y-%m-%d")] * len(day),
                    "lon": day["lon"].to_numpy(dtype="float64").round(6),
                    "lat": day["lat"].to_numpy(dtype="float64").round(6),
                }
            )

            for time_key in times:
                pred = np.asarray(
                    models[time_key].predict(x, num_threads=512),
                    dtype="float32",
                )
                if len(pred) != len(grid):
                    raise RuntimeError(
                        f"Prediction length mismatch for {gcm} {time_key} "
                        f"on {date_value.date()}"
                    )
                out[f"LST_{time_key}"] = pred

            table = pa.Table.from_pandas(out, preserve_index=False)

            if year not in writers:
                final = output_path(output_root, gcm, year)
                tmp = tempfile.TemporaryDirectory(
                    prefix=f"{gcm}_hist_pred_{year}_",
                    dir=out_dir,
                )
                temp_dirs[year] = tmp
                temp_path = Path(tmp.name) / final.name
                temp_paths[year] = temp_path
                writers[year] = pq.ParquetWriter(
                    temp_path,
                    table.schema,
                    compression="snappy",
                )

            writers[year].write_table(table)
            del x, day, out, table
            gc.collect()
    finally:
        for writer in writers.values():
            writer.close()

    missing_years = sorted(set(HIST_YEARS) - seen_years)
    if missing_years:
        for tmp in temp_dirs.values():
            tmp.cleanup()
        raise RuntimeError(
            f"{gcm} historical HiCPC input is missing years: {missing_years}"
        )
    if not seen_dates:
        for tmp in temp_dirs.values():
            tmp.cleanup()
        raise RuntimeError(f"No {gcm} HiCPC records found in 2003-2014")

    for year, temp_path in temp_paths.items():
        final = output_path(output_root, gcm, year)
        if final.exists():
            final.unlink()
        temp_path.replace(final)
        temp_dirs[year].cleanup()
        log(f"{gcm} {year}: wrote Pre-QDM historical LST -> {final}")

    log(
        f"{gcm}: historical prediction completed; days={len(seen_dates):,}, "
        f"years={min(seen_years)}-{max(seen_years)}"
    )


def main() -> None:
    args = parse_args()
    gcms = discover_gcms(args.hicpc_input_root, args.gcm)
    models, features, model_info = load_models(args.model_dir, args.times)
    del model_info
    grid = read_reference_grid(args.reference_parquet)

    print("=" * 100, flush=True)
    print("Formal Code2: GCM historical LST prediction", flush=True)
    print(f"GCMs              : {gcms}", flush=True)
    print(f"HiCPC input root  : {args.hicpc_input_root}", flush=True)
    print(f"Model dir         : {args.model_dir}", flush=True)
    print(f"Reference parquet : {args.reference_parquet}", flush=True)
    print(f"Output root       : {args.output_root}", flush=True)
    print(f"Historical period : {HIST_START.date()} / {HIST_END.date()}", flush=True)
    print(f"Times             : {args.times}", flush=True)
    print(f"Features          : {features}", flush=True)
    print("=" * 100, flush=True)

    for gcm in gcms:
        sources = historical_sources(args.hicpc_input_root, gcm)
        feature_map = inspect_sources_and_feature_map(sources, features)

        print(f"\nGCM={gcm}", flush=True)
        print(
            f"  historical input : {args.hicpc_input_root / gcm / 'historical'}",
            flush=True,
        )
        print(f"  input files      : {len(sources)}", flush=True)
        print(f"  feature map      : {feature_map}", flush=True)
        print(
            f"  output           : "
            f"{args.output_root / gcm / '02_历史时期' / 'Pre-QDM'}",
            flush=True,
        )

        if args.dry_run:
            continue

        predict_one_gcm(
            gcm=gcm,
            sources=sources,
            models=models,
            model_features=features,
            feature_map=feature_map,
            grid=grid,
            output_root=args.output_root,
            times=args.times,
            overwrite=args.overwrite,
        )

    if args.dry_run:
        print("\nDry-run passed. No historical LST was predicted.", flush=True)
    else:
        print("\nFormal Code2 completed.", flush=True)


if __name__ == "__main__":
    main()
