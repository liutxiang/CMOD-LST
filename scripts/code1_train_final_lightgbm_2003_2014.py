from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
from pathlib import Path
from time import time
from typing import Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


INPUT_DIR = Path("data/training")
OUTPUT_DIR = Path(
    "data/models"
)

TRAIN_YEARS = list(range(2003, 2015))

EXPECTED_DAYS = 365
EXPECTED_ROWS_PER_DAY = 97371
EXPECTED_ROWS_PER_YEAR = EXPECTED_DAYS * EXPECTED_ROWS_PER_DAY

RANDOM_STATE = 42
QUANTILE = 0.5
MODEL_TYPE = "lightgbm_quantile"
SCHEME = "without_radiation_final_2003_2014"

TARGETS = {
    "0130": "LST_0130",
    "1030": "LST_1030",
    "1330": "LST_1330",
    "2230": "LST_2230",
}

FEATURES = [
    "date_sin",
    "date_cos",
    "lon",
    "lat",
    "tasmax_cmfd_daily",
    "tasmin_cmfd_daily",
    "tas_cmfd_daily",
    "pr",
    "DEM",
    "slope",
    "aspect",
]

REQUIRED_COLUMNS = ["date", *FEATURES, *TARGETS.values()]

MODEL_PARAMS = {
    "objective": "quantile",
    "alpha": QUANTILE,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "metric": "quantile",

    "n_jobs": min(os.cpu_count() or 1, 512),
    "verbose": -1,
    "seed": RANDOM_STATE,
    "force_col_wise": True,
}

NUM_BOOST_ROUND = 800


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


def log_status(message: str) -> None:
    print(f"{message} | maxrss={memory_gb():.2f} GB", flush=True)


def parquet_path(year: int) -> Path:
    return INPUT_DIR / f"{year}.parquet"


def train_paths() -> list[Path]:
    paths = [parquet_path(year) for year in TRAIN_YEARS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing training parquet files: {missing}")
    return paths


def validate_year(path: Path) -> None:
    year = path.stem
    source = pq.ParquetFile(path)
    available = source.schema_arrow.names

    missing = [column for column in REQUIRED_COLUMNS if column not in available]
    if missing:
        raise RuntimeError(f"{year}: missing required columns: {missing}")

    if source.num_row_groups != EXPECTED_DAYS:
        raise RuntimeError(
            f"{year}: expected {EXPECTED_DAYS} row groups, "
            f"got {source.num_row_groups}"
        )

    if source.metadata.num_rows != EXPECTED_ROWS_PER_YEAR:
        raise RuntimeError(
            f"{year}: expected {EXPECTED_ROWS_PER_YEAR:,} rows, "
            f"got {source.metadata.num_rows:,}"
        )

    column_indexes = {name: idx for idx, name in enumerate(available)}
    null_counts = {column: 0 for column in REQUIRED_COLUMNS}

    for row_group in progress(
        range(source.num_row_groups),
        desc=f"validate {year}",
        unit="day",
    ):
        rg = source.metadata.row_group(row_group)

        if rg.num_rows != EXPECTED_ROWS_PER_DAY:
            raise RuntimeError(
                f"{year}: row group {row_group + 1} expected "
                f"{EXPECTED_ROWS_PER_DAY:,} rows, got {rg.num_rows:,}"
            )

        for column in REQUIRED_COLUMNS:
            stats = rg.column(column_indexes[column]).statistics
            if stats is None or stats.null_count is None:
                values = source.read_row_group(
                    row_group,
                    columns=[column],
                ).to_pandas()[column]
                null_counts[column] += int(values.isna().sum())
            else:
                null_counts[column] += int(stats.null_count)

    remaining_nulls = {
        column: count for column, count in null_counts.items() if count
    }
    if remaining_nulls:
        raise RuntimeError(f"{year}: null values remain: {remaining_nulls}")

    log_status(
        f"validated {year}: "
        f"rows={source.metadata.num_rows:,}, days={source.num_row_groups}"
    )


def validate_inputs(paths: list[Path]) -> None:
    for path in progress(
        paths,
        desc="validate 2003-2014 parquet files",
        unit="year",
    ):
        validate_year(path)


def read_year(path: Path, target: str) -> pd.DataFrame:
    year = path.stem
    columns = [*FEATURES, target]

    log_status(f"reading year={year} target={target}")
    df = pd.read_parquet(path, columns=columns)

    null_rows = int(df.isna().any(axis=1).sum())
    if null_rows:
        raise RuntimeError(
            f"{year}: {null_rows:,} rows contain nulls for target {target}"
        )

    for column in columns:
        df[column] = df[column].astype("float32", copy=False)

    log_status(
        f"loaded year={year} target={target} rows={len(df):,}"
    )
    return df


class MetricAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_err = 0.0
        self.sum_abs_err = 0.0
        self.sum_sq_err = 0.0

    def update(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> None:
        y = np.asarray(y_true, dtype="float64")
        pred = np.asarray(y_pred, dtype="float64")
        err = pred - y

        self.n += len(y)
        self.sum_y += float(np.sum(y))
        self.sum_y2 += float(np.sum(y * y))
        self.sum_err += float(np.sum(err))
        self.sum_abs_err += float(np.sum(np.abs(err)))
        self.sum_sq_err += float(np.sum(err * err))

    def values(self) -> dict[str, float | int]:
        if self.n == 0:
            raise RuntimeError("No samples available for metrics")

        sst = self.sum_y2 - (self.sum_y * self.sum_y / self.n)
        r2 = 1.0 - self.sum_sq_err / sst if sst > 0 else float("nan")

        return {
            "R2": float(r2),
            "RMSE": float(np.sqrt(self.sum_sq_err / self.n)),
            "MAE": float(self.sum_abs_err / self.n),
            "Bias": float(self.sum_err / self.n),
            "n_samples": int(self.n),
        }


def rounds_for_chunks(n_chunks: int) -> list[int]:


    base = NUM_BOOST_ROUND // n_chunks
    extra = NUM_BOOST_ROUND % n_chunks
    return [base + (1 if i < extra else 0) for i in range(n_chunks)]


def train_model_from_year_paths(
    target_key: str,
    target: str,
    paths: list[Path],
    log_period: int,
    purpose: str,
) -> lgb.Booster:
    if not paths:
        raise RuntimeError("No training paths supplied")

    model = None
    round_plan = rounds_for_chunks(len(paths))

    for idx, (path, rounds) in enumerate(
        progress(
            list(zip(paths, round_plan)),
            desc=f"{purpose} {target_key}",
            unit="year",
        ),
        start=1,
    ):
        train_df = read_year(path, target)
        x_train = train_df[FEATURES]
        y_train = train_df[target]

        train_set = lgb.Dataset(
            x_train,
            label=y_train,
            feature_name=FEATURES,
            free_raw_data=True,
        )

        log_status(
            f"{purpose}: chunk {idx}/{len(paths)} "
            f"year={path.stem} time={target_key} "
            f"rows={len(train_df):,} rounds={rounds}"
        )

        model = lgb.train(
            MODEL_PARAMS,
            train_set,
            num_boost_round=rounds,
            init_model=model,
            keep_training_booster=True,
            callbacks=[lgb.log_evaluation(period=log_period)],
        )

        del train_set, x_train, y_train, train_df
        gc.collect()

    if model is None:
        raise RuntimeError(
            f"Training failed: time={target_key}, purpose={purpose}"
        )

    if model.current_iteration() != NUM_BOOST_ROUND:
        raise RuntimeError(
            f"{purpose} {target_key}: expected "
            f"{NUM_BOOST_ROUND} total boosting rounds, "
            f"got {model.current_iteration()}"
        )

    return model


def evaluate_model_on_training_period(
    model: lgb.Booster,
    target_key: str,
    target: str,
    paths: list[Path],
) -> dict[str, float | int]:
    overall = MetricAccumulator()

    for path in progress(
        paths,
        desc=f"evaluate final model {target_key}",
        unit="year",
    ):
        train_df = read_year(path, target)
        pred = model.predict(train_df[FEATURES])

        overall.update(
            train_df[target],
            pred,
        )

        log_status(
            f"evaluated final model "
            f"year={path.stem} time={target_key} "
            f"rows={len(train_df):,}"
        )

        del train_df, pred
        gc.collect()

    return overall.values()


def save_final_model(
    model: lgb.Booster,
    target_key: str,
    metrics: dict[str, float | int],
) -> None:
    model_path = OUTPUT_DIR / f"q50_{target_key}.pkl"

    joblib.dump(
        {
            "model": model,
            "model_type": MODEL_TYPE,
            "scheme": SCHEME,
            "training_period": "2003-2014",
            "time": target_key,
            "features": FEATURES,
            "model_parameters": MODEL_PARAMS,
            "num_boost_round": NUM_BOOST_ROUND,
            "metrics": metrics,
            "intended_use": (
                "Formal production LightGBM model for historical GCM and "
                "GCM-SSP LST prediction."
            ),
        },
        model_path,
    )

    print(f"saved final production model: {model_path}", flush=True)


def feature_importance_frame(
    model: lgb.Booster,
    target_key: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": target_key,
            "feature": FEATURES,
            "importance": model.feature_importance(
                importance_type="gain"
            ),
        }
    )

    frame = frame.sort_values(
        "importance",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    frame["rank"] = np.arange(
        1,
        len(frame) + 1,
        dtype="int64",
    )

    return frame[
        ["time", "feature", "importance", "rank"]
    ]


def train_final_model(
    target_key: str,
    target: str,
    paths: list[Path],
    log_period: int,
) -> tuple[
    dict[str, float | int | str],
    pd.DataFrame,
]:
    start = time()

    model = train_model_from_year_paths(
        target_key=target_key,
        target=target,
        paths=paths,
        log_period=log_period,
        purpose="FINAL 2003-2014",
    )

    metrics = evaluate_model_on_training_period(
        model=model,
        target_key=target_key,
        target=target,
        paths=paths,
    )

    save_final_model(
        model=model,
        target_key=target_key,
        metrics=metrics,
    )

    importance = feature_importance_frame(
        model=model,
        target_key=target_key,
    )

    print(
        f"final model {target_key}: {metrics}, "
        f"elapsed={(time() - start) / 60:.2f} min",
        flush=True,
    )

    return {"time": target_key, **metrics}, importance


def save_model_info() -> None:
    info = {
        "code": "formal_production_code1_final_lightgbm",
        "training_period": "2003-2014",
        "final_model_training_period": "2003-2014",
        "final_evaluation_model_training_period": "2003-2014",
        "times": list(TARGETS.keys()),
        "targets": TARGETS,
        "feature_names": FEATURES,
        "required_columns": REQUIRED_COLUMNS,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "expected_days_per_year": EXPECTED_DAYS,
        "expected_rows_per_day": EXPECTED_ROWS_PER_DAY,
        "expected_rows_per_year": EXPECTED_ROWS_PER_YEAR,
        "model_type": MODEL_TYPE,
        "scheme": SCHEME,
        "model_parameters": MODEL_PARAMS,
        "num_boost_round_final_model": NUM_BOOST_ROUND,
        "boosting_round_interpretation": (
            "800 total boosting rounds per final model, "
            "distributed across the 12 yearly training chunks."
        ),
        "final_model_file_format": "joblib pkl",
        "final_model_files": {
            time_key: f"q50_{time_key}.pkl"
            for time_key in TARGETS
        },
        "contains_oof": False,
        "contains_independent_validation": False,
        "source_reference": "Historical LightGBM training workflow",
    }

    out = OUTPUT_DIR / "model_info.json"
    out.write_text(
        json.dumps(
            info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved model info: {out}", flush=True)


def dry_run(targets: list[str], paths: list[Path]) -> None:
    print("==== Formal production Code1 dry-run ====", flush=True)
    print(f"Input dir: {INPUT_DIR}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print(f"Years: {TRAIN_YEARS}", flush=True)
    print(f"Targets: {targets}", flush=True)
    print(f"Features: {FEATURES}", flush=True)
    print(f"Total boosting rounds per model: {NUM_BOOST_ROUND}", flush=True)
    print(
        f"Round plan across {len(paths)} yearly chunks: "
        f"{rounds_for_chunks(len(paths))}",
        flush=True,
    )
    print(f"Model params: {MODEL_PARAMS}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Formal production Code1: train the four final 2003-2014 "
            "LightGBM q50 LST prediction models only."
        )
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=TARGETS.keys(),
        default=list(TARGETS),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
    )
    parser.add_argument(
        "--log-period",
        type=int,
        default=20,
    )
    parser.add_argument("--n-jobs", type=int, default=min(os.cpu_count() or 1, 512))
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the formal production model output directory before starting.",
    )

    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    MODEL_PARAMS["n_jobs"] = args.n_jobs

    if args.clear_output and OUTPUT_DIR.exists():
        print(
            f"Removing previous output directory: {OUTPUT_DIR}",
            flush=True,
        )
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = train_paths()

    if not args.skip_validation:
        validate_inputs(paths)

    if args.dry_run:
        dry_run(
            targets=list(args.targets),
            paths=paths,
        )
        return

    final_training_rows: list[
        dict[str, float | int | str]
    ] = []
    importance_frames: list[pd.DataFrame] = []

    final_metrics_path = OUTPUT_DIR / "final_training_metrics.csv"
    importance_path = OUTPUT_DIR / "feature_importance.csv"
    if final_metrics_path.exists():
        final_training_rows.extend(
            pd.read_csv(final_metrics_path).to_dict("records")
        )
    if importance_path.exists():
        importance_frames.append(pd.read_csv(importance_path))

    for target_key in progress(
        args.targets,
        desc="Formal Code1 targets",
        unit="target",
    ):
        target = TARGETS[target_key]

        print(
            "\\n"
            + "=" * 80
            + f"\\nTIME {target_key}: final 2003-2014 production model"
            + "\\n"
            + "=" * 80,
            flush=True,
        )

        model_path = OUTPUT_DIR / f"q50_{target_key}.pkl"
        if model_path.exists():
            print(
                f"skip existing final model: {model_path}",
                flush=True,
            )
            continue

        final_metrics, importance = train_final_model(
            target_key=target_key,
            target=target,
            paths=paths,
            log_period=args.log_period,
        )

        final_training_rows.append(final_metrics)
        importance_frames.append(importance)

        pd.DataFrame(final_training_rows).to_csv(
            final_metrics_path,
            index=False,
        )
        pd.concat(
            importance_frames,
            ignore_index=True,
        ).to_csv(
            importance_path,
            index=False,
        )

        gc.collect()

    save_model_info()

    print("\\nFormal production Code1 completed.", flush=True)
    print(f"Final models: {OUTPUT_DIR}/q50_*.pkl", flush=True)
    print(f"Training metrics: {final_metrics_path}", flush=True)
    print(f"Feature importance: {importance_path}", flush=True)


if __name__ == "__main__":
    main()
