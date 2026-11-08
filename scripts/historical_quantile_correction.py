from __future__ import annotations
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


def apply_qdm_to_target_year_parallel(
    code4,
    cache_dir: Path,
    qdm_handles,
    year_infos: dict,
    target_year: int,
    first_future_year: int,
    last_future_year: int,
    n_grid: int,
    grid_chunk: int,
    max_workers: int = 4,
):


    del first_future_year, last_future_year
    pred_map = code4.ensure_prediction_map(cache_dir, target_year, n_grid, mode="r")
    corr_map = code4.open_corrected_map(cache_dir, target_year, n_grid)
    stats = {
        t: {"sum_pred": 0., "sum_corr": 0., "sum_correction": 0., "n": 0, "applied": 0}
        for t in code4.TIMES
    }
    coverage = []
    info = year_infos[target_year]

    def calculate(job):
        t, tidx, month, rows, g0, g1, bias, qhist, valid = job
        pred = np.asarray(pred_map[rows, tidx, g0:g1], dtype="float64")
        p_hist = code4.percentile_from_historical_quantiles(
            pred, qhist[:, g0:g1].T
        )
        delta = code4.interpolation_from_percentile(
            p_hist, bias[:, g0:g1].T.astype("float64")
        )
        applied = (
            valid[g0:g1][None, :]
            & np.isfinite(pred)
            & np.isfinite(p_hist)
            & np.isfinite(delta)
        )
        corrected = pred.astype("float32")
        corrected[applied] = (pred[applied] + delta[applied]).astype("float32")
        corr_map[rows, tidx, g0:g1] = corrected
        ok = np.isfinite(pred) & np.isfinite(corrected)
        nums = (0., 0., 0., 0, 0)
        if np.any(ok):
            p = pred[ok]
            c = corrected[ok].astype("float64")
            nums = (float(np.sum(p)), float(np.sum(c)), float(np.sum(c - p)),
                    int(np.sum(ok)), int(np.sum(applied & ok)))


        upper = np.asarray(qhist[-1, g0:g1], dtype="float64")
        lower = np.asarray(qhist[0, g0:g1], dtype="float64")
        hi = int(np.sum(applied & (pred > upper[None, :])))
        lo = int(np.sum(applied & (pred < lower[None, :])))
        return t, month, g0, int(np.sum(valid[g0:g1])), int(np.sum(applied)), hi, lo, nums

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for month in range(1, 13):
                rows = np.asarray(info["month_rows"][month], dtype="int16")
                if len(rows) < 28 or len(rows) > 31:
                    raise RuntimeError(f"Unexpected calendar month length {target_year}-{month}: {len(rows)}")
                tasks = []
                totals = {}
                for tidx, t in enumerate(code4.TIMES):
                    ds = qdm_handles[t]
                    for required in ("historical_bias", "q_pred_hist", "qdm_valid"):
                        if required not in ds.variables:
                            raise RuntimeError(f"Historical Code3 QDM missing {required} for {t}")
                    bias = np.asarray(ds["historical_bias"][month - 1, :, :], dtype="float32")
                    qhist = np.asarray(ds["q_pred_hist"][month - 1, :, :], dtype="float32")
                    valid = np.asarray(ds["qdm_valid"][month - 1, :], dtype="int8") == 1
                    if bias.shape != (99, n_grid) or qhist.shape != (99, n_grid):
                        raise RuntimeError(f"Invalid historical QDM grid shape for {t}")
                    totals[t] = [0, 0, 0, 0]
                    for g0 in range(0, n_grid, grid_chunk):
                        g1 = min(g0 + grid_chunk, n_grid)
                        tasks.append((t, tidx, month, rows, g0, g1, bias, qhist, valid))
                for t, _, g0, elig, applied, high, low, nums in pool.map(calculate, tasks):
                    a = totals[t]
                    a[0] += elig
                    a[1] += applied
                    a[2] += high
                    a[3] += low
                    acc = stats[t]
                    acc["sum_pred"] += nums[0]
                    acc["sum_corr"] += nums[1]
                    acc["sum_correction"] += nums[2]
                    acc["n"] += nums[3]
                    acc["applied"] += nums[4]
                for t in code4.TIMES:
                    e, a, high, low = totals[t]
                    total = int(len(rows) * n_grid)
                    coverage.append({
                        "year": target_year, "time": t, "month": month,
                        "method": "historical_quantile_additive",
                        "reference_start": 2003, "reference_end": 2014,
                        "seasonal_months": "+".join(f"{m:02d}" for m in code4.month_window(month)),
                        "n_target_days": int(len(rows)),
                        "n_total_grids": n_grid,
                        "n_eligible_grids": e,
                        "eligible_grid_fraction": e/n_grid,
                        "applied_cells": a,
                        "target_cells": total,
                        "applied_fraction": a/total,
                        "above_hist_p99_cells": high,
                        "below_hist_p01_cells": low,
                    })
                corr_map.flush()
                print(f"HISTORICAL_QUANTILE_MONTH_DONE {target_year}-{month:02d} method=historical_quantile_additive", flush=True)
                gc.collect()
        annual = []
        for t in code4.TIMES:
            acc = stats[t]
            n = acc["n"]
            annual.append({
                "year": target_year, "time": t, "method": "historical_quantile_additive",
                "reference_start": 2003, "reference_end": 2014,
                "n_cells": n,
                "annual_mean_LST_pred": acc["sum_pred"]/n if n else np.nan,
                "annual_mean_LST_corr": acc["sum_corr"]/n if n else np.nan,
                "annual_mean_correction": acc["sum_correction"]/n if n else np.nan,
                "qdm_applied_fraction": acc["applied"]/n if n else np.nan,
            })
        return code4.corrected_year_cache_path(cache_dir,target_year),annual,coverage
    finally:
        corr_map.flush()
        del corr_map, pred_map
        gc.collect()
