# CMOD LST: core generation scripts

This repository contains the main Python scripts for generating the CMIP6–MODIS Land Surface Temperature (CMOD LST) dataset for China. The dataset provides daily land surface temperature at four Terra/Aqua local overpass times over 2003–2014 and projected conditions for 2015–2099 under four SSP scenarios and ten CMIP6 models.

**Dataset:** https://doi.org/10.11888/Terre.tpdc.303656

## Processing steps

| Script | Function |
|---|---|
| `code1_train_final_lightgbm_2003_2014.py` | Train four LightGBM quantile-regression models using historical reference LST |
| `code2_predict_gcm_historical_lst_2003_2014.py` | Produce historical GCM-driven LST estimates before bias correction |
| `fit_historical_quantiles.py` | Estimate GCM-specific historical LST quantiles and additive bias fields |
| `code4_generate_historical_future_cmb_lst.py` | Generate historical and future LST products |
| `historical_quantile_correction.py` | Apply historical-quantile additive correction to projected LST |
| `release_netcdf_writer.py` | Write and validate annual NetCDF files |

The future correction maps each projected pre-correction LST value to its percentile in the corresponding GCM's **2003–2014 historical pre-correction LST distribution** and adds the historical bias at that percentile. Values beyond historical P1–P99 receive a fixed endpoint additive bias, without clipping projected LST. Existing variable names containing `qdm` are retained for data-file compatibility; the future correction does not use a future-window percentile distribution.

## Requirements

Python 3.10 is the tested interpreter. Install the versions listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The production-scale workload requires substantial memory and computational resources. The scripts use the Unix `resource` module and are intended for Linux or macOS.

## Input layout

Run the scripts from the repository root. The following local directories are expected; they are excluded from version control:

```text
data/
  training/
  hicpc/
  models/
  work/
  cache/
  diagnostics/
  release/
  manifests/
```

`data/training/` must contain prepared historical annual Parquet files (2003–2014). `data/hicpc/` must contain prepared historical/future GCM predictor files organized by model and scenario. The training input must include the four reference LST columns, meteorological predictors, grid coordinates, and ancillary predictors used by the models. The scripts expect the consistent 97,371-point China GRID V2 domain.

This repository does not distribute the original TRIMS, CMFD, HiCPC/CMIP6 or ancillary source files, prepared Parquet inputs, trained LightGBM model binaries, or full NetCDF release. Obtain third-party inputs from their original providers and observe the relevant data-use conditions. Preparing those inputs is outside the scope of these six scripts.

## Execution

With the required prepared inputs available:

```bash
python scripts/code1_train_final_lightgbm_2003_2014.py --dry-run
python scripts/code1_train_final_lightgbm_2003_2014.py --n-jobs 16
python scripts/code2_predict_gcm_historical_lst_2003_2014.py --gcm ACCESS-ESM1-5
python scripts/fit_historical_quantiles.py --gcm ACCESS-ESM1-5
python scripts/code4_generate_historical_future_cmb_lst.py --gcm ACCESS-ESM1-5 --ssp ssp245 --skip-historical --end-year 2099
```

The `--n-jobs` value should match available CPU resources. The script defaults to the smaller of available logical CPUs and 512; the original high-performance training setup used 512 threads. Use `--help` on each main script for available arguments and paths.

For the complete future workflow, the current implementation checks for at least 12 consecutive years of prepared future predictor data; this is an input-processing requirement, **not** the percentile window used for future bias correction. Annual publication outputs end in 2099.

## Scope and limitations

These are the core model-training, historical-quantile correction, projection-generation and annual NetCDF-writing scripts, not a self-contained end-to-end data preparation package. The full input datasets, model artifacts and large-scale execution environment are not provided, so complete numerical reproduction of all released years requires obtaining and preparing those resources.

## Citation and licensing

Dataset DOI: https://doi.org/10.11888/Terre.tpdc.303656

The source code in this repository is licensed under the [MIT License](LICENSE). This license applies only to the code in this repository, not to the original third-party input datasets or the published CMOD LST data product.

A versioned software DOI will be added after archiving an official GitHub release.
