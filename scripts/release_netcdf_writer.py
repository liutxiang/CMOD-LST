from __future__ import annotations

import calendar
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

TIMES = ("0130", "1030", "1330", "2230")
LST_VARS = tuple(f"LST_{t}" for t in TIMES)
TIME_UNITS = "days since 1970-01-01 00:00:00"
CALENDAR = "proleptic_gregorian"
PRODUCT_NAME = (
    "A 0.1° multi-model historical–future land surface temperature dataset "
    "over China driven by CMIP6 and constrained by MODIS-based observations"
)
SSP_CODES = {
    "SSP1-2.6": "ssp126",
    "SSP2-4.5": "ssp245",
    "SSP3-7.0": "ssp370",
    "SSP5-8.5": "ssp585",
}


def published_year_path(release_root: Path, gcm: str, scenario: str, year: int) -> Path:
    if scenario == "historical":
        return release_root / gcm / "01_historical_2003-2014" / f"{gcm}_historical_{year}.nc"
    if scenario not in SSP_CODES:
        raise ValueError(f"Unsupported publication scenario: {scenario!r}")
    return (release_root / gcm / "02_future_2015-2099" / scenario /
            f"{gcm}_{SSP_CODES[scenario]}_{year}.nc")


def _day_int(date: pd.Timestamp) -> int:
    return int((date - pd.Timestamp("1970-01-01")).days)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_release_year(path: Path, n_grid: int, year: int) -> bool:

    if not path.exists() or path.suffix.lower() != ".nc":
        return False
    try:
        with Dataset(str(path), "r") as ds:
            days = 366 if calendar.isleap(year) else 365
            if ds.data_model != "NETCDF4":
                return False
            if "time" not in ds.dimensions or "point" not in ds.dimensions:
                return False
            if len(ds.dimensions["time"]) != days or len(ds.dimensions["point"]) != n_grid:
                return False
            if ds.getncattr("Conventions") != "CF-1.10":
                return False
            for c in ("lon", "lat"):
                if c not in ds.variables or ds.variables[c].shape != (n_grid,):
                    return False
                if ds.variables[c].dtype != np.dtype("float64"):
                    return False
            tv = ds.variables["time"]
            if tv.dtype != np.dtype("int32") or tv.getncattr("calendar") != CALENDAR or tv.getncattr("units") != TIME_UNITS:
                return False
            expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
            expected = np.asarray([_day_int(d) for d in expected_dates], dtype="int32")
            if not np.array_equal(np.asarray(tv[:], dtype="int32"), expected):
                return False
            if "crs" not in ds.variables:
                return False
            for v in LST_VARS:
                if v not in ds.variables:
                    return False
                variable = ds.variables[v]
                if variable.shape != (days, n_grid) or variable.dtype != np.dtype("float32"):
                    return False
                filters = variable.filters()
                if not filters["zlib"] or not filters["shuffle"] or filters["complevel"] != 4:
                    return False
                if variable.chunking() != [1, n_grid]:
                    return False
        return True
    except (OSError, RuntimeError, AttributeError, KeyError, ValueError):
        return False


class ReleaseNetCDFWriter:


    def __init__(self, output_path: Path, *, gcm: str, scenario: str, year: int,
                 lon: np.ndarray, lat: np.ndarray,
                 manifest_root: Path | None = None, source_files: list[str] | None = None,
                 overwrite: bool = False):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(f"Release output already exists: {self.output_path}")
        self.tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        if self.tmp.exists():
            raise RuntimeError(f"Stale incomplete release temporary file: {self.tmp}")
        self.year = int(year)
        self.gcm = str(gcm)
        self.scenario = str(scenario)
        self.lon = np.asarray(lon, dtype="float64")
        self.lat = np.asarray(lat, dtype="float64")
        if self.lon.shape != self.lat.shape or self.lon.ndim != 1:
            raise ValueError("Invalid 1D longitude/latitude grid")
        if not np.all(np.isfinite(self.lon)) or not np.all(np.isfinite(self.lat)):
            raise ValueError("Non-finite release grid coordinates")
        self.n_grid = self.lon.size
        self.expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        self.n_days = len(self.expected_dates)
        self.n_written = 0


        self._expected_digests = {name: [] for name in LST_VARS}
        self._all_values_verified = False
        self.manifest_root = manifest_root
        self.source_files = list(source_files or [])
        self.ds = Dataset(str(self.tmp), "w", format="NETCDF4")
        try:
            self._initialize()
        except BaseException:
            self.abort()
            raise

    def _initialize(self) -> None:
        ds = self.ds
        n = self.n_grid
        ds.createDimension("time", self.n_days)
        ds.createDimension("point", n)
        t = ds.createVariable("time", "i4", ("time",))
        t.standard_name = "time"
        t.long_name = "time"
        t.units = TIME_UNITS
        t.calendar = CALENDAR
        t.axis = "T"
        lo = ds.createVariable("lon", "f8", ("point",))
        lo.standard_name = "longitude"
        lo.long_name = "longitude of land point"
        lo.units = "degrees_east"
        la = ds.createVariable("lat", "f8", ("point",))
        la.standard_name = "latitude"
        la.long_name = "latitude of land point"
        la.units = "degrees_north"
        lo[:] = self.lon
        la[:] = self.lat
        self.time_var = t
        self.values = {}
        for v in LST_VARS:
            x = ds.createVariable(
                v, "f4", ("time", "point"),
                zlib=True, complevel=4, shuffle=True,
                chunksizes=(1, n), fill_value=np.float32(np.nan))
            stamp = v.split("_", 1)[1]
            x.long_name = f"land surface temperature at {stamp[:2]}:{stamp[2:]} local time"
            x.units = "degree_Celsius"
            x.coordinates = "lon lat"
            x.grid_mapping = "crs"
            self.values[v] = x
        crs = ds.createVariable("crs", "i4")
        crs.grid_mapping_name = "latitude_longitude"
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.crs_wkt = (
            'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
            'ELLIPSOID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
            'CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north],'
            'AXIS["geodetic longitude (Lon)",east],ANGLEUNIT["degree",0.0174532925199433]]'
        )
        ds.Conventions = "CF-1.10"
        ds.title = f"Daily land surface temperature projections over China: {self.gcm} {self.scenario}"
        ds.summary = "Daily 0.1-degree land surface temperature over China land points at four local overpass times."
        ds.product_full_name = PRODUCT_NAME
        ds.GCM = self.gcm
        ds.scenario = self.scenario
        ds.spatial_resolution = "0.1 degree"
        ds.spatial_representation = "land-point compressed representation; lon/lat stored once along point dimension"
        ds.n_land_points = n
        ds.temporal_resolution = "daily"
        ds.local_times = "01:30, 10:30, 13:30, 22:30"
        ds.source_format = "Direct Code4 corrected float32 array; no intermediate final-product Parquet"
        ds.source_files = "; ".join(self.source_files)
        ds.compression = "NetCDF4 zlib level 4 with shuffle"
        if self.scenario != "historical":
            ds.bias_correction_method = "historical_quantile_additive"
            ds.bias_correction_percentile_reference = "2003-2014 GCM-specific historical Pre-QDM LST q_pred_hist"
            ds.bias_correction_reference = "2003-2014 historical_bias = TRIMS LST quantile minus Pre-QDM LST quantile"
            ds.quantile_range = "P1-P99 at 1% intervals; linear interpolation"
            ds.quantile_tail_policy = "constant additive P1/P99 bias, no clipping of future LST"
            ds.future_quantile_window = "none; future 12-year self-CDF disabled"
        ds.note = "Gregorian leap days retained; daily float32 LST values are not quantized."

    def write_day(self, date: pd.Timestamp, values: dict[str, np.ndarray]) -> None:
        if self.n_written >= self.n_days:
            raise RuntimeError("Too many daily output records")
        actual = pd.Timestamp(date).normalize()
        expected = self.expected_dates[self.n_written]
        if actual != expected:
            raise RuntimeError(f"Non-consecutive date {actual.date()}, expected {expected.date()}")
        checked = {}
        for name in LST_VARS:
            data = np.asarray(values[name], dtype="float32")
            if data.shape != (self.n_grid,) or not np.all(np.isfinite(data)):
                raise RuntimeError(f"{actual.date()} {name}: invalid grid shape or non-finite data")
            checked[name] = data
        idx = self.n_written
        self.time_var[idx] = np.int32(_day_int(actual))
        for name, data in checked.items():
            self.values[name][idx, :] = data
            self._expected_digests[name].append(
                hashlib.sha256(np.ascontiguousarray(data).tobytes()).digest()
            )
        self.n_written += 1

    def close(self) -> None:
        if self.ds is None:
            raise RuntimeError("Release writer is already closed")
        if self.n_written != self.n_days:
            self.abort()
            raise RuntimeError(f"Incomplete output for {self.year}: {self.n_written}/{self.n_days}")
        self.ds.close()
        self.ds = None
        if not _validate_pending_release(self.tmp, self.n_grid, self.year):
            self.tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Invalid direct NetCDF output: {self.tmp}")


        try:
            self._verify_exact_written_data()
        except BaseException:
            self.tmp.unlink(missing_ok=True)
            raise
        self.tmp.replace(self.output_path)
        if self.manifest_root is not None:
            self._write_manifest()

    def _verify_exact_written_data(self) -> None:
        with Dataset(str(self.tmp), "r") as ds:
            if not np.array_equal(
                np.asarray(ds.variables["lon"][:], dtype="float64").view("uint64"),
                np.ascontiguousarray(self.lon).view("uint64")
            ) or not np.array_equal(
                np.asarray(ds.variables["lat"][:], dtype="float64").view("uint64"),
                np.ascontiguousarray(self.lat).view("uint64")
            ):
                raise RuntimeError(f"Direct NetCDF coordinate mismatch: {self.tmp}")
            for name in LST_VARS:
                expected = self._expected_digests[name]
                if len(expected) != self.n_days:
                    raise RuntimeError(f"Missing input day checksums for {name}")
                variable = ds.variables[name]
                for day in range(self.n_days):
                    actual = np.ascontiguousarray(
                        np.asarray(variable[day, :], dtype="float32")
                    )
                    digest = hashlib.sha256(actual.tobytes()).digest()
                    if digest != expected[day]:
                        raise RuntimeError(
                            f"Direct NetCDF bitwise verification FAILED for "
                            f"{self.gcm} {self.scenario} {self.year} day={day + 1} "
                            f"{name}; refusing to publish damaged data"
                        )
        self._all_values_verified = True

    def _write_manifest(self) -> None:
        if self.manifest_root is None:
            return
        subdir = ("01_historical_2003-2014" if self.scenario == "historical"
                  else f"02_future_2015-2099/{self.scenario}")
        path = self.manifest_root / self.gcm / subdir / (self.output_path.name + ".manifest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "output": str(self.output_path),
            "size_bytes": self.output_path.stat().st_size,
            "sha256": sha256(self.output_path),
            "gcm": self.gcm, "scenario": self.scenario,
            "n_time": self.n_days, "n_points": self.n_grid,
            "start_date": f"{self.year}-01-01", "end_date": f"{self.year}-12-31",
            "compression_level": 4, "source_files": self.source_files,
            "bias_correction_method": ("historical_quantile_additive" if self.scenario != "historical" else "historical_QDM_unchanged"),
            "all_daily_lst_bitwise_verified": self._all_values_verified,
            "verification": "SHA256 of every original daily float32 array equals the re-read NetCDF daily float32 array",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def abort(self) -> None:
        ds = getattr(self, "ds", None)
        if ds is not None:
            ds.close()
            self.ds = None
        if hasattr(self, "tmp"):
            self.tmp.unlink(missing_ok=True)


def _validate_pending_release(path: Path, n_grid: int, year: int) -> bool:

    try:
        with Dataset(str(path), "r") as ds:
            if len(ds.dimensions["point"]) != n_grid:
                return False
            if len(ds.dimensions["time"]) != (366 if calendar.isleap(year) else 365):
                return False
            if ds.getncattr("Conventions") != "CF-1.10":
                return False
            dates = np.asarray(ds.variables["time"][:], dtype="int32")
            expected = np.asarray([_day_int(d) for d in pd.date_range(f"{year}-01-01", f"{year}-12-31")], dtype="int32")
            if not np.array_equal(dates, expected):
                return False
            for v in LST_VARS:
                if v not in ds.variables or ds.variables[v].dtype != np.dtype("float32"):
                    return False
                if ds.variables[v].shape != (len(expected), n_grid):
                    return False
            return True
    except (OSError, RuntimeError, KeyError, ValueError):
        return False


class NetCDFTableSink:


    def __init__(self, path: Path, *, gcm: str, year: int, lon: np.ndarray, lat: np.ndarray,
                 manifest_root: Path | None = None, source_files: list[str] | None = None,
                 overwrite: bool = False):
        self.writer = ReleaseNetCDFWriter(
            path, gcm=gcm, scenario="historical", year=year, lon=lon, lat=lat,
            manifest_root=manifest_root, source_files=source_files, overwrite=overwrite)

    def write_table(self, table, row_group_size: int | None = None) -> None:
        cols = {v: table.column(v).to_numpy(zero_copy_only=False) for v in LST_VARS}
        dates = table.column("date").to_pandas()
        date = pd.Timestamp(dates.iloc[0] if hasattr(dates, "iloc") else dates[0])
        self.writer.write_day(date, cols)

    def close(self) -> None:
        self.writer.close()

    def abort(self) -> None:
        self.writer.abort()


def write_future_release_year(output: Path, corrected_cache: Path,
                              year_info: dict, year: int, lon: np.ndarray, lat: np.ndarray,
                              *, gcm: str, scenario: str, manifest_root: Path | None,
                              overwrite: bool = False) -> None:
    n_grid = len(lon)
    values = np.memmap(str(corrected_cache), mode="r", dtype="float32",
                       shape=(366, len(TIMES), n_grid))
    rows = np.asarray(year_info["rows"], dtype="int16")
    dates = pd.DatetimeIndex(year_info["dates"])
    if len(rows) != len(dates):
        raise RuntimeError(f"Year {year}: mismatched future cache rows/dates")
    writer = ReleaseNetCDFWriter(
        output, gcm=gcm, scenario=scenario, year=year, lon=lon, lat=lat,
        manifest_root=manifest_root, overwrite=overwrite)
    try:
        for ridx, date in zip(rows, dates):
            if not 0 <= ridx < 366:
                raise RuntimeError(f"Year {year}: invalid cache row {ridx}")
            writer.write_day(date, {
                f"LST_{time}": values[int(ridx), j, :]
                for j, time in enumerate(TIMES)
            })
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        del values
