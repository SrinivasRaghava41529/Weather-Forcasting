"""
src/data/cleaner.py — All cleaning logic in one auditable place.

PROFESSIONAL PRINCIPLE — why a dedicated module:
  Every cleaning decision is business logic that evolves.
  Isolating it here means:
    - You can audit every decision in one file
    - Each step is unit-testable independently
    - The pipeline and the notebook call the same clean()
    - When you revisit this project in 6 months, the reasoning is here

DATA QUALITY ISSUES FOUND IN EDA (01_eda.ipynb):
  1. Sentinel -9999: OT has 50 rows, wv has 1 row — sensor error codes
  2. Timing gap: one ~100-minute gap in the 10-minute sequence
  3. Column names: cryptic units and encoding artefacts (µ, ²)
"""

import pandas as pd
import numpy as np
import logging
from src.config import cfg

# Module-level logger — every step announces itself.
# In production logs you can trace exactly where a failure happened.
logger = logging.getLogger(__name__)


# ── Column rename map ──────────────────────────────────────────────────────
# Professional rule: cryptic source names appear ONLY here.
# Everything downstream uses the clean names.
# Changing a name? Edit this dict. Nothing else needs to change.
COLUMN_RENAME = {
    "p (mbar)":           "pressure_mbar",
    "T (degC)":           "temp_c",
    "Tpot (K)":           "temp_potential_k",
    "Tdew (degC)":        "dew_point_c",
    "rh (%)":             "humidity_pct",
    "VPmax (mbar)":       "vapor_pressure_max",
    "VPact (mbar)":       "vapor_pressure_act",
    "VPdef (mbar)":       "vapor_pressure_def",
    "sh (g/kg)":          "specific_humidity",
    "H2OC (mmol/mol)":    "h2o_concentration",
    "rho (g/m**3)":       "air_density",
    "wv (m/s)":           "wind_speed",
    "max. wv (m/s)":      "wind_speed_max",
    "wd (deg)":           "wind_direction",
    "rain (mm)":          "rain_mm",
    "raining (s)":        "raining_s",
    "SWDR (W/m\ufffd)":          "solar_radiation",
    "PAR (\ufffdmol/m\ufffd/s)":       "par",
    "max. PAR (\ufffdmol/m\ufffd/s)":  "par_max",
    "Tlog (degC)":        "temp_logged",
    "OT":                 "output_temp",
}

def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace -9999 sensor error codes with NaN.

    WHY NaN INSTEAD OF DROPPING:
      Dropping rows would create irregular time gaps, which breaks
      time-series resampling and lag feature calculation.
      NaN preserves the row's time slot while marking it as unknown.
      pandas resampling skips NaN in aggregations automatically.

    WHY NOT JUST FILTER OUTLIERS:
      -9999 is not a statistical outlier — it's a known error code
      defined by the sensor manufacturer. Treating it as a real
      measurement (even an extreme one) would be wrong.
    """
    sentinel = cfg.sentinel_value
    sentinel_counts = (df == sentinel).sum()
    affected = sentinel_counts[sentinel_counts > 0]

    if affected.empty:
        logger.info("No sentinel values found")
        return df

    logger.info(f"Replacing sentinel ({sentinel}) in {len(affected)} columns:")
    for col, count in affected.items():
        logger.info(f"  {col}: {count} values")

    return df.replace(sentinel, np.nan)


def interpolate_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values using time-aware linear interpolation.

    WHY 'method=time' NOT 'method=linear':
      method='linear' treats each row as equally spaced regardless of
      timestamps. method='time' weights the interpolation by the actual
      elapsed time — essential when your index has an irregular gap.

    WHY limit=6:
      6 steps × 10-minute intervals = 60 minutes maximum fill.
      A sensor outage longer than 1 hour means something went wrong.
      We'd rather keep NaN (and potentially lose a few rows to dropna
      later) than fabricate an hour of weather data.

    AFTER RESAMPLING TO HOURLY:
      The limit becomes 6 × 1-hour = 6 hours. Still conservative.
    """
    nan_before = df.isna().sum().sum()

    if nan_before == 0:
        logger.info("No NaN values to interpolate")
        return df

    df_interpolated = df.interpolate(method="time", limit=6)
    nan_after = df_interpolated.isna().sum().sum()
    filled = nan_before - nan_after

    logger.info(f"Interpolated {filled} values ({nan_after} NaN remain after limit)")
    return df_interpolated


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the canonical column name map.

    Only renames columns that exist — safe to call even if the source
    schema changes slightly (encoding variants, extra columns, etc.).
    Logs which columns were actually renamed vs skipped.
    """
    rename_map = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    skipped = [k for k in COLUMN_RENAME if k not in df.columns]

    df = df.rename(columns=rename_map)

    logger.info(f"Renamed {len(rename_map)} columns")
    if skipped:
        logger.debug(f"Skipped (not found in data): {skipped}")

    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downsample from 10-minute to hourly by taking the mean.

    WHY AFTER RENAME:
      After rename, column names are readable in logs and debug output.
      Resampling before rename means log messages say "rho (g/m**3)"
      instead of "air_density" — harder to read and grep.

    WHY MEAN AND NOT SUM OR LAST:
      Temperature, pressure, humidity are instantaneous measurements —
      their hourly representative value is the mean of the period.
      Rain is the exception: it should be summed. But rain is not our
      target here, so mean is acceptable for the whole dataset.

    10-min → hourly: 52,696 rows → ~8,760 rows (one year of hours)
    """
    freq = cfg.resample_freq
    rows_before = len(df)
    df_hourly = df.resample(freq).mean()
    rows_after = len(df_hourly)

    logger.info(
        f"Resampled {rows_before:,} → {rows_after:,} rows "
        f"(freq={freq}, reduction={rows_before/rows_after:.1f}x)"
    )
    return df_hourly


def validate_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final sanity check — raise a warning if values violate physical limits.

    WHY VALIDATION IS SEPARATE FROM CLEANING:
      Cleaning fixes known problems. Validation catches unknown problems —
      things that slipped through, or new issues in future data batches.
      This step never modifies data. It only warns.

    PHYSICAL LIMITS USED:
      These are real-world extremes, not dataset-specific values.
      If a future batch has a real reading near the limit, it passes.
      Only impossible values (instrument failures we haven't seen before)
      will trigger warnings.
    """
    # Column name → (physical_min, physical_max)
    # Using renamed column names (this runs after rename_columns)
    checks = {
        "temp_c":       (-89.0, 57.0),    # all-time Earth records
        "pressure_mbar": (870.0, 1085.0), # extreme low/high sea level
        "humidity_pct":  (0.0,  100.0),   # definition of relative humidity
        "wind_speed":    (0.0,  115.0),   # category 5 hurricane ≈ 90 m/s
    }

    any_violations = False
    for col, (low, high) in checks.items():
        if col not in df.columns:
            continue
        violations = ((df[col] < low) | (df[col] > high)).sum()
        if violations > 0:
            logger.warning(
                f"VALIDATION: '{col}' has {violations} values "
                f"outside physical range [{low}, {high}]"
            )
            any_violations = True

    if not any_violations:
        logger.info("Validation passed — all columns within physical limits")

    return df  # always return unchanged — never modify in validation

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline — single entry point for automation.

    This is the function the pipeline script calls.
    This is the function the notebook imports and calls.
    Same function. Same result. Every time.

    Step order is deliberate and load-bearing:
      1. Replace sentinels   ← MUST be first (before any math)
      2. Interpolate         ← fills the NaN created in step 1
      3. Rename columns      ← readable names for all downstream code
      4. Resample            ← MUST be after sentinel replacement
      5. Validate            ← MUST be last (checks the final output)

    Args:
        df: Raw DataFrame as returned by loader.load_raw()

    Returns:
        Cleaned, hourly-resampled DataFrame ready for feature engineering
    """
    logger.info("=" * 40)
    logger.info("Starting cleaning pipeline")
    logger.info(f"Input:  {df.shape[0]:,} rows × {df.shape[1]} columns")

    df = replace_sentinels(df)
    df = interpolate_gaps(df)
    df = rename_columns(df)
    df = resample_to_hourly(df)
    df = validate_output(df)

    logger.info(f"Output: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info("Cleaning pipeline complete")
    logger.info("=" * 40)

    return df

