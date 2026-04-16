"""
src/features/transforms.py — Feature engineering for weather time-series.

DESIGN PRINCIPLE:
  Every function transforms a DataFrame and returns a DataFrame.
  They are composable — call them individually in notebooks for
  experimentation, or chain them all through build_feature_matrix()
  in the automated pipeline.

LEAKAGE HISTORY (important — do not revert):
  Three rounds of leakage were identified and fixed:

  Round 1 — output_temp (OT):
    r=0.99 with temp_c, same physical sensor, same timestamp.
    Dropped in LEAKAGE_COLUMNS.

  Round 2 — physics-derived columns:
    dew_point, air_density, vapor_pressure_max/def, specific_humidity,
    h2o_concentration, dew_depression are all algebraic functions of
    temp_c. Ridge exploited these to reconstruct temp_c exactly
    (R²=1.000, MAE=0.055°C) rather than learning temporal dynamics.
    Dropped in PHYSICS_DERIVED.

  Round 3 — ambiguous forecasting target:
    Original target was temp_c at T (current). Fixed by using
    create_forecast_target() to shift target to T+1 (next hour).
    This makes the task honest: predict future from present.

  After all fixes, expected results:
    - temp_c (current) dominates SHAP / coefficients
    - GBM beats Ridge (nonlinear seasonal patterns matter)
    - MAE ~0.25–0.45°C, R² ~0.97–0.98
    - Fold performance stable, never pinned at 1.000
"""

import numpy as np
import pandas as pd
import logging

from src.config import cfg

logger = logging.getLogger(__name__)


# ── Leakage column lists ───────────────────────────────────────────────────
# Centralised here so notebooks can import them for auditing.

LEAKAGE_COLUMNS = [
    # Near-duplicate temperature sensors — same physical quantity,
    # same timestamp as target. r > 0.98 with temp_c.
    "temp_potential_k",   # Tpot = temp_c + 9.8°C/km × altitude, r=0.995
    "temp_logged",        # logger temperature sensor, r=0.982
    "output_temp",        # OT sensor, r=0.990
    "wind_direction",
]

PHYSICS_DERIVED = [
    # Variables that are algebraic functions of temp_c.
    # Ridge can invert these formulas to reconstruct temp_c exactly —
    # it becomes a physics calculator, not a time-series forecaster.
    "vapor_pressure_max",   # = 6.1078 × exp(17.27×T / (T+237.3))  pure f(temp_c)
    "vapor_pressure_def",   # = VPmax − VPact                       inherits VPmax
    "dew_depression",       # = temp_c − dew_point_c                encodes temp_c directly
    "air_density",          # = f(pressure, temp_kelvin)            function of temp_c
    "specific_humidity",    # = 0.622 × VPact / (P − VPact)        through VPact→dew_point
    "h2o_concentration",    # = specific_humidity × air_density     compound of above
    "par_max",
    "par",
    "solar_radiation",
    "raining_s",
    "rain_mm",
]


# ── Individual transform functions ────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lagged copies of the target column as features.

    WHY shift(lag) AND NOT rolling:
      df[col].shift(1) gives the exact value from 1 row ago.
      For hourly data this is "1 hour ago" — no averaging, no
      smoothing, just the raw historical value at that moment.

    WHY minimum lag of 1:
      We never include shift(0) — that would be the current value,
      which becomes the target after create_forecast_target() shifts
      it forward. Shift(1) is the most recent PAST observation.

    Args:
      target_col: column to lag. Defaults to "temp_c".
      lags: list of integer hour offsets. Defaults to cfg.lag_hours.
    """
    col      = target_col or "temp_c"
    lag_list = lags or cfg.lag_hours

    for lag in lag_list:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    logger.info(f"Added {len(lag_list)} lag features: {lag_list}")
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean and std over multiple time windows.

    WHY shift(1) before rolling:
      .shift(1).rolling(w) computes the rolling average of the
      PREVIOUS w hours — not including the current row.
      Without shift(1), .rolling(w).mean() includes the current
      observation in the window, which leaks the present into the
      feature and gives the model an unfair advantage.

    WHAT each statistic tells the model:
      rolling_mean_Nh  → short-term trend direction
      rolling_std_Nh   → volatility / rate of change
    """
    col         = target_col or "temp_c"
    window_list = windows or cfg.rolling_windows
    shifted     = df[col].shift(1)

    for w in window_list:
        df[f"{col}_roll_mean_{w}h"] = shifted.rolling(window=w).mean()
        df[f"{col}_roll_std_{w}h"]  = shifted.rolling(window=w).std()

    logger.info(f"Added rolling mean+std for windows (hours): {window_list}")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode time-of-day, month, and day-of-year as cyclical features.

    THE FORMULA:
      For a quantity that cycles over period P:
        sin_feature = sin(2π × value / P)
        cos_feature = cos(2π × value / P)

      Both sin AND cos are required together — sin alone cannot
      distinguish 2h from 10h (same sine value). The (sin, cos) pair
      uniquely identifies every point on the cycle.

    WHY is_daytime as a binary flag:
      Solar radiation is zero at night by definition — there is a
      structural regime break between day and night. A binary flag
      lets the model learn different behaviour for each regime
      explicitly rather than inferring it from cyclical features.
    """
    idx = df.index

    def cyclical_encode(values: pd.Series, period: int) -> tuple:
        angle = 2 * np.pi * values / period
        return np.sin(angle), np.cos(angle)

    # Hour of day (0–23, period=24)
    df["hour_sin"], df["hour_cos"] = cyclical_encode(
        pd.Series(idx.hour, index=idx), 24
    )

    # Month of year (1–12, period=12)
    df["month_sin"], df["month_cos"] = cyclical_encode(
        pd.Series(idx.month, index=idx), 12
    )

    # Day of year (1–365, period=365)
    df["dayofyear_sin"], df["dayofyear_cos"] = cyclical_encode(
        pd.Series(idx.day_of_year, index=idx), 365
    )

    # Daytime flag: 1 between 6am and 8pm, 0 otherwise
    df["is_daytime"] = ((idx.hour >= 6) & (idx.hour <= 20)).astype(int)

    logger.info("Added cyclical time features: hour, month, dayofyear, is_daytime")
    return df


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose wind speed + direction into u (east-west) and v
    (north-south) vector components.

    WHY decompose:
      Wind direction is circular — 359° and 1° are physically the
      same wind but numerically 358 units apart. Decomposing into
      u/v removes the circularity so the model sees a continuous space.

    METEOROLOGICAL CONVENTION:
      Direction = "wind FROM" direction (0° = FROM north)
        u = −speed × sin(direction)   east-west component
        v = −speed × cos(direction)   north-south component
    """
    if "wind_direction" not in df.columns or "wind_speed" not in df.columns:
        logger.warning("Wind columns not found — skipping u/v decomposition")
        return df

    wd_rad     = np.deg2rad(df["wind_direction"])
    df["wind_u"] = -df["wind_speed"] * np.sin(wd_rad)
    df["wind_v"] = -df["wind_speed"] * np.cos(wd_rad)

    logger.info("Added wind vector components: wind_u, wind_v")
    return df


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-derived features that carry INDEPENDENT causal
    information — i.e. features NOT recoverable from temp_c alone.

    KEPT:
      pressure_tendency = 1-hour change in pressure.
        Falling pressure → approaching weather front.
        Rising pressure → clearing conditions.
        This is a genuine causal forecasting signal independent of
        temperature, used by meteorologists since the 19th century.

    REMOVED (previously here, now in PHYSICS_DERIVED drop list):
      dew_depression = temp_c − dew_point_c.
        Encodes temp_c directly. Ridge reads it as:
          temp_c = dew_point + depression
        Both temp_c and dew_point_c are already in the feature set
        separately — their difference adds no new information and
        introduces an implicit copy of temp_c.
    """
    if "pressure_mbar" in df.columns:
        df["pressure_tendency"] = df["pressure_mbar"].diff(1)
        logger.info("Added pressure_tendency")

    return df


def create_multi_forecast_targets(
    df: pd.DataFrame,
    horizons: list[int] = [1, 6, 12, 24, 48],
    source_col: str = "temp_c",
) -> pd.DataFrame:
    """
    Shift the target column forward by multiple horizons simultaneously.
    This prepares the dataset for Multi-Output Regression.
    """
    for h in horizons:
        target_name = f"{source_col}_next_{h}h"
        df[target_name] = df[source_col].shift(-h)
    
    logger.info(f"Created multi-forecast targets for horizons: {horizons}")
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop two categories of problematic columns:

    1. LEAKAGE_COLUMNS — near-duplicate temperature sensors with
       r > 0.98 vs temp_c, measured at the same timestamp.
       Keeping them means the model predicts current temp from
       a near-copy of current temp — not forecasting.

    2. PHYSICS_DERIVED — columns that are algebraic functions of
       temp_c. Ridge exploited these to reconstruct temp_c with
       near-perfect accuracy (R²=1.000) rather than learning the
       temporal dynamics of how temperature changes over time.

    Safe to call even if some columns are already absent — only
    drops what is actually present in the dataframe.
    """
    all_to_drop = LEAKAGE_COLUMNS + PHYSICS_DERIVED
    to_drop     = [c for c in all_to_drop if c in df.columns]

    if to_drop:
        df = df.drop(columns=to_drop)
        logger.info(f"Dropped {len(to_drop)} redundant/derived columns: {to_drop}")
    else:
        logger.info("drop_redundant_columns: nothing to drop (already clean)")

    return df


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity check the final feature matrix before saving.

    Logs a warning for any leakage column still present.
    Never modifies the dataframe — observation only.
    """
    all_bad = LEAKAGE_COLUMNS + PHYSICS_DERIVED
    still_present = [c for c in all_bad if c in df.columns]

    if still_present:
        logger.warning(
            f"VALIDATION FAILED — leakage columns still present: "
            f"{still_present}. Check drop_redundant_columns()."
        )
    else:
        logger.info(
            "Feature validation passed — no leakage columns present."
        )

    if "temp_c_next_1h" not in df.columns:
        logger.warning(
            "VALIDATION: 'temp_c_next_1h' not found. "
            "Was create_forecast_target() called?"
        )

    return df


# ── Orchestrator ───────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    forecast_horizons: list[int] = [1, 6, 12, 24, 48],
) -> pd.DataFrame:
    """
    Full feature engineering pipeline, upgraded for Multi-Output.
    """
    logger.info("Building multi-output feature matrix...")
    rows_in = len(df)
    
    df = add_time_features(df)
    df = add_wind_components(df)
    df = add_physics_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    
    # NEW: Create multiple targets
    df = create_multi_forecast_targets(df, horizons=forecast_horizons)
    
    df = drop_redundant_columns(df)
    
    df = df.dropna()
    logger.info(
        f"Feature matrix: {len(df):,} rows × {df.shape[1]} columns "
        f"({rows_in - len(df)} rows dropped)"
    )
    return df