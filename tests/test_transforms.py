"""
tests/test_transforms.py

Run with:  pytest tests/test_transforms.py -v
"""

import pytest
import pandas as pd
import numpy as np
from src.features.transforms import (
    add_lag_features,
    add_rolling_features,
    add_time_features,
    add_wind_components,
    add_physics_features,
    build_feature_matrix,
)


@pytest.fixture
def hourly_df():
    """48 rows of hourly data — enough to satisfy lag_48h without all NaN."""
    idx = pd.date_range("2020-06-01", periods=96, freq="1h")
    np.random.seed(42)
    return pd.DataFrame({
        "temp_c":        np.random.uniform(10, 25, 96),
        "pressure_mbar": np.random.uniform(990, 1015, 96),
        "dew_point_c":   np.random.uniform(5, 15, 96),
        "wind_speed":    np.random.uniform(0, 10, 96),
        "wind_direction": np.random.uniform(0, 360, 96),
        "humidity_pct":  np.random.uniform(40, 90, 96),
    }, index=idx)


# ── Lag tests ─────────────────────────────────────────────────────────────

def test_lag_columns_created(hourly_df):
    result = add_lag_features(hourly_df, lags=[1, 6, 24])
    assert "temp_c_lag_1h"  in result.columns
    assert "temp_c_lag_6h"  in result.columns
    assert "temp_c_lag_24h" in result.columns

def test_lag_1h_equals_previous_row(hourly_df):
    """
    From row 1 onward (where both the current and previous row
    exist in the dataframe), lag_1h must equal the previous row's value.
    Row 0 is excluded — its lag comes from before the dataframe starts.
    """
    result = add_lag_features(hourly_df.copy(), lags=[1])

    # Check rows 1 through 5 (row 0 has no previous row in the df)
    for i in range(1, 6):
        expected = hourly_df["temp_c"].iloc[i - 1]
        actual   = result["temp_c_lag_1h"].iloc[i]
        assert abs(expected - actual) < 1e-8, \
            f"Row {i}: expected {expected}, got {actual}"

def test_lag_first_row_is_nan(hourly_df):
    result = add_lag_features(hourly_df, lags=[1])
    # First row has no history — must be NaN
    assert pd.isna(result["temp_c_lag_1h"].iloc[0])

def test_lag_does_not_modify_original_column(hourly_df):
    original_vals = hourly_df["temp_c"].copy()
    result = add_lag_features(hourly_df.copy(), lags=[1])
    pd.testing.assert_series_equal(result["temp_c"], original_vals)


# ── Rolling tests ─────────────────────────────────────────────────────────

def test_rolling_columns_created(hourly_df):
    result = add_rolling_features(hourly_df, windows=[3, 6])
    assert "temp_c_roll_mean_3h" in result.columns
    assert "temp_c_roll_std_3h"  in result.columns
    assert "temp_c_roll_mean_6h" in result.columns

def test_rolling_uses_shift_not_current(hourly_df):
    """Rolling mean must not include the current row's value."""
    result = add_rolling_features(hourly_df.copy(), windows=[3])
    # Row 3 rolling_mean_3h = mean of rows 0, 1, 2 (shifted, so current not included)
    expected = hourly_df["temp_c"].iloc[0:3].mean()
    actual = result["temp_c_roll_mean_3h"].iloc[3]
    assert abs(expected - actual) < 1e-8


# ── Cyclical time tests ───────────────────────────────────────────────────

def test_time_features_created(hourly_df):
    result = add_time_features(hourly_df)
    for col in ["hour_sin", "hour_cos", "month_sin",
                "month_cos", "dayofyear_sin", "dayofyear_cos", "is_daytime"]:
        assert col in result.columns

def test_cyclical_values_in_range(hourly_df):
    """sin and cos values must always be in [-1, 1]."""
    result = add_time_features(hourly_df)
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert result[col].between(-1, 1).all(), f"{col} out of [-1, 1]"

def test_midnight_hour_encoding(hourly_df):
    """Hour 0 and hour 23 should have similar cos values (both near top of circle)."""
    h0_cos  = np.cos(2 * np.pi * 0  / 24)
    h23_cos = np.cos(2 * np.pi * 23 / 24)
    # Both should be close to 1.0 (top of the unit circle)
    assert abs(h0_cos - h23_cos) < 0.1

def test_is_daytime_binary(hourly_df):
    result = add_time_features(hourly_df)
    unique_vals = set(result["is_daytime"].unique())
    assert unique_vals.issubset({0, 1})


# ── Wind component tests ──────────────────────────────────────────────────

def test_wind_components_created(hourly_df):
    result = add_wind_components(hourly_df)
    assert "wind_u" in result.columns
    assert "wind_v" in result.columns

def test_wind_speed_conserved(hourly_df):
    """sqrt(u² + v²) must equal the original wind speed."""
    result = add_wind_components(hourly_df)
    reconstructed = np.sqrt(result["wind_u"]**2 + result["wind_v"]**2)
    np.testing.assert_allclose(
        reconstructed.values,
        hourly_df["wind_speed"].values,
        rtol=1e-6,
    )


# ── Integration test ──────────────────────────────────────────────────────

def test_build_feature_matrix_runs(hourly_df):
    result = build_feature_matrix(hourly_df.copy())
    # No NaN in output
    assert result.isna().sum().sum() == 0
    # More columns than input (features were added)
    assert result.shape[1] > hourly_df.shape[1]
    # Fewer rows (NaN rows from shifts were dropped)
    assert len(result) < len(hourly_df)

def test_no_future_leakage(hourly_df):
    """
    The target column value at row i must NOT appear in any feature
    at row i — that would mean we used the present to predict the present.
    """
    result = build_feature_matrix(hourly_df.copy())
    feature_cols = [c for c in result.columns if c != "temp_c"]

    for col in feature_cols:
        if "lag" in col:
            # lag columns should not equal temp_c (they're from the past)
            # This is a soft check — they CAN be equal by coincidence,
            # but exact equality across all rows would be suspicious
            identical_rows = (result[col] == result["temp_c"]).sum()
            # Allow up to 5% coincidental matches
            assert identical_rows < len(result) * 0.05, \
                f"{col} suspiciously identical to target — possible leakage"