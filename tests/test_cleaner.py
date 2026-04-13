"""
tests/test_cleaner.py

Each test covers ONE behaviour. When a test fails, the name tells
you exactly what broke — no debugging needed.

Run with:  pytest tests/test_cleaner.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.cleaner import (
    replace_sentinels,
    interpolate_gaps,
    rename_columns,
    validate_output,
    clean,
    COLUMN_RENAME,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """
    Minimal 10-row DataFrame mirroring the real dataset structure.
    Reused by multiple tests — change it once, affects all tests.
    """
    idx = pd.date_range("2020-01-01", periods=10, freq="10min")
    return pd.DataFrame({
        "T (degC)": [5.0, 5.1, 5.2, 5.3, 5.4,
                     5.5, 5.6, 5.7, 5.8, 5.9],
        "p (mbar)": [1010.0] * 10,
        "rh (%)":   [80.0]   * 10,
        "wv (m/s)": [2.0]    * 10,
        "OT":       [420.0]  * 10,
    }, index=idx)


@pytest.fixture
def df_with_sentinel(sample_df):
    """Injects one sentinel value for testing replacement."""
    df = sample_df.copy()
    df.loc[df.index[3], "OT"] = -9999.0
    return df


# ── Tests: replace_sentinels ───────────────────────────────────────────────

def test_sentinel_replaced_with_nan(df_with_sentinel):
    """The -9999 value must become NaN after replacement."""
    result = replace_sentinels(df_with_sentinel)
    assert result["OT"].isna().sum() == 1

def test_no_sentinel_values_remain(df_with_sentinel):
    """No -9999 values should survive replacement."""
    result = replace_sentinels(df_with_sentinel)
    assert (result == -9999.0).sum().sum() == 0

def test_clean_rows_untouched(df_with_sentinel):
    """Only the sentinel row changes — all other values stay the same."""
    result = replace_sentinels(df_with_sentinel)
    # Row 0 had no sentinel — it should be unchanged
    assert result["OT"].iloc[0] == 420.0

def test_no_sentinels_leaves_df_unchanged(sample_df):
    """When there are no sentinels, the DataFrame is returned as-is."""
    result = replace_sentinels(sample_df)
    pd.testing.assert_frame_equal(result, sample_df)


# ── Tests: interpolate_gaps ────────────────────────────────────────────────

def test_short_gap_is_filled(sample_df):
    """A single NaN surrounded by valid values should be interpolated."""
    df = sample_df.copy()
    df.loc[df.index[4], "T (degC)"] = np.nan
    result = interpolate_gaps(df)
    assert not result["T (degC)"].isna().any()

def test_long_gap_stays_nan(sample_df):
    """
    A gap of 8 consecutive NaN values exceeds limit=6.
    Those rows should remain NaN after interpolation.
    """
    df = sample_df.copy()
    df.loc[df.index[1:9], "T (degC)"] = np.nan  # 8 rows
    result = interpolate_gaps(df)
    # Some NaN should remain because gap > limit
    assert result["T (degC)"].isna().sum() > 0


# ── Tests: rename_columns ─────────────────────────────────────────────────

def test_columns_renamed_correctly(sample_df):
    """Known source column names must map to their clean equivalents."""
    result = rename_columns(sample_df)
    assert "temp_c"    in result.columns
    assert "T (degC)"  not in result.columns

def test_unknown_columns_untouched(sample_df):
    """Columns not in COLUMN_RENAME must be left exactly as-is."""
    df = sample_df.copy()
    df["my_custom_col"] = 1.0
    result = rename_columns(df)
    assert "my_custom_col" in result.columns


# ── Tests: validate_output ────────────────────────────────────────────────

def test_validation_passes_clean_data(sample_df):
    """Valid data must pass validation without raising or modifying anything."""
    df = rename_columns(sample_df)
    result = validate_output(df)
    # validate_output must always return the same data unchanged
    pd.testing.assert_frame_equal(result, df)


# ── Integration test: full clean() pipeline ────────────────────────────────

def test_full_clean_pipeline_runs(df_with_sentinel):
    """
    The clean() function must run end-to-end without error.
    Shape check: 10 rows at 10-min → 1 or 2 hourly rows after resample.
    """
    result = clean(df_with_sentinel)
    # Sentinel is gone
    assert (result == -9999.0).sum().sum() == 0
    # Output is hourly — fewer rows than input
    assert len(result) < len(df_with_sentinel)
    # Columns are renamed
    assert "temp_c" in result.columns

def test_clean_output_has_no_sentinels(df_with_sentinel):
    """End-to-end: no -9999 values survive the full pipeline."""
    result = clean(df_with_sentinel)
    assert (result == -9999.0).sum().sum() == 0