"""
tests/test_train.py

Run with:  pytest tests/test_train.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.models.train import (
    compute_metrics,
    naive_baseline,
    time_series_cv,
    train_final_model,
    save_model,
    load_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def small_feature_df():
    """
    96-row feature dataframe — enough for meaningful CV splits.
    Uses a sinusoidal target to simulate real temperature patterns.
    """
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=96, freq="1h")
    t   = np.linspace(0, 4 * np.pi, 96)

    return pd.DataFrame({
        "temp_c":        10 + 5 * np.sin(t) + np.random.normal(0, 0.3, 96),
        "temp_c_lag_1h": 10 + 5 * np.sin(t - 1) + np.random.normal(0, 0.3, 96),
        "temp_c_lag_24h":10 + 5 * np.sin(t - 24) + np.random.normal(0, 0.3, 96),
        "hour_sin":       np.sin(2 * np.pi * np.arange(96) % 24 / 24),
        "hour_cos":       np.cos(2 * np.pi * np.arange(96) % 24 / 24),
        "pressure_mbar":  1010 + np.random.normal(0, 2, 96),
    }, index=idx)


# ── compute_metrics tests ─────────────────────────────────────────────────

def test_perfect_prediction_metrics():
    """Perfect predictions must yield MAE=0, RMSE=0, R²=1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = compute_metrics(y, y)
    assert m["mae"]  == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"]   == 1.0

def test_metrics_return_all_keys():
    y    = np.array([10.0, 12.0, 14.0])
    pred = np.array([10.5, 11.5, 14.5])
    m    = compute_metrics(y, pred)
    for key in ["mae", "rmse", "mape", "r2"]:
        assert key in m

def test_mae_is_non_negative():
    y    = np.random.uniform(0, 30, 100)
    pred = y + np.random.normal(0, 2, 100)
    m    = compute_metrics(y, pred)
    assert m["mae"] >= 0

def test_r2_perfect_fit_is_one():
    y = np.linspace(0, 10, 50)
    m = compute_metrics(y, y)
    assert abs(m["r2"] - 1.0) < 1e-10

def test_r2_mean_prediction_is_zero():
    """Predicting the mean always gives R²=0."""
    y    = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.full_like(y, y.mean())
    m    = compute_metrics(y, pred)
    assert abs(m["r2"]) < 1e-10


# ── naive_baseline tests ──────────────────────────────────────────────────

def test_naive_baseline_returns_metrics():
    y_train = pd.Series([10.0, 11.0, 12.0, 13.0])
    y_test  = pd.Series([13.5, 14.0, 13.0, 12.5])
    m = naive_baseline(y_train, y_test)
    for key in ["mae", "rmse", "mape", "r2"]:
        assert key in m

def test_naive_baseline_uses_last_train_value():
    """Naive forecast = last training value for all test rows."""
    y_train = pd.Series([10.0, 11.0, 15.0])   # last = 15.0
    y_test  = pd.Series([15.0, 15.0, 15.0])   # all = 15 → MAE = 0
    m = naive_baseline(y_train, y_test)
    assert m["mae"] == 0.0


# ── time_series_cv tests ──────────────────────────────────────────────────

def test_cv_returns_correct_fold_count(small_feature_df):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    results = time_series_cv(small_feature_df, pipe, n_splits=3)
    assert len(results) == 3

def test_cv_each_fold_has_metrics(small_feature_df):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    results = time_series_cv(small_feature_df, pipe, n_splits=3)
    for fold in results:
        for key in ["mae", "rmse", "r2", "fold"]:
            assert key in fold

def test_cv_train_size_grows_each_fold(small_feature_df):
    """Each fold must have a larger training set than the previous."""
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    results = time_series_cv(small_feature_df, pipe, n_splits=4)
    train_sizes = [f["train_size"] for f in results]
    assert train_sizes == sorted(train_sizes), \
        "Training set must grow monotonically — fold order violated"

def test_cv_no_negative_mae(small_feature_df):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    results = time_series_cv(small_feature_df, pipe, n_splits=3)
    for fold in results:
        assert fold["mae"] >= 0


# ── save / load artifact tests ────────────────────────────────────────────

def test_save_and_load_roundtrip(small_feature_df, tmp_path, monkeypatch):
    """
    Saved artifact must load with identical pipeline and feature list.
    Uses monkeypatch to redirect cfg.model_path to a temp directory
    so tests never touch the real artifacts/ folder.
    """
    from src import config as cfg_module

    temp_model_path = tmp_path / "model.pkl"
    monkeypatch.setattr(cfg_module.cfg, "model_path", temp_model_path)

    pipeline, feature_cols = train_final_model(
        small_feature_df, model_name="ridge"
    )
    save_model(pipeline, feature_cols, cv_metrics={"mae": 0.5})

    loaded = load_model()
    assert loaded["feature_cols"] == feature_cols
    assert "pipeline"      in loaded
    assert "model_version" in loaded
    assert "cv_metrics"    in loaded
    assert loaded["cv_metrics"]["mae"] == 0.5

def test_load_raises_if_no_artifact(tmp_path, monkeypatch):
    """load_model() must raise FileNotFoundError with a helpful message."""
    from src import config as cfg_module

    monkeypatch.setattr(
        cfg_module.cfg, "model_path", tmp_path / "nonexistent.pkl"
    )
    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        load_model()