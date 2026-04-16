import numpy as np
import pandas as pd
import joblib
import logging
from typing import Any

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import cfg

logger = logging.getLogger(__name__)


# ── Candidate models ───────────────────────────────────────────────────────
# All candidates are evaluated with identical CV, identical folds, identical
# metrics. No cherry-picking. Add new candidates here — never in the loop.
#
# WHY THESE THREE:
#   Ridge:            Linear baseline. If Ridge beats tree models, the
#                     feature-target relationship is near-linear. Simpler
#                     is better in that case.
#   RandomForest:     Robust to outliers, handles nonlinearity, naturally
#                     produces feature importances for SHAP.
#   GradientBoosting: Usually best on tabular data after physics leakage
#                     is removed — captures seasonal nonlinearities that
#                     Ridge cannot.

CANDIDATE_MODELS: dict[str, Any] = {
    "ridge": Ridge(alpha=1.0),

    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=cfg.random_state,
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=cfg.random_state,
    ),
}


# ══════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute a standard suite of regression metrics.

    WHY FOUR METRICS INSTEAD OF ONE:

      MAE  (Mean Absolute Error)
           Average error in original units (°C).
           Interpretable: "on average we're X degrees off."
           Treats all errors equally — a 5°C miss is 5× a 1°C miss.

      RMSE (Root Mean Squared Error)
           Penalises large errors more than MAE.
           Use when big misses are especially costly, e.g. frost events.

      MAPE (Mean Absolute Percentage Error)
           Scale-independent. Good for non-technical stakeholders.
           Can be misleading when actuals are near zero.

      R²   (Coefficient of Determination)
           Fraction of variance explained by the model.
           R²=1.0 → perfect. R²=0.0 → no better than predicting the mean.
           Negative R² → worse than the mean — a red flag.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # Guard against division by zero in MAPE
    nonzero = np.abs(y_true) > 1e-8
    if nonzero.sum() > 0:
        mape = np.mean(
            np.abs(
                (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
            )
        ) * 100
    else:
        mape = float("nan")

    return {
        "mae":  round(float(mae),  4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 4),
        "r2":   round(float(r2),   4),
    }


# ══════════════════════════════════════════════════════════════════════════
# NAIVE BASELINE  (Fix 1)
# ══════════════════════════════════════════════════════════════════════════

def naive_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
   
    # Each prediction = previous test value
    # First test row uses last training value (no test history yet)
    last_train_val = float(y_train.iloc[-1])
    y_pred = y_test.shift(1).fillna(last_train_val).values

    metrics = compute_metrics(y_test.values, y_pred)
    logger.info(
        f"Persistence baseline: MAE={metrics['mae']:.3f}°C  "
        f"RMSE={metrics['rmse']:.3f}  R²={metrics['r2']:.4f}"
    )
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# FEATURE / TARGET SPLIT  (Fix 2)
# ══════════════════════════════════════════════════════════════════════════

def _prepare_X_y(
    df: pd.DataFrame,
    target_col: str = "temp_c_next_1h",   # ← FIXED: was "temp_c"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the feature matrix into X (features) and y (target).

    CRITICAL — WHY THIS DEFAULT MATTERS:
      The old default was target_col="temp_c". That caused:
        1. "temp_c" (current temperature) excluded from features
        2. "temp_c_next_1h" (the future answer) left inside X
        3. Model predicted current temp using next hour's temp as input
        4. R²=1.000 trivially — it was reading the answer

      The correct target is "temp_c_next_1h" — created by
      create_forecast_target() in transforms.py using shift(-1).

      With this fix:
        - X contains temp_c (current), lag features, time features, etc.
        - y contains temp_c_next_1h (1 hour in the future)
        - The model must extrapolate forward, not reconstruct the present
    """
    if target_col not in df.columns:
        available = [c for c in df.columns if "temp_c" in c]
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame.\n"
            f"Columns containing 'temp_c': {available}\n"
            f"Did you forget to run create_forecast_target() in transforms.py?"
        )

    feature_cols = [c for c in df.columns if c != target_col]
    return df[feature_cols], df[target_col]

def _prepare_X_y_multi(
    df: pd.DataFrame,
    target_prefix: str = "temp_c_next_",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the feature matrix into X (features) and y (MULTIPLE targets).
    y is now a DataFrame, not a Series.
    """
    # Find all columns that start with the target prefix
    target_cols = [c for c in df.columns if c.startswith(target_prefix)]
    
    if not target_cols:
        raise ValueError(f"No target columns found starting with '{target_prefix}'")
        
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    return df[feature_cols], df[target_cols]


# ══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def time_series_cv(
    df: pd.DataFrame,
    model: Any,
    n_splits: int | None = None,
    target_col: str = "temp_c_next_1h",   # ← FIXED: was "temp_c"
) -> list[dict]:
    """
    Walk-forward cross-validation — the only honest CV for time series.

    WHAT HAPPENS IN EACH FOLD:
      Fold 1: train on rows 1–1451,  test on rows 1452–2902
      Fold 2: train on rows 1–2902,  test on rows 2903–4353
      ...each fold trains on all past, tests only on future.

    WHY NOT STANDARD K-FOLD:
      Standard k-fold shuffles rows randomly before splitting. For time
      series this means fold 3 might train on December and test on March.
      The model sees the future during training — that is data leakage.
      TimeSeriesSplit enforces strict temporal ordering in every fold.

    WHAT GOOD FOLD RESULTS LOOK LIKE:
      Slightly improving or stable MAE as training size grows.
      R² in 0.96–0.99 range. Never 1.000.
      A step-change from MAE=0.095 → 0.020 is a red flag.

    RETURNS:
      List of per-fold metric dicts. Check individually for distribution
      shift — degrading performance in later folds = seasonal mismatch.
    """
    n_splits = n_splits or cfg.cv_folds
    X, y     = _prepare_X_y(df, target_col)
    tscv     = TimeSeriesSplit(n_splits=n_splits)
    results  = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train = X.iloc[train_idx].values
        X_test  = X.iloc[test_idx].values
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        model.fit(X_train, y_train.values)
        y_pred = model.predict(X_test)

        metrics               = compute_metrics(y_test.values, y_pred)
        metrics["fold"]       = fold_idx
        metrics["train_size"] = len(train_idx)
        metrics["test_size"]  = len(test_idx)
        results.append(metrics)

        logger.info(
            f"  Fold {fold_idx}/{n_splits} "
            f"(train={len(train_idx):,}  test={len(test_idx):,}) "
            f"MAE={metrics['mae']:.3f}  R²={metrics['r2']:.4f}"
        )

    return results


# ══════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON  (Fix 3)
# ══════════════════════════════════════════════════════════════════════════

def compare_models(
    df: pd.DataFrame,
    target_col: str = "temp_c_next_1h",   # ← FIXED: was "temp_c"
) -> pd.DataFrame:
    """
    Train every candidate model with identical CV and return a
    ranked comparison DataFrame sorted by MAE ascending.

    Also computes the persistence baseline so you immediately see
    which models add genuine value over the simplest possible forecast.

    PROFESSIONAL RULE:
      Every model gets the same folds, same data, same metrics.
      The naive baseline is always computed first and included in the
      comparison — never omit it to make ML models look better.
    """
    X, y    = _prepare_X_y(df, target_col)
    results = []

    # ── Persistence baseline first ────────────────────────────────────────
    split_idx        = int(len(y) * 0.8)
    y_train_baseline = y.iloc[:split_idx]
    y_test_baseline  = y.iloc[split_idx:]
    baseline_metrics = naive_baseline(y_train_baseline, y_test_baseline)
    baseline_metrics["model"] = "persistence_baseline"
    results.append(baseline_metrics)

    # ── ML candidates ─────────────────────────────────────────────────────
    for name, model in CANDIDATE_MODELS.items():
        logger.info(f"\nTraining {name}...")

        # Pipeline with StandardScaler:
        #   Ridge needs scaling (L2 penalty is sensitive to feature scale).
        #   Tree models don't strictly need it, but consistency is worth more
        #   than marginal speed gain — one interface, one .fit(), one .predict().
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  model),
        ])

        fold_metrics = time_series_cv(df, pipe, target_col=target_col)

        avg = {
            "model": name,
            "mae":   round(np.mean([f["mae"]  for f in fold_metrics]), 4),
            "rmse":  round(np.mean([f["rmse"] for f in fold_metrics]), 4),
            "mape":  round(np.mean([f["mape"] for f in fold_metrics]), 4),
            "r2":    round(np.mean([f["r2"]   for f in fold_metrics]), 4),
        }
        results.append(avg)
        logger.info(
            f"  {name}: "
            f"MAE={avg['mae']:.3f}  RMSE={avg['rmse']:.3f}  R²={avg['r2']:.4f}"
        )

    comparison = (
        pd.DataFrame(results)
        .sort_values("mae")
        .reset_index(drop=True)
    )
    return comparison


# ══════════════════════════════════════════════════════════════════════════
# FINAL TRAINING AND PERSISTENCE  (Fix 3 continued)
# ══════════════════════════════════════════════════════════════════════════

def train_final_model(
    df: pd.DataFrame,
    model_name: str = "random_forest",
) -> tuple[Pipeline, list[str], list[str]]:
    """
    Train the chosen model on ALL available data.
    Now returns: (pipeline, feature_cols, target_cols)
    """
    X_df, y_df = _prepare_X_y_multi(df)
    
    feature_cols = X_df.columns.tolist()
    target_cols = y_df.columns.tolist() # Keep track of the output order!
    
    X = X_df.values
    y = y_df.values
    
    model = CANDIDATE_MODELS[model_name]
    pipe  = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  model),
    ])
    
    logger.info(f"Training final {model_name} on {len(X):,} samples for {len(target_cols)} horizons...")
    pipe.fit(X, y)
    logger.info("Final multi-output model training complete")
    
    return pipe, feature_cols, target_cols


def save_model(
    pipeline: Pipeline,
    feature_cols: list[str],
    target_cols: list[str],          # <-- Multi-output parameter
    cv_metrics: dict | None = None,
) -> None:
    """
    Save model artifact: pipeline + feature list + target list + metadata.
    """
    artifact = {
        "pipeline":      pipeline,
        "feature_cols":  feature_cols,
        "target_cols":   target_cols,    # <-- FIXED: Save the list of 5 targets!
        "model_version": cfg.model_version,
        "cv_metrics":    cv_metrics or {},
    }

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib  # ensuring joblib is available
    joblib.dump(artifact, cfg.model_path)
    
    logger.info(f"Artifact saved → {cfg.model_path}")
    logger.info(f"  version      : {cfg.model_version}")
    logger.info(f"  features     : {len(feature_cols)}")
    logger.info(f"  targets      : {len(target_cols)} horizons") # <-- FIXED: dynamic logging


def load_model() -> dict[str, Any]:
    
    if not cfg.model_path.exists():
        raise FileNotFoundError(
            f"No model artifact found at: {cfg.model_path}\n"
            f"Run one of:\n"
            f"  python scripts/run_pipeline.py\n"
            f"  or execute the final cells of notebooks/04_modelling.ipynb"
        )

    artifact = joblib.load(cfg.model_path)

    # FIXED: Validate against the new Multi-Output structure ('target_cols' instead of 'target_col')
    required_keys = {"pipeline", "feature_cols", "model_version", "target_cols"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise ValueError(
            f"Artifact is missing keys: {missing}\n"
            f"Retrain the model — the artifact predates the Multi-Output upgrade."
        )

    # Log the successful load
    logger.info(
        f"Model loaded: version={artifact.get('model_version', 'unknown')}  "
        f"features={len(artifact.get('feature_cols', []))}  "
        f"targets={len(artifact.get('target_cols', []))}"
    )

    return artifact


# ══════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC HELPERS  (for use in notebooks only — not called by pipeline)
# ══════════════════════════════════════════════════════════════════════════

def check_no_leakage(df: pd.DataFrame) -> None:
    """
    Assert that the feature matrix contains no known leakage columns.
    Call this at the top of 04_modelling.ipynb before any training.

    Raises AssertionError with a clear message if any leak is detected.
    Never modifies data.
    """
    MUST_NOT_EXIST = [
        "temp_c_next_1h",      # target — must be in y, never in X
        "output_temp",          # concurrent temp sensor (OT column)
        "temp_potential_k",     # mathematical transform of temp_c
        "temp_logged",          # logger reading of temp_c
        "air_density",          # algebraic function of temp_c
        "dew_depression",       # temp_c − dew_point = encodes temp_c
        "vapor_pressure_max",   # Clausius-Clapeyron of temp_c
        "vapor_pressure_def",   # VPmax − VPact, inherits VPmax's leakage
        "specific_humidity",    # function of dew_point → temp chain
        "h2o_concentration",    # compound of specific_humidity + air_density
    ]

    MUST_EXIST = [
        "temp_c",               # current temperature — legitimate feature
        "temp_c_lag_1h",        # 1-hour lag — key temporal predictor
        "pressure_mbar",        # independent atmospheric measurement
    ]

    feature_cols = [c for c in df.columns if c != "temp_c_next_1h"]

    leaks_found  = [c for c in MUST_NOT_EXIST if c in feature_cols]
    missing_keys = [c for c in MUST_EXIST if c not in df.columns]

    if leaks_found:
        raise AssertionError(
            f"\n🚨 LEAKAGE DETECTED in feature matrix:\n"
            f"   Present (must be absent): {leaks_found}\n"
            f"   Re-run notebooks/03_features.ipynb after checking transforms.py"
        )

    if missing_keys:
        raise AssertionError(
            f"\n🚨 EXPECTED COLUMNS MISSING from feature matrix:\n"
            f"   Missing: {missing_keys}\n"
            f"   Check build_feature_matrix() in transforms.py"
        )

    logger.info("✅ Leakage check passed — feature matrix is clean")
    logger.info(f"   Total columns   : {len(df.columns)}")
    logger.info(f"   Feature columns : {len(feature_cols)}")
    logger.info(f"   Target          : temp_c_next_1h")


def ridge_coefficients(
    df: pd.DataFrame,
    target_col: str = "temp_c_next_1h",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Fit Ridge on the full dataset and return sorted coefficients.

    Used in notebooks to verify that temp_c_lag_1h dominates and
    no leakage column has an outsized coefficient.

    HEALTHY OUTPUT:
      temp_c           coefficient ≈ 0.85–0.95  (dominant)
      temp_c_lag_1h    coefficient ≈ 0.05–0.12
      dew_point_c      coefficient ≈ small, positive
      temp_c_next_1h   absent entirely

    RED FLAGS:
      dew_point_c      > 1.0    → physics leakage still present
      air_density      dominant  → physics leakage still present
      temp_c_next_1h   present   → target leaked into features
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    feature_cols = [c for c in df.columns if c != target_col]
    X = StandardScaler().fit_transform(df[feature_cols].values)
    y = df[target_col].values

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    coef_df = (
        pd.DataFrame({
            "feature":     feature_cols,
            "coefficient": ridge.coef_,
            "abs_coef":    np.abs(ridge.coef_),
        })
        .sort_values("abs_coef", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return coef_df