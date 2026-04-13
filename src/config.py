# src/config.py
"""
Single source of truth for all project settings.

Professional rule: if a value appears in more than one file,
it belongs in config.py. No exceptions.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

# Resolve the absolute project root regardless of where Python is called from.
# __file__ is this config.py → .parent is src/ → .parent is the project root.
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── Environment ───────────────────────────────────────────────────────
    environment: str = "development"
    debug: bool = True

    # ── Data paths ────────────────────────────────────────────────────────
    raw_data_path: Path       = ROOT / "data" / "raw"       / "weather.csv"
    processed_data_path: Path = ROOT / "data" / "processed" / "weather_clean.parquet"
    features_data_path: Path  = ROOT / "data" / "features"  / "weather_features.parquet"

    # ── Artifacts ─────────────────────────────────────────────────────────
    model_path: Path    = ROOT / "artifacts" / "model.pkl"
    model_version: str  = "v1.0.0"

    # ── Dataset facts discovered during EDA ───────────────────────────────
    # These aren't arbitrary — they come from Phase 2 exploration.
    # Putting them here means every module agrees on the same values.
    date_column: str       = "date"
    target_column: str     = "T (degC)"   # what we forecast
    sentinel_value: float  = -9999.0      # sensor error code in the data
    resample_freq: str     = "1h"         # 10-min → hourly aggregation

    # ── Feature engineering ───────────────────────────────────────────────
    lag_hours: list[int]      = [1, 2, 3, 6, 12, 24, 48]
    rolling_windows: list[int] = [3, 6, 12, 24]

    # ── Model training ────────────────────────────────────────────────────
    test_size: float    = 0.2
    random_state: int   = 42
    cv_folds: int       = 5

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str         = "0.0.0.0"
    api_port: int         = 8000
    api_key: str          = "dev-key-change-in-production"
    allowed_origins: list[str] = [
        "http://localhost:8501",   # Streamlit dashboard
        "http://localhost:3000",   # React (if you ever switch)
    ]

    # Reads from .env file automatically — overrides defaults above
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """
    Always call this — never instantiate Settings() directly.
    lru_cache means it's created once, reused everywhere.
    This is the Dependency Injection pattern FastAPI also uses.
    """
    return Settings()


# Short alias for notebooks and scripts:
#   from src.config import cfg
cfg = get_settings()