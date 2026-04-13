"""
src/data/loader.py — Centralized data loading.

PROFESSIONAL PRINCIPLE:
  Never scatter `pd.read_csv()` calls across your project.
  If the file format changes (e.g., CSV to Parquet) or the date column
  changes name, you only want to update the code in ONE place.
"""
import pandas as pd
import logging
from src.config import cfg

# Set up logging so we can trace data loading in production
logger = logging.getLogger(__name__)

def load_raw() -> pd.DataFrame:
    """
    Loads the raw weather dataset from disk.
    Parses the date column and sets it as a DatetimeIndex.
    """
    logger.info(f"Loading raw data from: {cfg.raw_data_path}")
    
    try:
        df = pd.read_csv(
            cfg.raw_data_path,
            parse_dates=[cfg.date_column],
            index_col=cfg.date_column,
            encoding="utf-8",
            encoding_errors="replace"  # Handles encoding anomalies like µ and ² in col names
        )
        logger.info(f"Successfully loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
        return df
        
    except FileNotFoundError:
        logger.error(f"FATAL: Raw data file not found at {cfg.raw_data_path}")
        logger.error("Please ensure your weather.csv is placed in the data/raw/ directory.")
        raise

def load_processed() -> pd.DataFrame:
    """
    Loads cleaned (processed) data from disk.
    """
    logger.info(f"Loading processed data from: {cfg.processed_data_path}")
    
    try:
        df = pd.read_parquet(cfg.processed_data_path)
        logger.info(f"Loaded processed data: {df.shape}")
        return df
    
    except FileNotFoundError:
        logger.error(f"Processed file not found at {cfg.processed_data_path}")
        raise

def load_features() -> pd.DataFrame:
    """
    Load feature-engineered dataset.
    """
    logger.info(f"Loading features from: {cfg.features_data_path}")
    
    try:
        df = pd.read_parquet(cfg.features_data_path)
        logger.info(f"Loaded features: {df.shape}")
        return df
    
    except FileNotFoundError:
        logger.error(f"Features file not found at {cfg.features_data_path}")
        raise