"""
Data Cleaning and Feature Engineering Pipeline

This script loads raw content data, performs cleaning, validation, and feature engineering
for the Creator Intelligence ML pipeline. Suitable for production deployment.

Environment Variables:
    PROJECT_ROOT: Base project directory (defaults to script parent's parent directory)
    LOG_LEVEL: Logging level (default: INFO)
"""

import logging
import sys
from pathlib import Path
import os

import pandas as pd

# ===========================
# Configuration and Logging
# ===========================

# Setup paths
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent))
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_FILE = DATA_DIR / 'raw_data.csv'
CLEANED_DATA_FILE = DATA_DIR / 'cleaned_data.csv'

# Setup logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'data_cleaning.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'numeric_cols': ['views', 'likes', 'comments', 'shares', 'watch_time', 'followers_gained'],
    'date_col': 'date',
    'min_views': 0,
    'engagement_categories': 3,  # Number of quantile bins
    'virality_weights': {'likes': 2, 'comments': 3, 'shares': 5}
}


def validate_input_file(file_path: Path) -> bool:
    """Check if input file exists and is readable."""
    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        return False
    if not file_path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return False
    return True


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw data from CSV file with error handling."""
    try:
        logger.info(f"Loading raw data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Perform basic data cleaning and validation.
    
    Args:
        df: Raw dataframe
        config: Configuration dictionary
        
    Returns:
        Cleaned dataframe
    """
    logger.info("Starting data cleaning process")
    initial_rows = len(df)
    
    # Convert date column
    try:
        df[config['date_col']] = pd.to_datetime(df[config['date_col']], errors='coerce')
        logger.info("Date column converted successfully")
    except KeyError:
        logger.error(f"Date column '{config['date_col']}' not found in data")
        raise
    
    # Remove invalid dates
    invalid_dates = df[config['date_col']].isna().sum()
    df = df.dropna(subset=[config['date_col']])
    if invalid_dates > 0:
        logger.warning(f"Removed {invalid_dates} rows with invalid dates")
    
    # Validate and convert numeric columns
    missing_cols = [col for col in config['numeric_cols'] if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing numeric columns: {missing_cols}")
        raise ValueError(f"Required columns missing: {missing_cols}")
    
    df[config['numeric_cols']] = df[config['numeric_cols']].apply(pd.to_numeric, errors='coerce')
    logger.info("Numeric columns converted")
    
    # Fill missing numeric values
    null_counts = df[config['numeric_cols']].isna().sum()
    df[config['numeric_cols']] = df[config['numeric_cols']].fillna(0)
    if null_counts.sum() > 0:
        logger.warning(f"Filled {null_counts.sum()} null values with 0")
    
    # Remove rows with zero or negative views
    df = df[df['views'] > config['min_views']]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} rows with invalid view counts")
    
    logger.info(f"Data cleaning complete. Final dataset: {len(df)} rows")
    return df


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Create derived features for ML models.
    
    Args:
        df: Cleaned dataframe
        config: Configuration dictionary
        
    Returns:
        Dataframe with engineered features
    """
    logger.info("Starting feature engineering")
    
    try:
        # Engagement Rate
        df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']) / df['views']
        
        # Virality Score (weighted)
        weights = config['virality_weights']
        df['virality_score'] = (
            (weights['likes'] * df['likes'] + 
             weights['comments'] * df['comments'] + 
             weights['shares'] * df['shares']) / df['views']
        )
        
        # Watch Time per View
        df['watch_time_per_view'] = df['watch_time'] / df['views']
        
        # Follower Conversion Rate
        df['follower_conversion_rate'] = df['followers_gained'] / df['views']
        
        logger.info("Engagement and performance metrics calculated")
        
        # Time Features
        df['day_of_week'] = df[config['date_col']].dt.day_name()
        df['month'] = df[config['date_col']].dt.month
        df['week'] = df[config['date_col']].dt.isocalendar().week
        
        logger.info("Time features extracted")
        
        # Content Performance Tagging
        df['performance_category'] = pd.qcut(
            df['engagement_rate'],
            q=config['engagement_categories'],
            labels=['Low', 'Medium', 'High'],
            duplicates='drop'
        )
        
        logger.info("Performance categories assigned")
        return df
        
    except Exception as e:
        logger.error(f"Feature engineering error: {e}")
        raise


def save_cleaned_data(df: pd.DataFrame, file_path: Path) -> bool:
    """
    Save cleaned data to CSV file.
    
    Args:
        df: Processed dataframe
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(file_path, index=False)
        logger.info(f"Cleaned data saved to {file_path}")
        logger.info(f"Output shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return True
        
    except IOError as e:
        logger.error(f"Error saving data: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while saving: {e}")
        return False


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Data Cleaning Pipeline Started")
    logger.info("=" * 60)
    
    try:
        # Validate input
        if not validate_input_file(RAW_DATA_FILE):
            logger.error("Input validation failed. Exiting.")
            return False
        
        # Load, clean, and engineer
        df = load_raw_data(RAW_DATA_FILE)
        df = clean_data(df, CONFIG)
        df = engineer_features(df, CONFIG)
        
        # Save results
        if not save_cleaned_data(df, CLEANED_DATA_FILE):
            logger.error("Failed to save cleaned data")
            return False
        
        logger.info("=" * 60)
        logger.info("✅ Data pipeline completed successfully")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.error("=" * 60)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

