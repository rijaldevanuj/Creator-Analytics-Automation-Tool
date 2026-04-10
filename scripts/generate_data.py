"""
Synthetic Data Generation Pipeline

Generates synthetic creator content performance data for testing and development.
Creates realistic engagement metrics based on platform and content type distributions.

Environment Variables:
    PROJECT_ROOT: Base project directory (defaults to script parent's parent directory)
    LOG_LEVEL: Logging level (default: INFO)
    RANDOM_SEED: Random seed for reproducibility (optional)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import os
import random

import pandas as pd
import numpy as np

# ===========================
# Configuration and Logging
# ===========================

# Setup paths
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent))
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_FILE = DATA_DIR / 'raw_data.csv'

# Setup logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'generate_data.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility (if specified)
if os.getenv('RANDOM_SEED'):
    random.seed(int(os.getenv('RANDOM_SEED')))
    np.random.seed(int(os.getenv('RANDOM_SEED')))
    logger.info(f"Random seed set to {os.getenv('RANDOM_SEED')}")

# Configuration
CONFIG = {
    'num_rows': int(os.getenv('NUM_ROWS', 1200)),
    'start_date': datetime(2025, 1, 1),
    'date_range_days': 180,
    'platforms': ['Instagram', 'YouTube'],
    'content_types': {
        'Instagram': ['Reel', 'Post', 'Story'],
        'YouTube': ['Video', 'Short']
    },
    'base_views_range': (2000, 50000),
    'virality_multiplier': (1.5, 3),  # For Reels/Shorts
    'engagement_ranges': {
        'likes': (0.05, 0.15),
        'comments_ratio': (0.05, 0.2),
        'shares_ratio': (0.02, 0.1),
        'watch_time_ratio': (0.3, 1.2),
        'followers_ratio': (0.005, 0.02)
    }
}


def generate_metrics(views: int, content_type: str) -> dict:
    """
    Generate realistic engagement metrics based on content type.
    
    Args:
        views: Number of views
        content_type: Type of content (Reel, Post, Video, etc.)
        
    Returns:
        Dictionary with engagement metrics
    """
    engagement_ranges = CONFIG['engagement_ranges']
    
    likes = int(views * random.uniform(*engagement_ranges['likes']))
    comments = int(likes * random.uniform(*engagement_ranges['comments_ratio']))
    shares = int(likes * random.uniform(*engagement_ranges['shares_ratio']))
    watch_time = int(views * random.uniform(*engagement_ranges['watch_time_ratio']))
    followers_gained = int(views * random.uniform(*engagement_ranges['followers_ratio']))
    
    return {
        'likes': max(0, likes),
        'comments': max(0, comments),
        'shares': max(0, shares),
        'watch_time': max(0, watch_time),
        'followers_gained': max(0, followers_gained)
    }


def calculate_views(content_type: str, base_views: int) -> int:
    """
    Calculate views based on content type.
    Reels and Shorts have higher virality potential.
    
    Args:
        content_type: Type of content
        base_views: Base view count
        
    Returns:
        Adjusted view count
    """
    if content_type in ['Reel', 'Short']:
        return int(base_views * random.uniform(*CONFIG['virality_multiplier']))
    return base_views


def generate_synthetic_data(num_rows: int) -> pd.DataFrame:
    """
    Generate synthetic creator content performance data.
    
    Args:
        num_rows: Number of data rows to generate
        
    Returns:
        DataFrame with synthetic data
    """
    logger.info(f"Generating {num_rows} synthetic data rows")
    
    try:
        data = []
        start_date = CONFIG['start_date']
        platforms = CONFIG['platforms']
        content_types = CONFIG['content_types']
        base_views_range = CONFIG['base_views_range']
        date_range = CONFIG['date_range_days']
        
        for i in range(num_rows):
            # Select platform and content type
            platform = random.choice(platforms)
            content_type = random.choice(content_types[platform])
            
            # Generate date
            date = start_date + timedelta(days=random.randint(0, date_range))
            
            # Generate views and engagement metrics
            base_views = random.randint(*base_views_range)
            views = calculate_views(content_type, base_views)
            metrics = generate_metrics(views, content_type)
            
            # Append row
            data.append([
                platform,
                date.strftime("%Y-%m-%d"),
                content_type,
                views,
                metrics['likes'],
                metrics['comments'],
                metrics['shares'],
                metrics['watch_time'],
                metrics['followers_gained']
            ])
            
            # Progress logging (every 10%)
            if (i + 1) % (num_rows // 10 or 1) == 0:
                logger.debug(f"Generated {i + 1}/{num_rows} rows")
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[
            'platform',
            'date',
            'content_type',
            'views',
            'likes',
            'comments',
            'shares',
            'watch_time',
            'followers_gained'
        ])
        
        logger.info(f"Successfully generated {len(df)} rows of synthetic data")
        logger.info(f"Platform distribution:\n{df['platform'].value_counts().to_string()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}", exc_info=True)
        raise


def save_data(df: pd.DataFrame, output_path: Path) -> bool:
    """
    Save generated data to CSV file.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        logger.info(f"Output shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Log basic statistics
        logger.info(f"Views range: {df['views'].min()} - {df['views'].max()}")
        logger.info(f"Average engagement rate: {((df['likes'] + df['comments'] + df['shares']) / df['views']).mean():.4f}")
        
        return True
        
    except IOError as e:
        logger.error(f"Error saving data: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while saving: {e}", exc_info=True)
        return False


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Synthetic Data Generation Pipeline Started")
    logger.info("=" * 60)
    
    try:
        # Generate data
        num_rows = CONFIG['num_rows']
        df = generate_synthetic_data(num_rows)
        
        # Save data
        if not save_data(df, OUTPUT_FILE):
            logger.error("Failed to save generated data")
            return False
        
        logger.info("=" * 60)
        logger.info("✅ Data generation pipeline completed successfully")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.error("=" * 60)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

