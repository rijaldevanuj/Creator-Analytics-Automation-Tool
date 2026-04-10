"""
Content Performance Analysis Pipeline

This script analyzes content metrics from the Creator Intelligence database,
generating insights on best performing content types, platforms, posting times,
and top-performing posts. Suitable for production deployment.

Environment Variables:
    PROJECT_ROOT: Base project directory (defaults to script parent's parent directory)
    DATABASE_PATH: Path to SQLite database (defaults to PROJECT_ROOT/database/db.sqlite3)
    LOG_LEVEL: Logging level (default: INFO)
"""

import logging
import sys
import sqlite3
import os
from pathlib import Path

import pandas as pd

# ===========================
# Configuration and Logging
# ===========================

# Setup paths
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent))
DATABASE_PATH = Path(os.getenv('DATABASE_PATH', PROJECT_ROOT / 'database' / 'db.sqlite3'))
LOGS_DIR = PROJECT_ROOT / 'logs'

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / 'analysis.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Analysis Queries Configuration
ANALYSIS_QUERIES = {
    'best_content_type': """
        SELECT content_type, AVG(engagement_rate) as avg_engagement
        FROM content_metrics
        GROUP BY content_type
        ORDER BY avg_engagement DESC
        LIMIT 1;
    """,
    'platform_performance': """
        SELECT platform, AVG(engagement_rate) as avg_engagement
        FROM content_metrics
        GROUP BY platform
        ORDER BY avg_engagement DESC;
    """,
    'best_posting_day': """
        SELECT day_of_week, AVG(engagement_rate) as avg_engagement
        FROM content_metrics
        GROUP BY day_of_week
        ORDER BY avg_engagement DESC
        LIMIT 1;
    """,
    'top_posts': """
        SELECT platform, content_type, views, engagement_rate
        FROM content_metrics
        ORDER BY engagement_rate DESC
        LIMIT 5;
    """
}


# ===========================
# Analysis Functions
# ===========================

def run_analysis():
    """Execute content performance analysis and log results."""
    try:
        logger.info(f"Connecting to database: {DATABASE_PATH}")
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Execute all queries
        logger.info("Running content performance analysis queries")
        
        best_content = pd.read_sql(ANALYSIS_QUERIES['best_content_type'], conn)
        platform_perf = pd.read_sql(ANALYSIS_QUERIES['platform_performance'], conn)
        best_day = pd.read_sql(ANALYSIS_QUERIES['best_posting_day'], conn)
        top_posts = pd.read_sql(ANALYSIS_QUERIES['top_posts'], conn)
        
        conn.close()
        logger.info("Database connection closed successfully")
        
        # Display insights
        logger.info("Analysis complete. Displaying results.")
        print("\n" + "="*50)
        print("🔥 BEST CONTENT TYPE:")
        print("="*50)
        print(best_content)
        
        print("\n" + "="*50)
        print("📊 PLATFORM PERFORMANCE:")
        print("="*50)
        print(platform_perf)
        
        print("\n" + "="*50)
        print("📅 BEST DAY TO POST:")
        print("="*50)
        print(best_day)
        
        print("\n" + "="*50)
        print("🚀 TOP POSTS:")
        print("="*50)
        print(top_posts)
        
        logger.info("Analysis results displayed successfully")
        return {
            'best_content': best_content,
            'platform_performance': platform_perf,
            'best_day': best_day,
            'top_posts': top_posts
        }
        
    except FileNotFoundError:
        logger.error(f"Database file not found: {DATABASE_PATH}")
        raise
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        raise


# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":
    logger.info("Starting content performance analysis")
    try:
        results = run_analysis()
        logger.info("Analysis completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Analysis failed: {e}")
        sys.exit(1)



