"""
Data Loading to SQLite Pipeline

Loads cleaned content metrics data into SQLite database with proper indexing
and data integrity checks. Suitable for production deployment.

Environment Variables:
    PROJECT_ROOT: Base project directory (defaults to script parent's parent directory)
    LOG_LEVEL: Logging level (default: INFO)
    DB_NAME: Database name (default: db.sqlite3)
    TABLE_NAME: Table name to load data into (default: content_metrics)
    BACKUP_DB: Whether to backup existing database (default: True)
"""

import logging
import sys
import sqlite3
from pathlib import Path
from datetime import datetime
import os

import pandas as pd

# ===========================
# Configuration and Logging
# ===========================

# Setup paths
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent))
DATA_DIR = PROJECT_ROOT / 'data'
DATABASE_DIR = PROJECT_ROOT / 'database'
CLEANED_DATA_FILE = DATA_DIR / 'cleaned_data.csv'
DB_FILE = DATABASE_DIR / os.getenv('DB_NAME', 'db.sqlite3')

# Setup logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'load_to_sql.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'table_name': os.getenv('TABLE_NAME', 'content_metrics'),
    'backup_db': os.getenv('BACKUP_DB', 'True').lower() == 'true',
    'batch_size': 500,
    'timeout': 30,
    'indexes': [
        ('idx_platform', 'platform'),
        ('idx_date', 'date'),
        ('idx_content_type', 'content_type'),
        ('idx_engagement_rate', 'engagement_rate')
    ]
}


def validate_input_file(file_path: Path) -> bool:
    """Check if input CSV file exists and is readable."""
    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        return False
    if not file_path.is_file():
        logger.error(f"Path is not a file: {file_path}")
        return False
    logger.info(f"Input file validated: {file_path}")
    return True


def load_cleaned_data(file_path: Path) -> pd.DataFrame:
    """
    Load cleaned data from CSV file with validation.
    
    Args:
        file_path: Path to cleaned CSV file
        
    Returns:
        DataFrame with cleaned data
    """
    try:
        logger.info(f"Loading cleaned data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Validate required columns
        required_cols = ['date', 'platform', 'views', 'likes', 'comments', 'shares']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")
        
        return df
        
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def backup_database(db_path: Path) -> bool:
    """
    Create backup of existing database before replacement.
    
    Args:
        db_path: Path to database file
        
    Returns:
        True if backup successful or no existing DB, False on error
    """
    if not db_path.exists():
        logger.info("No existing database to backup")
        return True
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
        
        # Use SQLite backup API for safe backup
        source_conn = sqlite3.connect(str(db_path), timeout=CONFIG['timeout'])
        backup_conn = sqlite3.connect(str(backup_path))
        
        with backup_conn:
            source_conn.backup(backup_conn)
        
        source_conn.close()
        backup_conn.close()
        
        logger.info(f"Database backed up to {backup_path}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database backup error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during backup: {e}")
        return False


def connect_database(db_path: Path) -> sqlite3.Connection:
    """
    Establish connection to SQLite database.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Database connection object
    """
    try:
        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path), timeout=CONFIG['timeout'])
        conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database: {db_path}")
        return conn
        
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise


def load_data_to_database(df: pd.DataFrame, conn: sqlite3.Connection, table_name: str) -> bool:
    """
    Load DataFrame into SQLite database table.
    
    Args:
        df: DataFrame to load
        conn: Database connection
        table_name: Target table name
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Loading {len(df)} rows into table '{table_name}'")
        
        # Load data with replace strategy to ensure clean data
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Verify load
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        logger.info(f"Successfully loaded {row_count} rows into '{table_name}'")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Database insert error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False


def create_indexes(conn: sqlite3.Connection, table_name: str, indexes: list) -> bool:
    """
    Create indexes on specified columns for query optimization.
    
    Args:
        conn: Database connection
        table_name: Target table name
        indexes: List of (index_name, column_name) tuples
        
    Returns:
        True if successful, False otherwise
    """
    cursor = conn.cursor()
    
    try:
        logger.info("Creating indexes")
        
        for index_name, column_name in indexes:
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name});"
            )
            logger.debug(f"Index created: {index_name} on {table_name}({column_name})")
        
        conn.commit()
        logger.info(f"Successfully created {len(indexes)} indexes")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Index creation error: {e}")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        conn.rollback()
        return False


def get_database_statistics(conn: sqlite3.Connection, table_name: str) -> dict:
    """
    Retrieve statistics about loaded data.
    
    Args:
        conn: Database connection
        table_name: Table name
        
    Returns:
        Dictionary with statistics
    """
    try:
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(DISTINCT platform) FROM {table_name}")
        unique_platforms = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(DISTINCT content_type) FROM {table_name}")
        unique_content_types = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table_name}")
        date_range = cursor.fetchone()
        
        stats = {
            'total_rows': total_rows,
            'unique_platforms': unique_platforms,
            'unique_content_types': unique_content_types,
            'date_range': f"{date_range[0]} to {date_range[1]}" if date_range[0] else "N/A"
        }
        
        return stats
        
    except Exception as e:
        logger.warning(f"Could not retrieve statistics: {e}")
        return {}


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("Data Loading to SQL Pipeline Started")
    logger.info("=" * 60)
    
    conn = None
    
    try:
        # Validate input file
        if not validate_input_file(CLEANED_DATA_FILE):
            logger.error("Input validation failed. Exiting.")
            return False
        
        # Backup existing database if configured
        if CONFIG['backup_db']:
            if not backup_database(DB_FILE):
                logger.warning("Database backup failed, continuing anyway")
        
        # Load data
        df = load_cleaned_data(CLEANED_DATA_FILE)
        
        # Connect to database
        conn = connect_database(DB_FILE)
        
        # Load data into database
        if not load_data_to_database(df, conn, CONFIG['table_name']):
            logger.error("Failed to load data into database")
            return False
        
        # Create indexes
        if not create_indexes(conn, CONFIG['table_name'], CONFIG['indexes']):
            logger.error("Failed to create indexes")
            return False
        
        # Get statistics
        stats = get_database_statistics(conn, CONFIG['table_name'])
        if stats:
            logger.info(f"Database Statistics:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("=" * 60)
        logger.info("✅ Data loading pipeline completed successfully")
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.error("=" * 60)
        return False
        
    finally:
        # Ensure database connection is properly closed
        if conn:
            try:
                conn.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)



