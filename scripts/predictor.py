import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import logging
import sys

# ----------------------------
# LOGGING SETUP
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'predictor.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# PATH SETUP (ROBUST)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')
model_path = os.path.join(BASE_DIR, 'model.pkl')
logs_dir = os.path.join(BASE_DIR, 'logs')

# Create logs directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"Data path: {data_path}")
logger.info(f"Model path: {model_path}")

# ----------------------------
# LOAD DATA
# ----------------------------
try:
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    logger.error(f"Failed to load data: {str(e)}")
    sys.exit(1)

# ----------------------------
# FEATURES & TARGET
# ----------------------------
features = ['views', 'likes', 'comments', 'shares', 'watch_time', 'followers_gained']
target = 'engagement_rate'

# Validate required columns exist
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    logger.error(f"Missing required columns: {missing_cols}")
    logger.info(f"Available columns: {list(df.columns)}")
    sys.exit(1)

# Check for missing values
if df[features + [target]].isnull().any().any():
    logger.warning("Missing values detected. Dropping rows with NaN...")
    df = df.dropna(subset=features + [target])
    logger.info(f"Data shape after removing NaN: {df.shape}")

X = df[features]
y = df[target]

logger.info(f"Features shape: {X.shape}")
logger.info(f"Target shape: {y.shape}")

# ----------------------------
# TRAIN MODEL
# ----------------------------
try:
    logger.info("Starting model training...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Model training completed")

    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Training R² Score: {train_score:.4f}")
    logger.info(f"Testing R² Score: {test_score:.4f}")
    logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logger.info(f"R² Score: {r2:.4f}")

except Exception as e:
    logger.error(f"Model training failed: {str(e)}")
    sys.exit(1)

# ----------------------------
# SAVE MODEL
# ----------------------------
try:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✅ Model trained and saved at: {model_path}")
except Exception as e:
    logger.error(f"Failed to save model: {str(e)}")
    sys.exit(1)

logger.info("Pipeline completed successfully!")