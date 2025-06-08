# config.py
# Centralized configuration for deployment settings like namespace, target deployment, scaling limits, feature names, etc.
    # Stores Kubernetes deployment info
    # Defines model input features
    # Prevents hardcoded values in main logic

# Namespace where the deployment is running
import os

NAMESPACE = os.getenv("NAMESPACE", "default")

TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "adservice")
MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "2"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "10"))
SCALING_STEP = int(os.getenv("SCALING_STEP", "1"))
SEQ_LENGTH = int(os.getenv("SEQ_LENGTH", "10"))

FEATURE_NAMES = [
    "hour_of_day", 
    "day_of_week", 
    "cpu_usage_lag_1", 
    "cpu_usage_lag_5",
    "cpu_roll_mean_10",
    "mem_usage",
    "req_rate"
]