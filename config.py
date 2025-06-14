import os
import numpy as np

PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT","8428"))
STEP_SECONDS = int(os.getenv("STEP_SECONDS", "60"))
RETRAIN_THRESHOLD = 500  # Adjust based on your data volume and retraining frequency needs
RETRAIN_COOLDOWN = 3600  # seconds, e.g., 1 hour cooldown between retrains per pod

PROMETHEUS_DOMAIN = os.getenv("PROMETHEUS_DOMAIN", "http://localhost:")
PROMETHEUS_URL = f"{PROMETHEUS_DOMAIN}{PROMETHEUS_PORT}"
HISTORY_FILE =os.getenv("HISTORY_FILE", "../data/live_data/history.pkl")
MODELS_PATH = os.getenv("MODELS_PATH", "../data/models")
LOG_FILE  =  os.getenv("LOG_FILE", "./live_data")


LEARNING_DAYS = int(os.getenv("LEARNING_DAYS", "10"))
HISTORICAL_CPU_USAGE_FALLBACK = [np.random.uniform(0.1, 0.9) for _ in range(50)]  # Fallback sequence length
ENABLE_RETRAINING = os.getenv("ENABLE_RETRAINING", "false").lower() == "true"
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "500"))  # Min samples needed

# VictoriaMetrics / Prometheus URL

