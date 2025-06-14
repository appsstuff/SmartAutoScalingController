import os
import numpy as np
from prometheus_client import Gauge

PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT","8428"))
STEP_SECONDS = int(os.getenv("STEP_SECONDS", "60"))
RETRAIN_THRESHOLD = 500  # Adjust based on your data volume and retraining frequency needs
RETRAIN_COOLDOWN = 3600  # seconds, e.g., 1 hour cooldown between retrains per pod

PROMETHEUS_DOMAIN = os.getenv("PROMETHEUS_DOMAIN", "http://34.73.82.105:")
PROMETHEUS_URL = f"{PROMETHEUS_DOMAIN}{PROMETHEUS_PORT}"

HISTORY_FILE =os.getenv("MODEL_PATH", "./data/live_data/history.pkl")
 
LEARNING_DAYS = int(os.getenv("LEARNING_DAYS", "10"))
HISTORICAL_CPU_USAGE_FALLBACK = [np.random.uniform(0.1, 0.9) for _ in range(50)]  # Fallback sequence length
ENABLE_RETRAINING = os.getenv("ENABLE_RETRAINING", "false").lower() == "true"
MODELS_PATH = os.getenv("MODEL_PATH", "./data/models")

# VictoriaMetrics / Prometheus URL


# Define custom metrics
DECISION_GAUGE = Gauge('autoscaler_decision', 'Last autoscaler decision', ['service'])
CPU_USAGE_GAUGE = Gauge('autoscaler_cpu_usage', 'Current CPU usage from feature vector', ['service'])
MEM_USAGE_GAUGE = Gauge('autoscaler_mem_usage', 'Current memory usage (MB)', ['service'])
REQ_RATE_GAUGE = Gauge('autoscaler_req_rate', 'Requests per second', ['service'])
MODEL_CONFIDENCE_GAUGE = Gauge('autoscaler_model_confidence', 'Model prediction confidence', ['service'])

