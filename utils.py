import os
import numpy as np
import pandas as pd
import joblib
import requests
import logging
from logging_loki import LokiHandler
from datetime import datetime, timedelta

# Global history dict
history = {}    
# VictoriaMetrics / Prometheus URL
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://vm-victoria-metrics-single-server.monitoring.svc.cluster.local:8428/api/v1/query")
GRAFANA_URL    = os.getenv("GRAFANA_URL", "http://grafana.monitoring.svc:3000")
LOKI_URL           = os.getenv("LOKI_URL", "http://loki.monitoring.svc:3100/loki/api/v1/push")


STEP_SECONDS = int(os.getenv("STEP_SECONDS", "60"))
HISTORY_FILE = "/data/history.pkl"
os.makedirs("/data", exist_ok=True)
LEARNING_DAYS  = os.getenv("LEARNING_DAYS",10)
HISTORICAL_CPU_USAGE_FALLBACK = [np.random.uniform(0.1, 0.9) for _ in range(50)]  # Fallback sequence length

def load_history():
    """Load saved history from disk if exists"""
    if os.path.exists(HISTORY_FILE):
        print(" Loading saved history...")
        return joblib.load(HISTORY_FILE)
    print("🆕 Starting with empty history")
    return {}

def save_history(history):
    """
    Save history to disk for future use
    """
    try:
        os.makedirs("/data", exist_ok=True)  # Double-check
        joblib.dump(history, HISTORY_FILE)
        print(f"💾 History saved to {HISTORY_FILE}")
    except Exception as e:
        print(f" Failed to save history: {e}")

def query_vm(query):
    try:
        response = requests.get(PROMETHEUS_URL, params={'query': query}, timeout=5)
        
        if response.status_code == 200:
            result = response.json().get('data', {}).get('result', [])
            if result:
                return float(result[0]['value'][1])
    except Exception as e:
        print(f" VM query failed: {e}")
    return np.random.uniform(0.1, 0.9)

def fetch_pod_metrics(pod_name="adservice", namespace="default"):
    """
    Fetch live metrics from VictoriaMetrics or Prometheus
    """
    queries = {
        "hour_of_day": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "cpu_usage_lag_1": query_vm(f'container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}}'),
        "cpu_usage_lag_5": query_vm(f'container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}} offset 5m'),
        "cpu_roll_mean_10": query_vm(f'avg_over_time(container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}}[10m])'),
        "mem_usage": query_vm(f'container_memory_usage_bytes{{namespace="{namespace}", container_name="{pod_name}"}}') / (1024 * 1024),
        "req_rate": query_vm(f'rate(http_requests_total{{namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])'),
        "latency": query_vm(f'histogram_quantile(0.95, sum(rate(http_request_latencies_bucket{{le="+Inf", namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])) by (le))'),
        "net_receive_KB": query_vm(f'rate(container_network_receive_bytes_total{{namespace="{namespace}", container_name="{pod_name}"}}[1m])') * 1024,
        "net_transmit_KB": query_vm(f'rate(container_network_transmit_bytes_total{{namespace="{namespace}", container_name="{pod_name}"}}[1m])') * 1024,
        "pod_restarts": query_vm(f'kube_pod_container_status_restarts_total{{namespace="{namespace}", container="{pod_name}"}}'),
        "pod_ready": query_vm(f'kube_pod_container_status_ready{{namespace="{namespace}", container="{pod_name}"}}')
    }
    
    return queries

def build_feature_vector(metrics, feature_list):
    return np.array([metrics[f] for f in feature_list if f in metrics])

def create_sequence(data, seq_length):
    if len(data) < seq_length:
        return np.array([])
    return np.array([data[i:i+seq_length] for i in range(len(data) - seq_length + 1)])

def record_live_data(pod_name, input_row, decision):
    log_dir = "/data/live_data"
    os.makedirs(log_dir, exist_ok=True)

    if isinstance(input_row, dict):
        metric_values = list(input_row.values())
        columns = list(input_row.keys())
    else:
        metric_values = input_row.flatten().tolist()
        columns = ["hour_of_day", "day_of_week", "cpu_usage_lag_1", "cpu_usage_lag_5", "cpu_roll_mean_10", "mem_usage", "req_rate"]

    df = pd.DataFrame([metric_values], columns=columns)
    df["decision"] = decision
    df["timestamp"] = datetime.now().isoformat()

    log_path = os.path.join(log_dir, f"{pod_name}_live_data.csv")
    df.to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))

def validate_service_config(svc):
    """Validate service dict has required keys"""
    if "pod_name" not in svc:
        raise KeyError("Missing 'pod_name'")
    if "feature_names" not in svc:
        raise KeyError("Missing 'feature_names'")
    if "seq_length" not in svc:
        raise KeyError("Missing 'seq_length'")
    
    
def fetch_historical_data(pod_name, namespace, seq_length=50, days=LEARNING_DAYS):
    """
    Query historical CPU usage from VictoriaMetrics or Prometheus.
    
    Args:
        pod_name (str): Name of the service/pod
        namespace (str): Kubernetes namespace
        seq_length (int): Required number of samples for model input
        days (int): How many days of history to query
    
    Returns:
        list: List of CPU usage samples (length = seq_length)
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    start = int(start_time.timestamp())
    end = int(end_time.timestamp())

    query = f'container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}}'
    params = {
        'query': query,
        'start': start,
        'end': end,
        'step': STEP_SECONDS
    }

    try:
        response = requests.get(PROMETHEUS_URL, params=params, timeout=5)
        if response.status_code == 200:
            result = response.json().get('data', {}).get('result', [])
            if not result:
                raise ValueError("No data returned from Prometheus")

            values = result[0].get('values', [])
            cpu_values = [float(v[1]) for v in values]

            if len(cpu_values) < seq_length:
                print(f" Only {len(cpu_values)} samples found — using fallback")
                return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]

            print(f"📊 Loaded {len(cpu_values)} historical entries for {pod_name}")
            return cpu_values[-seq_length:]

    except Exception as e:
        print(f" Historical fetch failed: {e}")

    # Use random fallback if all else fails
    print(" Using simulated history (no Prometheus/VictoriaMetrics data)")
    return [np.random.uniform(0.1, 0.9) for _ in range(seq_length)]


def send_grafana_annotation(message, tags=None):
    API_KEY = os.getenv("GRAFANA_API_KEY", "your_token_here")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "text": message,
        "tags": tags or [],
        "time": int(time.time() * 1000)
    }

    try:
        response = requests.post(f"{GRAFANA_URL}/api/annotations", json=payload, headers=headers)
        response.raise_for_status()
        logging.info(f" Grafana annotation sent: {message}")
    except Exception as e:
        logging.error(f" Failed to send annotation: {e}")



loki_handler = LokiHandler(
    url=LOKI_URL,
    tags={"application": "smart-autoscaler"},
    version="1"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), loki_handler]
)
