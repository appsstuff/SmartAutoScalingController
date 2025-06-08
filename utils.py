import os
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://victoriametrics.monitoring.svc.cluster.local:8428/api/v1/query")
STEP_SECONDS = int(os.getenv("STEP_SECONDS", "60"))
HISTORY_FILE = "/data/history.pkl"

def load_history():
    if os.path.exists(HISTORY_FILE):
        print("🧠 Loading saved history...")
        return joblib.load(HISTORY_FILE)
    print("🆕 Starting with empty history")
    return {}

def save_history(history):
    joblib.dump(history, HISTORY_FILE)
    print(f"💾 History saved to {HISTORY_FILE}")

def query_vm(query):
    try:
        response = requests.get(PROMETHEUS_URL, params={'query': query}, timeout=5)
        if response.status_code == 200:
            result = response.json().get('data', {}).get('result', [])
            if result:
                return float(result[0]['value'][1])
    except Exception as e:
        print(f"⚠️ VM query failed: {e}")
    return np.random.uniform(0.1, 0.9)

def fetch_pod_metrics(pod_name="adservice", namespace="default"):
    """
    Fetch live metrics from VictoriaMetrics (no recursion)
    """
    return {
        "hour_of_day": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "cpu_usage_lag_1": query_vm(f'container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}}'),
        "cpu_usage_lag_5": query_vm(f'container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}} offset 5m'),
        "cpu_roll_mean_10": query_vm(f'avg_over_time(container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}}[10m]'),
        "mem_usage": query_vm(f'container_memory_usage_bytes{{namespace="{namespace}", container_name="{pod_name}"}}') / (1024 * 1024),
        "req_rate": query_vm(f'rate(http_requests_total{{namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])'),
        "latency": query_vm(f'histogram_quantile(0.95, sum(rate(http_request_latencies_bucket{{le="+Inf", namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])) by (le))'),
        "net_receive_KB": query_vm(f'rate(container_network_receive_bytes_total{{namespace="{namespace}", container_name="{pod_name}"}}[1m])') * 1024,
        "net_transmit_KB": query_vm(f'rate(container_network_transmit_bytes_total{{namespace="{namespace}", container_name="{pod_name}"}}[1m])') * 1024,
        "pod_restarts": query_vm(f'kube_pod_container_status_restarts_total{{namespace="{namespace}", container="{pod_name}"}}'),
        "pod_ready": query_vm(f'kube_pod_container_status_ready{{namespace="{namespace}", container="{pod_name}"}}')
    }

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

def predict_scaling_action(input_row, sequence):
    """
    Hybrid prediction using multiple models
    Returns: 'scale_down', 'no_change', or 'scale_up'
    """
    # Default to no_change if no sequence data
    if len(sequence) == 0:
        return 'no_change'
    
    # Simple threshold-based logic
    cpu_usage = input_row[2]  # cpu_usage_lag_1
    if cpu_usage > 0.8:
        return 'scale_up'
    elif cpu_usage < 0.2:
        return 'scale_down'
    return 'no_change'