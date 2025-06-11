import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge, Enum
from config import (
PROMETHEUS_URL,
STEP_SECONDS ,
LEARNING_DAYS,
HISTORICAL_CPU_USAGE_FALLBACK
)
# Global history dict
history = {}

def query_vm(query):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query}, timeout=5)
        if response.status_code == 200:
            result = response.json().get('data', {}).get('result', [])
            if result and 'value' in result[0]:
                return float(result[0]['value'][1])
    except Exception as e:
        print(f"⚠️ VM query failed: {e}")
    return np.random.uniform(0.1, 0.9)

def fetch_pod_metrics(pod_name="adservice", namespace="default"):
    now = datetime.now()
    try:
        return {
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
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
    except Exception as e:
        print(f"⚠️ Failed to fetch real metrics — using fallback: {e}")
        return {
            "hour_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "cpu_usage_lag_1": np.random.uniform(0.1, 0.9),
            "cpu_usage_lag_5": np.random.uniform(0.1, 0.9),
            "cpu_roll_mean_10": np.random.uniform(0.1, 0.9),
            "mem_usage": np.random.uniform(200, 300),
            "req_rate": np.random.uniform(5, 20)
        }     

def build_feature_vector(metrics, feature_list):
    return np.array([metrics[f] for f in feature_list if f in metrics])

def create_sequence(data, seq_length):
    """
    Creates sequences for LSTM input.
    Returns empty array if not enough data or data is ragged
    """
    if len(data) < seq_length:
        return np.array([])
    
    # Ensure all items are same shape
    try:
        arr = np.array([np.array(row).flatten() for row in data])
        if len(arr.shape) != 2 or arr.shape[1] == 0:
            raise ValueError("Invalid feature vector shape")
        
        return np.array([arr[i:i+seq_length] for i in range(len(arr) - seq_length + 1)])
    except Exception as e:
        print(f" Invalid sequence: {e}")
        return np.array([])

def record_live_data(pod_name, input_row, decision):
    # Also send to VictoriaMetrics
    decision_value = {"scale_down": 0, "no_change": 1, "scale_up": 2}.get(decision, 1)
    send_to_victoriametrics("autoscaler_decision", decision_value, {"service": pod_name, "action": decision})
    send_to_victoriametrics("autoscaler_cpu_usage", input_row[2], {"service": pod_name})  # cpu_usage_lag_1
    send_to_victoriametrics("autoscaler_mem_usage", input_row[5], {"service": pod_name})
    send_to_victoriametrics("autoscaler_req_rate", input_row[6], {"service": pod_name})

def validate_service_config(svc):
    if "pod_name" not in svc:
        raise KeyError("Missing 'pod_name'")
    if "feature_names" not in svc:
        raise KeyError("Missing 'feature_names'")
    if "seq_length" not in svc:
        raise KeyError("Missing 'seq_length'")

def fetch_historical_data(pod_name, namespace, seq_length=50, days=LEARNING_DAYS):
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
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params, timeout=5)
        if response.status_code == 200:
            result = response.json().get('data', {}).get('result', [])
            if not result:
                raise ValueError("No data returned from Prometheus")

            values = result[0].get('values', [])
            cpu_values = [float(v[1]) for v in values]

            if len(cpu_values) < seq_length:
                print(f"Only {len(cpu_values)} samples found — using fallback")
                return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]

            print(f"📊 Loaded {len(cpu_values)} historical entries for {pod_name}")
            return cpu_values[-seq_length:]

    except Exception as e:
        print(f"Historical fetch failed: {e}")

    print("Using simulated history (no Prometheus/VictoriaMetrics data)")
    return [np.random.uniform(0.1, 0.9) for _ in range(seq_length)]


def send_to_victoriametrics(metric_name, value, labels):
    """
    Send custom metric to VictoriaMetrics using JSON line forma
    """
    try:
        # Timestamp in milliseconds
        now = datetime.now()
        timestamp = int(now.timestamp() * 1000)  # milliseconds

        # Format labels properly
        # Validate labels
        label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
        line = f"{metric_name}{{{label_str}}} {value} {timestamp}"

        # Use correct VM import endpoint
        url = f"{PROMETHEUS_URL}/api/v1/import"
        
        headers = {
            "Content-Type": "text/plain",
            "User-Agent": "SmartAutoScaler/1.0"
        }
        
        response = requests.post(url, data=line, headers=headers, timeout=5)

        if response.status_code == 204:
            print(f"✅ [VM] Sent: {line}")
            return True
        else:
            print(f"⚠️ [VM] Unexpected status: {response.status_code}, expected 204")
            print(f"🪲 Response: {response.text}")
            return False
    except Exception as e:
        print(f"🚫 [VM] Could not reach VictoriaMetrics: {e}")
        return False

def fetch_logged_decisions(pod_name, namespace="default", days=7):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    start = int(start_time.timestamp())
    end = int(end_time.timestamp())

    query = f'autoscaler_decision{{service="{pod_name}"}}'
    params = {'query': query, 'start': start, 'end': end, 'step': STEP_SECONDS}

    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params, timeout=5)
        if response.status_code == 200:
            result = response.json().get('data', {}).get('result', [])
            if result:
                return [float(v[1]) for v in result[0].get('values', [])]
    except Exception as e:
        print(f" Failed to fetch logged decisions: {e}")

    print("Falling back to synthetic decisions")
    return np.random.randint(0, 3, size=100).tolist()

def create_sequence_from_vm(pod_name, namespace, seq_length=10):
    cpu_values = fetch_historical_data(pod_name, namespace, days=10)

    if len(cpu_values) < seq_length:
        print(f"Not enough data from VM — using simulated history")
        return [np.random.uniform(0.1, 0.9) for _ in range(seq_length)]

    live_metrics = fetch_pod_metrics(pod_name, namespace)

    synthetic_features = []
    for v in cpu_values[-seq_length:]:
        synthetic_features.append(np.array([
            datetime.now().hour,
            datetime.now().weekday(),
            v * 0.9,  # cpu_usage_lag_1
            v * 0.7,  # cpu_usage_lag_5
            v * 0.8,  # cpu_roll_mean_10
            live_metrics.get("mem_usage", 200),
            live_metrics.get("req_rate", 10)
        ]))

    print(f"Loaded {len(synthetic_features)} entries from VM for {pod_name}")
    return np.array(synthetic_features)


def check_vm_connection():
    """Check if VictoriaMetrics is reachable"""
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": "up"}, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f" VM connection failed: {e}")
        return False