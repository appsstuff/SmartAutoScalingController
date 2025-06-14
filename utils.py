import os
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score
import numpy as np
import requests
import pandas as pd
import joblib
from model_inference import ModelManager
from config import (
    PROMETHEUS_URL,
    RETRAIN_THRESHOLD,
    STEP_SECONDS,
    LEARNING_DAYS,
    HISTORY_FILE,
    LOG_FILE,
    HISTORICAL_CPU_USAGE_FALLBACK
)

# Global history dict
metrics_history = {}

def query_vm(query):
    """
    Query VictoriaMetrics or Prometheus.
    Returns float value or fallback random number.
    """
    try:
        url = f"{PROMETHEUS_URL}/api/v1/query"
        print(f"📊 Querying {url} with: {query}")

        response = requests.get(url, params={'query': query}, timeout=5)

        if response.status_code != 200:
            print(f"⚠️ Query failed with status {response.status_code}")
            return np.random.uniform(0.1, 0.9)

        # Safely check content type
        if "application/json" not in response.headers.get("Content-Type", ""):
            print("🪲 Non-JSON response — using fallback")
            print("🪲 Raw response:", response.text[:200])
            return np.random.uniform(0.1, 0.9)

        data = response.json()
        result = data.get('data', {}).get('result', [])

        if not isinstance(result, list) or len(result) == 0:
            print("🪲 No valid results found — using fallback")
            return np.random.uniform(0.1, 0.9)

        first_result = result[0]
        if not isinstance(first_result, dict):
            print("🪲 First result not a dict — using fallback")
            return np.random.uniform(0.1, 0.9)

        value = first_result.get("value")
        if not isinstance(value, list) or len(value) < 2:
            print("🪲 Invalid 'value' format — using fallback")
            return np.random.uniform(0.1, 0.9)

        return float(value[1])

    except json.JSONDecodeError:
        print("🪲 Failed to decode response as JSON — using fallback")
        return np.random.uniform(0.1, 0.9)
    except Exception as e:
        print(f"🚫 VM query failed: {e}")
        return np.random.uniform(0.1, 0.9)
    
def fetch_pod_metrics(pod_name="adservice", namespace="default"):
    now = datetime.now()
    try:
        return {
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "cpu_usage_lag_1": query_vm(f'container_cpu_usage_seconds_total{{namespace="{namespace}",pod=~"{pod_name}.*"}}'),
            "cpu_usage_lag_5": query_vm(f'container_cpu_usage_seconds_total{{namespace="{namespace}", pod=~"{pod_name}.*"}} offset 5m'),
            "cpu_roll_mean_10": query_vm(f'avg_over_time(container_cpu_usage_seconds_total{{namespace="{namespace}", pod=~"{pod_name}.*"}}[10m])'),
            "mem_usage": query_vm(f'container_memory_usage_bytes{{namespace="{namespace}", pod=~"{pod_name}.*"}}') / (1024 * 1024),
            "req_rate": query_vm(f'rate(istio_requests_total{{namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])'),
            "latency": query_vm(f'histogram_quantile(0.95, sum(rate(http_request_latencies_bucket{{le="+Inf", namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])) by (le))'),
            "net_receive_KB": query_vm(f'rate(container_network_receive_bytes_total{{namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])') * 1024,
            "net_transmit_KB": query_vm(f'rate(container_network_transmit_bytes_total{{namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])') * 1024,
            "pod_restarts": query_vm(f'kube_pod_container_status_restarts_total{{namespace="{namespace}", pod=~"{pod_name}.*"}}'),
            "pod_ready": query_vm(f'kube_pod_container_status_ready{{namespace="{namespace}", pod=~"{pod_name}.*"}}')
        }
    except Exception as e:
        print(f"⚠️ Failed to fetch live metrics for {pod_name}: {e}")
        return {
            "hour_of_day": now.hour,
            "day_of_week": now.weekday(),
            "cpu_usage_lag_1": np.random.uniform(0.1, 0.9),
            "cpu_usage_lag_5": np.random.uniform(0.1, 0.9),
            "cpu_roll_mean_10": np.random.uniform(0.1, 0.9),
            "mem_usage": np.random.uniform(200, 300),
            "req_rate": np.random.uniform(5, 20),
            "latency": np.random.uniform(0.01, 0.1),
            "net_receive_KB": np.random.uniform(100, 300),
            "net_transmit_KB": np.random.uniform(100, 300),
            "pod_restarts": np.random.randint(0, 2),
            "pod_ready": 1.0
        }


def build_feature_vector(metrics, feature_list):
    """Builds input vector based on config-defined features"""
    try:
        return np.array([float(metrics[f]) for f in feature_list if f in metrics])
    except Exception as e:
        print(f"🪲 Invalid metric during vector building: {e}")
        return []


def create_sequence(data, seq_length):
    """
    Creates sequences for LSTM input.
    Returns empty array if invalid data
    """
    if not isinstance(data, (np.ndarray, list)):
        print("🪲 Invalid input type for sequence creation")
        return np.array([])

    try:
        arr = np.array(
            [np.array(row).flatten() for row in data if isinstance(row, (np.ndarray, list))]
        )
        if len(arr.shape) != 2 or arr.shape[1] == 0:
            raise ValueError("Invalid feature vector shape")

        if len(arr) < seq_length:
            print(f"🪲 Not enough samples ({len(arr)}) — need at least {seq_length}")
            return np.array([])

        return np.array([arr[i:i+seq_length] for i in range(len(arr) - seq_length + 1)])
    
    except Exception as e:
        print(f"🪲 Sequence creation failed: {e}")
        return np.array([])


def load_history():
    path = HISTORY_FILE
    if os.path.exists(path):
        print(f"🧠 Loading history from {path}")
        return joblib.load(path)
    else:
        print("🆕 Starting with empty history")
        return {}


def save_history(history_dict):
    path = HISTORY_FILE
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(history_dict, path)
        print(f"💾 History saved to {path}")
    except Exception as e:
        print(f"🚫 Failed to save history: {e}")


def record_live_data(pod_name, input_row, decision):
    log_dir = LOG_FILE
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().isoformat()

    line = f"{timestamp},{pod_name},{decision},{input_row}\n"

    with open(os.path.join(log_dir, "decisions.csv"), "a") as f:
        f.write(line)
    print(f"🪟 Decision logged for {pod_name}")
    
    

def fetch_historical_data(pod_name, namespace="default", seq_length=50, days=LEARNING_DAYS):
    """
    Fetch historical CPU usage from VictoriaMetrics or Prometheus.
    Returns list of values or fallback if no data.
    """
    end_time = int(datetime.now().timestamp())
    start_time = end_time - (days * 86400)  # seconds per day
    
    query = f'container_cpu_usage_seconds_total{{namespace="{namespace}", container="{pod_name}"}}'
    params = {
        'query': query,
        'start': start_time,
        'end': end_time,
        'step': STEP_SECONDS
    }
    
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params, timeout=5)
        
        if response.status_code != 200:
            print(f"⚠️ Historical query failed with status {response.status_code}")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]

        if "application/json" not in response.headers.get("Content-Type", ""):
            print("🪲 Received non-JSON response — using fallback history")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]
            
        data = response.json()
        result = data.get('data', {}).get('result', [])
        
        if not result:
            print("🪲 No historical data found — using synthetic fallback")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]
        
        values = result[0].get('values', [])
        if not values:
            print("🪲 Empty values array — using fallback")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]
        
        cpu_values = [float(v[1]) for v in values if len(v) > 1]
        if len(cpu_values) < seq_length:
            print(f"🪲 Not enough historical samples ({len(cpu_values)}) — using fallback")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]
        
        print(f"📊 Loaded {len(cpu_values)} historical entries for {pod_name}")
        return cpu_values[-seq_length:]

    except Exception as e:
        print(f"⚠️ Historical fetch failed: {e}")
        return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]    
    
def send_to_victoriametrics(metric_name, value, labels):
    """
    Send custom metric to VictoriaMetrics using Prometheus text format.
    Returns True if success
    """
    timestamp = int(datetime.now().timestamp() * 1000)  # milliseconds
    label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
    line = f"{metric_name}{{{label_str}}} {value} {timestamp}"

    url = f"{PROMETHEUS_URL}/api/v1/import/prometheus"
    headers = {"Content-Type": "text/plain","charset": "utf-8"}

    try:
        response = requests.post(url, data=line, headers=headers, timeout=5)
        if response.status_code == 204:
            print(f"✅ Sent to VictoriaMetrics: {line}")
            return True
        else:
            print(f"⚠️ Failed to write to VM: {response.status_code}")
            print(f"🪲 Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"🚫 Could not reach VictoriaMetrics: {e}")
        return False

def validate_service_config(svc):
    required_keys = ["pod_name", "feature_names", "seq_length", "min_replicas", "max_replicas", "scaling_step"]
    for key in required_keys:
        if key not in svc:
            raise KeyError(f"Missing '{key}' in service configuration: {svc}")


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
        if response.status_code != 200:
            print(f"⚠️ Historical query failed with status {response.status_code}")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]

        if "application/json" not in response.headers.get("Content-Type", ""):
            print("🪲 Received non-JSON response — using fallback history")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]
            
        data = response.json()
        result = data.get('data', {}).get('result', [])
        
        if not result:
            print("🪲 No historical data found — using synthetic fallback")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]
        
        values = result[0].get('values', [])
        if not values:
            print("🪲 Empty values array — using fallback")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]
        
        cpu_values = [float(v[1]) for v in values if len(v) > 1]
        if len(cpu_values) < seq_length:
            print(f"🪲 Not enough historical samples ({len(cpu_values)}) — using fallback")
            return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]
        
        print(f"📊 Loaded {len(cpu_values)} historical entries for {pod_name}")
        return cpu_values[-seq_length:]

    except Exception as e:
        print(f"⚠️ Historical fetch failed: {e}")
        return HISTORICAL_CPU_USAGE_FALLBACK[:seq_length]


def check_vm_connection():
    """Check if VictoriaMetrics is reachable"""
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": "up"}, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f" VM connection failed: {e}")
        return 

def load_training_data(pod_name, namespace="default"):
    """
    Load historical decisions and metrics for retraining
    Returns X_train, y_train
    """
    log_dir = LOG_FILE
    log_path = os.path.join(log_dir, "decisions.csv")

    if not os.path.exists(log_path):
        print(f"🪲 No training logs found for {pod_name} — skipping retraining")
        return None, None

    df = pd.read_csv(log_path)
    df = df[df['service'] == pod_name]

    if len(df) < RETRAIN_THRESHOLD:
        print(f"🪲 Not enough data for {pod_name} — need at least {RETRAIN_THRESHOLD}")
        return None, None

    # Map labels to numeric values
    X_train = df[["hour_of_day", "day_of_week", "cpu_usage_lag_1", "cpu_usage_lag_5", "cpu_roll_mean_10", "mem_usage", "req_rate"]].values
    y_train = df["action"].map({"scale_down": 0, "no_change": 1, "scale_up": 2}).values

    return X_train, y_train
    
def validate_model_performance(pod_name, namespace="default"):
    """
    Validate current model against recent decisions.
    Returns: float accuracy score
    """
    try:
        X_test, y_test = load_training_data(pod_name, namespace)
        if X_test is None or y_test is None:
            return 0.0

        manager = ModelManager(pod_name)
        predictions = [manager.predict(x) for x in X_test]
        predicted_labels = [{"scale_down": 0, "no_change": 1, "scale_up": 2}[p] for p in predictions]
        acc = accuracy_score(y_test, predicted_labels)
        print(f"🧪 Current model accuracy for {pod_name}: {acc:.2f}")
        return acc
    except Exception as e:
        print(f"🪲 Failed to validate {pod_name}: {e}")
        return 0.0
