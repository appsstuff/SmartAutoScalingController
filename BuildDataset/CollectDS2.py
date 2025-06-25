import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import os

# ======================
# Configuration Section
# ======================

VM_URL = "http://34.73.82.105:8428"  # Replace with your VM IP
OUTPUT_DIR = "./datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

END_TIME = datetime.now(timezone.utc)
START_TIME = END_TIME - timedelta(hours=6)
STEP = "60s"

# List of metrics to fetch
METRICS_CONFIG = {
    "cpu_usage": 'container_cpu_usage_seconds_total{namespace="default"}',
    "mem_usage": 'container_memory_usage_bytes{namespace="default"}',
    "req_rate": 'rate(istio_requests_total{ namespace="default"}[1m])',
    "latency": 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket[1m])) by (le))',
    "net_receive": 'rate(container_network_receive_bytes_total{namespace="default"}[1m])',
    "net_transmit": 'rate(container_network_transmit_bytes_total{namespace="default"}[1m])',
    "pod_restarts": 'kube_pod_container_status_restarts_total{namespace="default"}',
    "pod_ready": 'kube_pod_container_status_ready{namespace="default"}'
}

# ======================
# Helper Functions
# ======================

def query_vm(metric_query: str, start: datetime, end: datetime, step: str = "60s") -> dict:
    """Query VictoriaMetrics using PromQL range API"""
    params = {
        "query": metric_query,
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
        "step": step
    }
    try:
        response = requests.get(f"{VM_URL}/api/v1/query_range", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f" Request failed for '{metric_query}': {e}")
        return {"data": {"result": []}}

def vm_to_df(data: dict, metric_name: str) -> pd.DataFrame:
    """Convert VictoriaMetrics JSON response to DataFrame"""
    results = []
    for item in data.get("data", {}).get("result", []):
        metric = item["metric"]
        values = item["values"]
        for ts, value in values:
            results.append({
                "timestamp": datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc),
                "pod": metric.get("pod", "unknown"),
                "container": metric.get("container", "all"),
                "namespace": metric.get("namespace", "default"),
                "metric": metric_name,
                "value": float(value) if value is not None else np.nan
            })
    return pd.DataFrame(results)

def fetch_all_metrics(
    metrics: Dict[str, str],
    start: datetime,
    end: datetime,
    step: str = "60s"
) -> pd.DataFrame:
    """Fetch all metrics and concatenate into one DataFrame"""
    dfs = []
    for name, query in metrics.items():
        raw_data = query_vm(query, start, end, step)
        if raw_data.get("status") == "success":
            df = vm_to_df(raw_data, metric_name=name)
            if not df.empty:
                dfs.append(df)
            else:
                print(f"⚠️ No data returned for '{name}'")
        else:
            print(f" Query failed for '{name}'")

    if not dfs:
        print(" No metrics were successfully fetched.")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

# ======================
# Data Engineering
# ======================

def safe_add_lag(
    df: pd.DataFrame,
    group_col: str = "pod",
    metric_col: str = None,
    lag: int = 1
) -> pd.DataFrame:
    new_col_name = f"{metric_col}_lag_{lag}"
    if metric_col not in df.columns:
        print(f"⚠️ Column '{metric_col}' not found. Skipping '{new_col_name}'")
        return df
    df[new_col_name] = df.groupby(group_col)[metric_col].shift(lag)
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from timestamp"""
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["time_of_day_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["time_of_day_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    return df

def create_rolling_features(
    df: pd.DataFrame,
    metric: str,
    window: int = 10
) -> pd.DataFrame:
    col_mean = f"{metric}_roll_mean_{window}"
    col_std = f"{metric}_roll_std_{window}"

    if metric not in df.columns:
        print(f"⚠️ Metric '{metric}' not found. Skipping rolling features.")
        return df

    df[col_mean] = df.groupby("pod")[metric].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    df[col_std] = df.groupby("pod")[metric].transform(lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True)
    return df

def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    if "mem_usage" in df.columns:
        df["mem_usage_MB"] = df["mem_usage"] / (1024 * 1024)
    if "net_receive" in df.columns:
        df["net_receive_KB"] = df["net_receive"] / 1024
    if "net_transmit" in df.columns:
        df["net_transmit_KB"] = df["net_transmit"] / 1024
    return df

def filter_app_containers(df: pd.DataFrame, container_name: str = "server") -> pd.DataFrame:
    if container_name in df["container"].unique():
        return df[df["container"] == container_name]
    else:
        print(f"⚠️ Container '{container_name}' not found. Using all containers.")
        return df

def label_peak_usage(df: pd.DataFrame, metric: str = "cpu_usage", percentile: float = 0.95) -> pd.DataFrame:
    if metric not in df.columns:
        print(f" Column '{metric}' not found. Skipping peak labeling.")
        df["is_peak"] = 0
        return df
    threshold = df[metric].quantile(percentile)
    df[f"is_{metric}_peak"] = (df[metric] > threshold).astype(int)
    return df

def label_scale_decision(
    df: pd.DataFrame,
    metric: str = "cpu_usage",
    window: int = 5
) -> pd.DataFrame:
    diff_col = f"{metric}_diff_{window}"
    if metric not in df.columns:
        df["scale_decision"] = "no_change"
        return df

    df[diff_col] = df.groupby("pod")[metric].diff(window)
    conditions = [
        df[diff_col] > 0.1,
        df[diff_col] < -0.1
    ]
    choices = ["scale_up", "scale_down"]
    df["scale_decision"] = np.select(conditions, choices, default="no_change")
    return df

# ======================
# Main Preprocessing Function
# ======================

def preprocess_pod_metrics(
    df: pd.DataFrame
) -> pd.DataFrame:
    if df.empty:
        print(" Empty input DataFrame")
        return df

    print("\n🔍 Available metrics before pivot:")
    print(df['metric'].unique())

    # Pivot to wide format
    df_wide = df.pivot_table(
        index=["timestamp", "pod", "container", "namespace"],
        columns="metric",
        values="value"
    ).reset_index()

    # Filter only application containers
    df_app = filter_app_containers(df_wide, container_name="server")

    # Sort by timestamp
    df_app.sort_values("timestamp", inplace=True)

    # Fill missing values
    df_clean = fill_missing_values(df_app)

    # Convert units
    df_unit = convert_units(df_clean)

    # Create time-based features
    df_time = create_time_features(df_unit)

    # Add lagged features
    for metric in METRICS_CONFIG.keys():
        df_time = safe_add_lag(df_time, group_col="pod", metric_col=metric, lag=1)
        df_time = safe_add_lag(df_time, group_col="pod", metric_col=metric, lag=5)

    # Rolling features (only for cpu_usage if present)
    if "cpu_usage" in df_time.columns:
        df_time = create_rolling_features(df_time, "cpu_usage", window=10)

    # Label peaks and scale decisions
    df_final = label_peak_usage(df_time, metric="cpu_usage", percentile=0.95)
    df_final = label_scale_decision(df_final, metric="cpu_usage", window=5)

    print("\n✅ Final columns in dataset:")
    print(df_final.columns.tolist())

    return df_final

# ======================
# End-to-End Pipeline
# ======================

def generate_dataset(
    metrics: Dict[str, str] = METRICS_CONFIG,
    start_time: datetime = None,
    end_time: datetime = None,
    step: str = "60s",
    output_file: str = None
) -> pd.DataFrame:
    """
    End-to-end function to collect and preprocess dataset
    """

    if start_time is None:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=6)

    print("📊 Starting dataset collection pipeline...")

    print("🔍 Fetching raw metrics...")
    df_raw = fetch_all_metrics(metrics, start_time, end_time, step)

    if df_raw.empty:
        print(" No data collected. Aborting pipeline.")
        return df_raw

    print("🛠️ Preprocessing pod metrics...")
    df_processed = preprocess_pod_metrics(df_raw)

    if df_processed.empty:
        print(" No processed data. Check if input had valid metrics.")
        return df_processed

    print("💾 Saving final dataset...")
    if output_file is None:
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = f"./datasets/workload_dataset_v{version}.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Confirm req_rate presence
    if 'req_rate' in df_processed.columns:
        print("✅ req_rate is present in the final dataset!")
    else:
        print(" req_rate is NOT present in the final dataset!")

    df_processed.to_csv(output_file, index=False)
    print(f"✅ Dataset saved at {output_file}")

    return df_processed

# ======================
# Run the Pipeline
# ======================

if __name__ == "__main__":
    df = generate_dataset(metrics=METRICS_CONFIG)
    print("\n📈 Final dataset preview:")
    print(df.head())