import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_service_data(pod_name):
    """Load data from CSV logs"""
    log_path = f"/data/live_data/{pod_name}_live_data.csv"
    if not os.path.exists(log_path):
        print(f" No data found for {pod_name}")
        return None
    
    df = pd.read_csv(log_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def plot_cpu_trend(df, pod_name):
    """Plot CPU usage over time"""
    if "cpu_usage_lag_1" in df.columns:
        plt.figure(figsize=(12, 4))
        sns.lineplot(data=df, x=df.index, y="cpu_usage_lag_1", label="CPU Usage", color="blue")
        plt.title(f"📈 CPU Usage Trend — {pod_name}")
        plt.xlabel("Time")
        plt.ylabel("CPU Usage (normalized)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"/data/cpu_usage_{pod_name}.png")
        plt.close()

def plot_request_rate_trend(df, pod_name):
    """Plot request rate over time"""
    if "req_rate" in df.columns:
        plt.figure(figsize=(12, 4))
        sns.lineplot(data=df, x=df.index, y="req_rate", label="Request Rate", color="green")
        plt.title(f"📊 Request Rate Trend — {pod_name}")
        plt.xlabel("Time")
        plt.ylabel("Requests/Second")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"/data/request_rate_{pod_name}.png")
        plt.close()

def analyze_service_trend(df, pod_name):
    """Run full analysis on service data"""
    print(f"\n📊 Analyzing trend for {pod_name}...")
    
    plot_cpu_trend(df, pod_name)
    plot_request_rate_trend(df, pod_name)
    plot_decisions(df, pod_name)
    plot_memory_trend(df, pod_name)
    plot_decision_frequency(df, pod_name)
    
def plot_decisions(df, pod_name):
    """Plot scaling decisions over time"""
    decision_map = {"scale_down": -1, "no_change": 0, "scale_up": 1}
    df["decision_code"] = df["decision"].map(decision_map)

    plt.figure(figsize=(12, 3))
    sns.scatterplot(data=df, x=df.index, 
                    y="decision_code", hue="decision",
                    palette={"scale_down": "red", "no_change": "gray", "scale_up": "green"})

def plot_memory_trend(df, pod_name):
    """Plot memory usage over time"""
    if "memory_usage" in df.columns:
        plt.figure(figsize=(12, 4))
        sns.lineplot(data=df, x=df.index, y="memory_usage", label="Memory Usage", color="purple")
        plt.title(f"📊 Memory Usage Trend — {pod_name}")
        plt.xlabel("Time")
        plt.ylabel("Memory Usage (MB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"/data/memory_usage_{pod_name}.png")
        plt.close()

def plot_decision_frequency(df, pod_name):
    """Plot frequency of scaling decisions"""
    if "decision" in df.columns:
        plt.figure(figsize=(8, 6))
        decision_counts = df["decision"].value_counts()
        sns.barplot(x=decision_counts.index, y=decision_counts.values)
        plt.title(f"📊 Scaling Decision Frequency — {pod_name}")
        plt.xlabel("Decision")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"/data/decision_frequency_{pod_name}.png")
        plt.close()