import os
import requests
from datetime import datetime, timedelta
import numpy as np

class VictoriaMetricsClient:
    def __init__(self, base_url=None):
        self.base_url = base_url or PROMETHEUS_URL
        print(f"🔌 Connecting to VictoriaMetrics at {self.base_url}")
    
    def query_range(self, query, start, end, step="60s"):
        """
        Run range query on VictoriaMetrics/Prometheus
        Returns list of (timestamp, value)
        """
        url = f"{self.base_url}/api/v1/query_range"
        params = {
            'query': query,
            'start': int(start.timestamp()),
            'end': int(end.timestamp()),
            'step': step
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                result = response.json().get('data', {}).get('result', [])
                if result and 'values' in result[0]:
                    return [(int(ts), float(val)) for ts, val in result[0]['values']]
            else:
                print(f"🚫 Query failed with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"⚠️ Network error while querying VM: {e}")
        return []

    def get_cpu_usage(self, pod_name, namespace="default", days=7):
        """
        Get historical CPU usage for a service
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        query = f'container_cpu_usage_seconds_total'
        values = self.query_range(query, start=start_time, end=end_time, step="1m")
        if values:
            print(f"✅ Retrieved {len(values)} samples")
            for ts, val in values[-10:]:  # Show last 10 samples
                print(f"🕒 {datetime.fromtimestamp(ts)} → CPU Usage: {val:.5f}")
        else:
            print("❌ No data returned from VictoriaMetrics")
        return values

    def get_request_rate(self, pod_name, namespace="default", minutes=10):
        """
        Get request rate over the past N minutes
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)

        query = f'rate(http_requests_total{{namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])'
        print(f"📊 Fetching request rate for {pod_name} ({namespace}) — last {minutes} minutes")

        values = self.query_range(query, start=start_time, end=end_time, step="1m")
        if values:
            print(f"✅ Retrieved {len(values)} samples")
            for ts, val in values[-10:]:
                print(f"🕒 {datetime.fromtimestamp(ts)} → RPS: {val:.2f}")
        else:
            print("❌ No request rate data found")
        return values

    def get_mem_usage(self, pod_name, namespace="default", days=1):
        """
        Get historical memory usage
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        query = f'container_memory_usage_bytes{{namespace="{namespace}", container_name="{pod_name}"}}'
        print(f"📊 Fetching memory usage for {pod_name} — last {days} day(s)")

        values = self.query_range(query, start=start_time, end=end_time, step="1m")
        if values:
            print(f"✅ Retrieved {len(values)} samples")
            for ts, val in values[-10:]:
                print(f"🕒 {datetime.fromtimestamp(ts)} → Mem Usage: {val / (1024 * 1024):.2f} MB")
        else:
            print("❌ No memory usage data found")
        return values
    
if __name__ == "__main__":
    client = VictoriaMetricsClient()

    pod_name = "adservice"
    namespace = "default"

    # Step 1: Test CPU usage
    cpu_data = client.get_cpu_usage(pod_name, namespace, days=1)

    # Step 2: Test Request Rate
    rps_data = client.get_request_rate(pod_name, namespace, minutes=10)

    # Step 3: Test Memory Usage
    mem_data = client.get_mem_usage(pod_name, namespace, days=1)    