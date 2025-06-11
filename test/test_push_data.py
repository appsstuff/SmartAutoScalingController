import os
from datetime import datetime
import requests

class VictoriaMetricsPusher:
    def __init__(self, base_url=None):
        self.base_url = base_url or os.getenv("PROMETHEUS_URL", "http://34.73.82.105:8428")
        print(f"🔌 Connecting to VictoriaMetrics at {self.base_url}")

    def send_metric(self, metric_name, value, labels=None):
        """
        Send one metric line to VictoriaMetrics
        
        Args:
            metric_name (str): Metric name like 'autoscaler_decision'
            value (float): Value to send
            labels (dict): Labels like {'service': 'adservice', 'action': 'scale_up'}
        
        Returns:
            bool: True if success
        """
        timestamp = int(datetime.now().timestamp() * 1000)  # Milliseconds
        label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()]) if labels else ""
        line = f"{metric_name}{{{label_str}}} {value} {timestamp}"

        url = f"{self.base_url}/api/v1/import"
        headers = {"Content-Type": "text/plain"}

        try:
            response = requests.post(url, data=line, headers=headers, timeout=5)
            if response.status_code in [200, 204]:
                print(f"✅ Sent to VictoriaMetrics: {line}")
                return True
            else:
                print(f"⚠️ Failed to write to VM: {response.status_code}, expected 200 or 204")
                print(f"🪲 Response: {response.text}")
                return False
        except Exception as e:
            print(f"🚫 Could not reach VictoriaMetrics: {e}")
            return False
        
        
if __name__ == "__main__":
    pusher = VictoriaMetricsPusher()

    # Simulated values
    pod_name = "testservice"
    cpu_usage = 0.65
    mem_usage = 250
    req_rate = 15

    # Send decision
    decision_success = pusher.send_metric(
        "autoscaler_decision",
        2,
        {"service": pod_name, "action": "scale_up"}
    )

    # Send CPU usage
    cpu_success = pusher.send_metric(
        "autoscaler_cpu_usage",
        cpu_usage,
        {"service": pod_name}
    )

    # Send memory usage
    mem_success = pusher.send_metric(
        "autoscaler_mem_usage",
        mem_usage,
        {"service": pod_name}
    )

    # Send request rate
    req_success = pusher.send_metric(
        "autoscaler_req_rate",
        req_rate,
        {"service": pod_name}
    )

    if all([decision_success, cpu_success, mem_success, req_success]):
        print("🎉 All metrics successfully sent to VictoriaMetrics!")
    else:
        print("⚠️ Some metrics failed to send — check logs")        