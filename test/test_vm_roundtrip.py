import time
from utils import send_to_victoriametrics, query_vm

if __name__ == "__main__":
    pod_name = "testservice"
    namespace = "default"

    # Step 1: Send data
    decision_success = send_to_victoriametrics(
        "autoscaler_decision",
        1,
        {"service": pod_name, "action": "no_change"}
    )

    cpu_success = send_to_victoriametrics(
        "autoscaler_cpu_usage",
        0.65,
        {"service": pod_name}
    )

    mem_success = send_to_victoriametrics(
        "autoscaler_mem_usage",
        250,
        {"service": pod_name}
    )

    req_success = send_to_victoriametrics(
        "autoscaler_req_rate",
        15,
        {"service": pod_name}
    )

    if all([decision_success, cpu_success, mem_success, req_success]):
        print("🎉 All metrics sent to VictoriaMetrics")

    # Step 2: Try fetching back
    print("\n🔍 Trying to fetch back from VictoriaMetrics...")

    def fetch_metric_back(metric_name, labels_dict):
        label_query = ",".join([f'{k}="{v}"' for k, v in labels_dict.items()])
        query = f'{metric_name}{{{label_query}}}'
        result = query_vm(query)
        print(f"📊 {metric_name}: {result}")

    fetch_metric_back("autoscaler_decision", {"service": pod_name})
    fetch_metric_back("autoscaler_cpu_usage", {"service": pod_name})
    fetch_metric_back("autoscaler_mem_usage", {"service": pod_name})
    fetch_metric_back("autoscaler_req_rate", {"service": pod_name})