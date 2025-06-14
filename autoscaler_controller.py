# autoscaler_controller.py
import time
import os
from datetime import datetime
import numpy as np
from pod_config import AUTO_SCALE_SERVICES
from model_inference import predict_scaling_action
from k8s_scaler import apply_k8s_scaling
from prometheus_client import start_http_server
import threading
from flask import Flask, jsonify
from retrain_models import retrain_all_models
from config import (
    DECISION_GAUGE,
    CPU_USAGE_GAUGE,
    MEM_USAGE_GAUGE,
    REQ_RATE_GAUGE,
    PROMETHEUS_PORT,
    RETRAIN_THRESHOLD  # Add a threshold for retraining
)

from utils import (
    fetch_pod_metrics,
    build_feature_vector,
    send_to_victoriametrics,
    create_sequence,
    validate_service_config,
    fetch_historical_data,
    record_live_data,
    check_vm_connection,
    metrics_history
)

# ====== START HEALTH CHECK SERVER EARLY ======
def create_health_check_server(port=8080):
    app = Flask(__name__)
    
    @app.route('/healthz')
    def health_check():
        return jsonify({"status": "healthy"})
        
    @app.route('/ready')
    def ready_check():
        history_ready = len(metrics_history) > 0
        return jsonify({
            "status": "healthy" if history_ready else "unhealthy",
            "history_ready": history_ready
        }), 200 if history_ready else 503

    # Run Flask in background thread
    threading.Thread(target=lambda: app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )).start()

# Start health server before anything else
print(" Starting health check server...")
create_health_check_server(port=8080)
print(" Health check server running on port 8080")
# =============================================
print(f"Test Connection .... {check_vm_connection()}")

# Start Prometheus metrics server on port 8900
start_http_server(PROMETHEUS_PORT)
print(" Prometheus metrics endpoint started at :9090")

try:
    while True:
        total = len(AUTO_SCALE_SERVICES)
        print(f"\n Starting autoscaling round — {total} services")

        for idx, svc in enumerate(AUTO_SCALE_SERVICES):
            percent = int((idx + 1) / total * 100)

            try:
                validate_service_config(svc)

                pod_name = svc["pod_name"]
                namespace = svc.get("namespace", "default")
                seq_length = svc.get("seq_length", 10)
                feature_names = svc.get("feature_names", [])

                progress_bar = f"[{'#' * (percent // 5):<20}] {percent}% — Processing {pod_name}"
                print(f"\n{progress_bar}")

                # Step 1: Fetch live metrics
                print(f"🛠️  Step 1: Fetch live metrics  : {pod_name}")

                raw_metrics = fetch_pod_metrics(pod_name, namespace)

                # Step 2: Build feature vector
                print(f"🛠️  Step 2: Build feature vector : {pod_name}")
                input_row = build_feature_vector(raw_metrics, feature_names)
                if len(input_row) == 0:
                    print(f" No features built for {pod_name}")
                    continue

                key = f"{namespace}/{pod_name}"

                # Initialize list if key not present
                if key not in metrics_history:
                    metrics_history[key] = []

                # Append the latest input row
                metrics_history[key].append(input_row)

                # Step 3: Load fallback history if insufficient data
                print(f"🛠️  Step 3: Load fallback history if insufficient data : {pod_name}")
                if len(metrics_history[key]) < seq_length:
                    print(f"📥 No valid history found for {key} — fetching past data")
                    cpu_values = fetch_historical_data(pod_name, namespace, days=10)

                    synthetic_features = []
                    for v in cpu_values[-seq_length:]:
                        synthetic_features.append(np.array([
                            datetime.now().hour,
                            datetime.now().weekday(),
                            v * 0.9,  # cpu_usage_lag_1
                            v * 0.7,  # cpu_usage_lag_5
                            v * 0.8,  # cpu_roll_mean_10
                            raw_metrics.get("mem_usage", 200),
                            raw_metrics.get("req_rate", 10)
                        ]))
                    metrics_history[key] = synthetic_features
                    print(f"📊 Loaded {len(synthetic_features)} historical entries for {pod_name}")

                # Limit history size
                max_history = seq_length + 10
                if len(metrics_history[key]) > max_history:
                    metrics_history[key] = metrics_history[key][-max_history:]

                # Step 4: Create LSTM input
                print(f"🛠️  Step 4: Create LSTM input  : {pod_name}")
                seq_input = create_sequence(np.array(metrics_history[key]), seq_length)
                if len(seq_input) == 0:
                    print(f"⏳ Waiting — collecting initial sequence for {pod_name}")
                    continue

                # Step 5: Make prediction
                print(f"🛠️  Step 5: Make prediction  : {pod_name}")
                decision = predict_scaling_action(input_row, seq_input, service_name=pod_name)
                print(f" Decision for {pod_name}: {decision}")

                # Step 6: Log decision
                decision_value = {"scale_down": 0, "no_change": 1, "scale_up": 2}.get(decision, 1)

                if not check_vm_connection():
                    print(" VictoriaMetrics unreachable — skipping log push")
                else:
                    # Log decision
                    vm_success = send_to_victoriametrics(
                        "autoscaler_decision",
                        decision_value,
                        {"service": pod_name, "action": decision}
                    )

                    DECISION_GAUGE.labels(service=pod_name).set(decision_value)

                    # Also log CPU, Mem, Req Rate
                    vm_success &= send_to_victoriametrics(
                        "autoscaler_cpu_usage",
                        raw_metrics.get("cpu_usage_lag_1", 0.0),
                        {"service": pod_name}
                    )

                    vm_success &= send_to_victoriametrics(
                        "autoscaler_mem_usage",
                        raw_metrics.get("mem_usage", 200),
                        {"service": pod_name}
                    )

                    vm_success &= send_to_victoriametrics(
                        "autoscaler_req_rate",
                        raw_metrics.get("req_rate", 10),
                        {"service": pod_name}
                    )       

                    if not vm_success:
                        print("⚠️ Some metrics failed to save to VictoriaMetrics — falling back to local logs")
                    else:
                        print(f"📊 Logged decision to VictoriaMetrics for {pod_name}")

                # Log feature values
                CPU_USAGE_GAUGE.labels(service=pod_name).set(raw_metrics["cpu_usage_lag_1"])
                MEM_USAGE_GAUGE.labels(service=pod_name).set(raw_metrics["mem_usage"])
                REQ_RATE_GAUGE.labels(service=pod_name).set(raw_metrics["req_rate"])    
                record_live_data(pod_name, input_row, decision)

                # Step 7: Apply scaling
                print(f"🛠️  Step 7: Step 7: Apply scaling : {pod_name}")
                apply_k8s_scaling(pod_name, namespace, decision)

                # Step 8: Check if retraining is needed
                if len(metrics_history[key]) >= RETRAIN_THRESHOLD:
                    print(f"🛠️  Step 8: Retrain the all Models  : {pod_name}")
                    retrain_all_models()  # Call the retraining function

            except KeyError as ke:
                print(f" Invalid config: missing key '{ke}'")
            except Exception as e:
                print(f" Error processing {pod_name}: {e}")

        time.sleep(60)

except KeyboardInterrupt:
    print("\n Autoscaler stopped. Saving final history...")
