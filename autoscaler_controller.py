import time
import os
from datetime import datetime, timedelta
import numpy as np
from pod_config import AUTO_SCALE_SERVICES
from model_inference import predict_scaling_action
from k8s_scaler import apply_k8s_scaling
from utils import (
    fetch_pod_metrics,
    build_feature_vector,
    record_live_data,
    create_sequence,
    validate_service_config,
    load_history,
    save_history,
    fetch_historical_data
)

print(" Starting Smart AutoScaler Controller")

# Load history at startup
fetch_pod_metrics.history = load_history()

try:
    while True:
        total = len(AUTO_SCALE_SERVICES)
        print(f"\n🔄 Starting autoscaling round — {total} services")

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
                raw_metrics = fetch_pod_metrics(pod_name, namespace)

                # Step 2: Build feature vector
                input_row = build_feature_vector(raw_metrics, feature_names)
                if len(input_row) == 0:
                    print(f" No features built for {pod_name}")
                    continue

                # Step 3: Manage history for LSTM
                key = f"{namespace}/{pod_name}"

                if not hasattr(fetch_pod_metrics, 'history'):
                    fetch_pod_metrics.history = {}

                if key not in fetch_pod_metrics.history:
                    fetch_pod_metrics.history[key] = []

                # Only load historical data if history is too short
                if len(fetch_pod_metrics.history[key]) < seq_length:
                    print(f" No valid history found for {key} — fetching past data")
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
                    
                    fetch_pod_metrics.history[key] = synthetic_features
                    print(f"📊 Loaded {len(synthetic_features)} historical entries for {pod_name}")

                # Append latest metric
                fetch_pod_metrics.history[key].append(input_row)

                # Trim history to prevent memory overload
                if len(fetch_pod_metrics.history[key]) > seq_length + 10:
                    fetch_pod_metrics.history[key] = fetch_pod_metrics.history[key][-seq_length - 10:]

                # Step 4: Create sequence input for LSTM
                seq_input = create_sequence(np.array(fetch_pod_metrics.history[key]), seq_length)
                if len(seq_input) == 0:
                    print(f" Waiting — collecting initial data for {pod_name}")
                    continue

                # Step 5: Predict action
                decision = predict_scaling_action(input_row, seq_input, service_name=pod_name)
                print(f" Decision for {pod_name}: {decision}")

                # Step 6: Log for retraining — only if learning is enabled
                if os.getenv("ENABLE_RETRAINING", "true").lower() == "true":
                    record_live_data(pod_name, raw_metrics, decision)

                # Step 7: Apply Kubernetes scaling
                apply_k8s_scaling(pod_name, namespace, decision)

            except KeyError as ke:
                print(f" Invalid config: missing '{ke}'")
            except Exception as e:
                print(f" Error processing {pod_name}: {e}")

        # Save history only if retraining is enabled
        if os.getenv("ENABLE_RETRAINING", "true").lower() == "true":
            save_history(fetch_pod_metrics.history)
            print("💾 History saved for retraining")
        else:
            print(" Retraining disabled — history not saved")

        time.sleep(60)

except KeyboardInterrupt:
    print("\n Autoscaler stopped. Saving final history...")
    if os.getenv("ENABLE_RETRAINING", "true").lower() == "true":
        save_history(fetch_pod_metrics.history)
        print(" Final history saved.")
    else:
        print(" Retraining disabled — no history written")
        print(" Retraining disabled — no history written")