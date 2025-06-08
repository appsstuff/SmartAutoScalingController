import time
import os
from datetime import datetime, timedelta
import numpy as np
from config_multi import AUTO_SCALE_SERVICES
from k8s_scaler import apply_k8s_scaling
from model_inference import predict_scaling_action
from utils import (
    fetch_pod_metrics,
    build_feature_vector,
    record_live_data,
    create_sequence,
    validate_service_config,
    load_history,
    save_history
)



# Inside loop, after fetching metrics
ENABLE_RETRAINING = os.getenv("ENABLE_RETRAINING", "false").lower() == "true"

if ENABLE_RETRAINING:
    save_history(fetch_pod_metrics.history)
else:
    print(" Starting Smart AutoScaler Controller")

    fetch_pod_metrics.history = load_history()

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

                    print(f"[{'#' * (percent // 5)}{'-' * ((100 - percent) // 5)}] {percent}% — Processing {pod_name}")

                    # Step 1: Fetch live metrics
                    raw_metrics = fetch_pod_metrics(pod_name, namespace)

                    # Step 2: Build feature vector
                    input_row = build_feature_vector(raw_metrics, feature_names)
                    if len(input_row) == 0:
                        print(f" No features built for {pod_name}")
                        continue

                    # Step 3: Manage history for LSTM
                    key = f"{namespace}/{pod_name}"
                    if key not in fetch_pod_metrics.history:
                        fetch_pod_metrics.history[key] = []

                    fetch_pod_metrics.history[key].append(input_row)
                    if len(fetch_pod_metrics.history[key]) > seq_length + 10:
                        fetch_pod_metrics.history[key] = fetch_pod_metrics.history[key][-seq_length - 10:]

                    # Step 4: Create sequence input for LSTM
                    seq_input = create_sequence(np.array(fetch_pod_metrics.history[key]), seq_length)
                    if len(seq_input) == 0:
                        print(f" Waiting — collecting initial data for {pod_name}")
                        continue

                    # Step 5: Predict action
                    decision = predict_scaling_action(input_row, seq_input)
                    print(f" Decision for {pod_name}: {decision}")

                    # Step 6: Log for retraining
                    record_live_data(pod_name, raw_metrics, decision)
                    apply_k8s_scaling(pod_name, namespace, decision)
                except KeyError as ke:
                    print(f" Invalid config: missing '{ke}'")
                except Exception as e:
                    print(f" Error processing {pod_name}: {e}")

            # Save history at end of loop
            save_history(fetch_pod_metrics.history)
            print(" Final history saved.")

            time.sleep(60)

    except KeyboardInterrupt:
        print("\n Autoscaler stopped. Saving final history...")
        save_history(fetch_pod_metrics.history)
        print(" Final history saved.")