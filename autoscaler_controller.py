#!/usr/bin/env python3
"""
SmartAutoScaler Controller – runs autoscaling loop using hybrid policy engine
"""

import os
import time
from datetime import datetime, timedelta
import numpy as np
from pod_config import AUTO_SCALE_SERVICES
from model_inference import predict_scaling_action
from retrain_models import retrain_all_models
from utils import (
    fetch_pod_metrics,
    build_feature_vector,
    send_to_victoriametrics,
    create_sequence,
    validate_service_config,
    RETRAIN_THRESHOLD, 
    load_history,
    save_history,
    record_live_data,
    metrics_history,
    validate_model_performance,
    check_vm_connection
)
from metrics_server import start_metrics_server, set_metric_values

# Load history at startup
metrics_history = load_history()
print(".................. Start Smat Auto Scaller ...............")
print(".................... Muhammad Yassein ....................")

# Start metrics server early
start_metrics_server(port=8900)

# Main loop
try:
    while True:
        total = len(AUTO_SCALE_SERVICES)
        print(f"\n🔄 Starting autoscaling round — {total} services")

        for idx, svc in enumerate(AUTO_SCALE_SERVICES):
            percent = int((idx + 1) / total * 100)
            validate_service_config(svc)

            pod_name = svc["pod_name"]
            namespace = svc.get("namespace", "default")
            seq_length = svc.get("seq_length", 10)
            feature_names = svc.get("feature_names", [])

            progress_bar = f"[{'#' * (percent // 5):<20}] {percent}% — Processing {pod_name}"
            print(f"{progress_bar}")

            # Step 1: Fetch live metrics
            print(f"\n🧩 Step 1: Fetch live metrics : {pod_name}")
            raw_metrics = fetch_pod_metrics(pod_name, namespace)
            if not raw_metrics:
                print(f"🪲 No valid metrics fetched for {pod_name}")
                continue

            # Step 2: Build feature vector
            print(f"\n🧩 Step 2: Build feature vector : {pod_name}")
            input_row = build_feature_vector(raw_metrics, feature_names)
            if len(input_row) == 0:
                print(f"🪲 No features built for {pod_name}")
                continue

            # Step 3: Manage history
            key = f"{namespace}/{pod_name}"
            if key not in metrics_history:
                metrics_history[key] = []
            # Append latest data
            metrics_history[key].append(input_row)
            # Limit history size
            max_history = seq_length + 10
            if len(metrics_history[key]) > max_history:
                metrics_history[key] = metrics_history[key][-max_history:]

            # Step 4: Create sequence for LSTM
            seq_input = create_sequence(np.array(metrics_history[key]), seq_length)
            if len(seq_input) == 0:
                print(f"⏳ Waiting — collecting initial sequence for {pod_name}")
                continue

            # Step 5: Make prediction
            print(f"\n🧩 Step 5: Make prediction : {pod_name}")
            decision = predict_scaling_action(input_row, seq_input, service_name=pod_name)
            print(f"🧠 Decision for {pod_name}: {decision}")

            # Step 6: Log to VictoriaMetrics
            decision_value = {"scale_down": 0, "no_change": 1, "scale_up": 2}.get(decision, 1)
            set_metric_values(pod_name, raw_metrics,decision_value)
            if check_vm_connection():
                print(f"\n🧩 Step 6: Log to VictoriaMetrics: {pod_name}")

                send_to_victoriametrics("autoscaler_decision", decision_value, {"service": pod_name})
                send_to_victoriametrics("autoscaler_cpu_usage", raw_metrics["cpu_usage_lag_1"], {"service": pod_name})
                send_to_victoriametrics("autoscaler_mem_usage", raw_metrics["mem_usage"], {"service": pod_name})
                send_to_victoriametrics("autoscaler_req_rate", raw_metrics["req_rate"], {"service": pod_name})
                record_live_data(pod_name, input_row, decision)
            else:
               print("⚠️ VictoriaMetrics unreachable — skipping logs")
               
               # Use KEDA instedad
            #  # Step 7: Apply scaling 
            print(f"\n🧩 Step 7: Apply scaling Using KEDA .....")
            # scaling_success = scaler.scale(pod_name, namespace, decision)
            # if not scaling_success:
            #     print(f"🚫 Failed to scale {pod_name} — will retry next round")
            # else:
            #     print(f"✅ Replicas updated for {pod_name}")    
                
            # Step 8: Record live data for future retraining
            
            record_live_data(pod_name, input_row, decision)

            # Step 9: Retrain models if needed
            print(f"\n🧩 Step 9: Retrain models if needed: {pod_name}")

            if len(metrics_history[key]) >= RETRAIN_THRESHOLD:
                print(f"🔄 Retraining models for {pod_name}")
                retrain_all_models()  # Triggers retraining for all services

                # Step 10: Validate performance after retraining
                print(f"🧪 Validating model performance for {pod_name}")
                old_accuracy = validate_model_performance(pod_name)
                if old_accuracy < 0.7:
                    print(f"📈 Model accuracy improved for {pod_name}")
                else:
                    print(f"📉 Model already performing well → skipping retraining")

            
        # Save history periodically
        save_history(metrics_history)
        time.sleep(60)

except KeyboardInterrupt:
    print("\n🛑 Autoscaler stopped. Saving final history...")
    save_history(metrics_history)
    print("✅ Final history saved.")