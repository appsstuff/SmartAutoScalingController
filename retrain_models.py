import os
import numpy as np
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from tensorflow.keras.models import load_model, save_model
from utils import fetch_pod_metrics, build_feature_vector, create_sequence
from config_multi import AUTO_SCALE_SERVICES

print("🔄 Starting bulk model retraining process")

def collect_live_data(svc_config):
    """
    Collects recent metrics from VictoriaMetrics instead of CSV
    Returns synthetic training data for retraining
    """
    pod_name = svc_config["pod_name"]
    namespace = svc_config.get("namespace", "default")
    feature_names = svc_config.get("feature_names", [])
    seq_length = svc_config.get("seq_length", 10)

    print(f"🔄 Fetching live data for {pod_name}")
    raw_metrics = fetch_pod_metrics(pod_name, namespace)
    input_row = build_feature_vector(raw_metrics, feature_names)

    # Simulate label generation using hybrid logic
    decision = predict_scaling_action(input_row, create_sequence(input_row, seq_length))
    y_new = {"scale_down": 0, "no_change": 1, "scale_up": 2}[decision]
    return np.array([input_row]), np.array([y_new])

def retrain_for_service(svc_config, index):
    """
    Retrain all models for one service using live metrics
    """
    if "pod_name" not in svc_config:
        print(f" Skipping service #{index}: missing 'pod_name'")
        return

    pod_name = svc_config["pod_name"]
    namespace = svc_config.get("namespace", "default")
    feature_names = svc_config.get("feature_names", [])
    seq_length = svc_config.get("seq_length", 10)

    try:
        # Load pod-specific models
        gpr = joblib.load(f"models/gpr_model_{pod_name}.pkl")
        clf = XGBClassifier()
        clf.load_model(f"models/xgb_model_{pod_name}.json")
        model_lstm = load_model(f"models/lstm_model_{pod_name}.h5")
        scaler = joblib.load(f"models/scaler_{pod_name}.pkl")

        # Get live data
        X_new, y_new = collect_live_data(svc_config)
        if X_new is None or y_new is None:
            print(f"❌ No valid data for {pod_name}")
            return

        # Scale new data
        X_scaled = scaler.transform(X_new)

        # Retrain GPR
        print(f"🧠 Retraining GPR for {pod_name}")
        gpr.fit(X_scaled, y_new)
        joblib.dump(gpr, f"models/gpr_model_{pod_name}.pkl")

        # Retrain XGBoost
        print(f"🌲 Retraining XGBoost for {pod_name}")
        clf.fit(X_scaled, y_new)
        clf.save_model(f"models/xgb_model_{pod_name}.json")

        # Retrain LSTM
        from utils import create_sequence
        print(f"🧠 Retraining LSTM for {pod_name}")
        seq_input = create_sequence(X_scaled, seq_length)
        model_lstm.fit(seq_input, y_new, epochs=5, batch_size=32, verbose=0)
        model_lstm.save(f"models/lstm_model_{pod_name}.h5")

        print(f"✅ Models updated for {pod_name}")

    except Exception as e:
        print(f"❌ Failed to retrain for {pod_name}: {e}")

if __name__ == "__main__":
    for idx, svc in enumerate(AUTO_SCALE_SERVICES):
        retrain_for_service(svc, idx)