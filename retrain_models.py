import os
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential, save_model
from keras.layers import LSTM, Dense, Dropout
from pod_config import AUTO_SCALE_SERVICES
import logging
from utils import create_sequence, fetch_historical_data, fetch_logged_decisions
from config import (
    ENABLE_RETRAINING,
    MODELS_PATH,
    RETRAIN_THRESHOLD,
    RETRAIN_COOLDOWN
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

tf.config.optimizer.set_jit(True)

# Global dictionary to track retrain timestamps per pod
retrain_timestamps = {}

class ModelTrainer:
    def __init__(self, service_config):
        self.pod_name = service_config["pod_name"]
        self.namespace = service_config.get("namespace", "default")
        self.seq_length = service_config.get("seq_length", 10)
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self):
        path = f"../data/live_data/{self.pod_name}_live_data.csv"
        if not os.path.exists(path):
            logging.warning(f"No data found for {self.pod_name}")
            return None, None

        df = pd.read_csv(path)
        feature_columns = [
            "hour_of_day", "day_of_week", "cpu_usage_lag_1", "cpu_usage_lag_5",
            "cpu_roll_mean_10", "mem_usage", "req_rate",
            "latency", "net_receive_KB", "net_transmit_KB", "pod_restarts", "pod_ready"
        ]
        X = df[feature_columns].values
        y = df["decision"].map({"scale_down": 0, "no_change": 1, "scale_up": 2}).values
        return X, y

    def train_all(self):
        X, y = self.load_data()
        if X is None or len(X) < 50:
            logging.info(f"Skipping {self.pod_name}: not enough data")
            return

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, os.path.join(self.model_dir, f"scaler_model_{self.pod_name}.pkl"))

        # Train all models
        self.train_gpr(X_scaled, y)
        self.train_xgb(X_scaled, y)
        self.train_lstm(X_scaled, y)
        logging.info(f"Retraining completed for {self.pod_name}")

    def train_gpr(self, X, y):
        gpr = GaussianProcessRegressor()
        gpr.fit(X, y)
        joblib.dump(gpr, os.path.join(self.model_dir, f"gpr_model_{self.pod_name}.pkl"))

    def train_xgb(self, X, y):
        clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        clf.fit(X, y)
        clf.save_model(os.path.join(self.model_dir, f"xgb_model_{self.pod_name}.json"))

    def train_lstm(self, X, y):
        X_seq = create_sequence(X, self.seq_length)
        if len(X_seq) == 0:
            logging.info(f"Not enough sequence data for LSTM — skipping {self.pod_name}")
            return

        y_seq = y[-len(X_seq):]

        model = Sequential([
            LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_seq, y_seq, epochs=5, batch_size=16, verbose=0)
        model.save(os.path.join(self.model_dir, f"lstm_model_{self.pod_name}.h5"))

def retrain_all_models():
    logging.info("Starting retraining process using VictoriaMetrics data...")
    
    for svc in AUTO_SCALE_SERVICES:
        pod_name = svc["pod_name"]
        namespace = svc.get("namespace", "default")
        seq_length = svc.get("seq_length", 10)

        # Check if enough time has passed since the last retrain
        now = time.time()
        if pod_name in retrain_timestamps and (now - retrain_timestamps[pod_name]) < RETRAIN_COOLDOWN:
            logging.info(f"Cooldown period active for {pod_name}. Skipping retraining.")
            continue

        logging.info(f"Fetching data for {pod_name} ({namespace})")

        # Step 1: Get feature vector from VM
        X_train = fetch_historical_data(pod_name, namespace, days=10)
        if len(X_train) < RETRAIN_THRESHOLD:
            logging.warning(f"Not enough CPU data for {pod_name} — skipping retraining")
            continue

        # Step 2: Build synthetic features
        synthetic_features = []
        for v in X_train[-seq_length:]:
            synthetic_features.append(np.array([
                datetime.now().hour,
                datetime.now().weekday(),
                v * 0.9,  # cpu_usage_lag_1
                v * 0.7,  # cpu_usage_lag_5
                v * 0.8,  # cpu_roll_mean_10
                np.random.uniform(200, 300),  # mem_usage fallback
                np.random.uniform(5, 20)       # req_rate fallback
            ]))
        X_train = np.array(synthetic_features)

        # Step 3: Get decisions from VM
        y_train = fetch_logged_decisions(pod_name, namespace, days=10)
        if len(y_train) == 0:
            logging.warning(f"No decisions found for {pod_name} — using synthetic labels")
            y_train = np.random.randint(0, 3, size=len(X_train))

        # Step 4: Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)  # Use fit_transform for new data

        # Step 5: Create sequences for LSTM
        X_seq = create_sequence(X_scaled, seq_length)
        if len(X_seq) == 0:
            logging.warning(f"Not enough sequence data for {pod_name}")
            continue

        y_seq = y_train[-len(X_seq):]

        # Step 6: Retrain all models
        logging.info(f"Retraining all models for {pod_name}")
        trainer = ModelTrainer(svc)
        trainer.train_all()

        # Update the retrain timestamp
        retrain_timestamps[pod_name] = now
        logging.info(f"Models retrained for {pod_name}")

if __name__ == "__main__":
    if ENABLE_RETRAINING:
        logging.info("Starting bulk model retraining...")
        retrain_all_models()
