import os
import time
import numpy as np
import logging
import joblib
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config import ENABLE_RETRAINING, MODELS_PATH, RETRAIN_COOLDOWN
from utils import fetch_historical_data, create_sequence
from pod_config import AUTO_SCALE_SERVICES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable Intel optimizations
tf.config.optimizer.set_jit(True)

# Global dictionary to track retrain timestamps per pod
retrain_timestamps = {}

class ModelTrainer:
    def __init__(self, service_config):
        self.pod_name = service_config["pod_name"]
        self.namespace = service_config.get("namespace", "default")
        self.seq_length = service_config.get("seq_length", 10)
        self.model_dir = os.path.join(MODELS_PATH, self.pod_name)
        os.makedirs(self.model_dir, exist_ok=True)

    def train_gpr(self, X_train, y_train):
        """
        Train Gaussian Process Regressor on historical data
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF
            
            kernel = RBF(length_scale=1.0)
            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
            model.fit(X_train, y_train)
            model_path = os.path.join(self.model_dir, "gpr_model.pkl")
            joblib.dump(model, model_path)
            print(f"🧠 GPR model retrained for {self.pod_name}")
        except Exception as e:
            print(f"⚠️ Failed to train GPR model for {self.pod_name}: {e}")

    def train_xgb(self, X_train, y_train):
        """
        Train XGBoost regressor using CPU usage features
        """
        try:
            model = XGBRegressor(n_estimators=100, learning_rate=0.1)
            model.fit(X_train, y_train)
            model.save_model(os.path.join(self.model_dir, "xgb_model.json"))
            print(f"🧠 XGBoost model retrained for {self.pod_name}")
        except Exception as e:
            print(f"⚠️ Failed to train XGBoost model for {self.pod_name}: {e}")

    def train_lstm(self, X_seq, y_seq):
        """
        Train LSTM model for time-series prediction
        """
        try:
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(X_seq.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)
            model.save(os.path.join(self.model_dir, "lstm_model.h5"))
            print(f"🧠 LSTM model retrained for {self.pod_name}")
        except Exception as e:
            print(f"⚠️ Failed to train LSTM model for {self.pod_name}: {e}")

    def train_all(self):
        """
        Main method to trigger model retraining
        """
        try:
            raw_data = fetch_historical_data(self.pod_name, self.namespace, days=7)
            if len(raw_data) < 100:
                print(f"⚠️ Not enough historical data for {self.pod_name} — skipping retraining")
                return

            # Convert to NumPy array
            X_train = np.array(raw_data)
            if len(X_train.shape) != 2 or X_train.shape[1] == 0:
                raise ValueError(f"Invalid shape {X_train.shape} — expected 2D")

            # Assume CPU usage is the target variable
            y_train = X_train[:, 2]  # Index 2 → cpu_usage_lag_1
            X_train = X_train[:, :-1]  # All other metrics are features

            # Normalize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            joblib.dump(scaler, os.path.join(self.model_dir, "scaler_model.pkl"))

            # Train models
            self.train_gpr(X_scaled, y_train)
            self.train_xgb(X_scaled, y_train)

            # Prepare LSTM data
            X_seq = create_sequence(X_scaled, self.seq_length)
            y_seq = y_train[-len(X_seq):]

            if len(X_seq) > 0:
                self.train_lstm(X_seq, y_seq)
            else:
                print("🪲 Not enough sequence data for LSTM")

        except Exception as e:
            print(f"🚫 Retraining failed for {self.pod_name}: {e}")


def retrain_all_models():
    """
    Run model retraining for all services defined in AUTO_SCALE_SERVICES
    Only trains each service once per cooldown period
    """
    global retrain_timestamps
    now = time.time()

    for svc in AUTO_SCALE_SERVICES:
        pod_name = svc["pod_name"]
        namespace = svc.get("namespace", "default")
        seq_length = svc.get("seq_length", 10)

        if not ENABLE_RETRAINING:
            logging.info(f"Retraining disabled via config — skipping {pod_name}")
            continue

        if pod_name in retrain_timestamps and (now - retrain_timestamps[pod_name]) < RETRAIN_COOLDOWN:
            logging.info(f"Cooldown active for {pod_name}. Skipping.")
            continue

        logging.info(f"🔄 Starting retraining for {pod_name}")
        trainer = ModelTrainer(svc)
        trainer.train_all()
        retrain_timestamps[pod_name] = now
        logging.info(f"✅ Models retrained for {pod_name}")