import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, save_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pod_config import AUTO_SCALE_SERVICES

ENABLE_RETRAINING = os.getenv("ENABLE_RETRAINING", "false").lower() == "true"

class ModelTrainer:
    def __init__(self, service_config):
        self.pod_name = service_config["pod_name"]
        self.namespace = service_config.get("namespace", "default")
        self.seq_length = service_config.get("seq_length", 10)
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self):
        path = f"/data/live_data/{self.pod_name}_live_data.csv"
        if not os.path.exists(path):
            print(f"[WARN] No data found for {self.pod_name}")
            return None, None

        df = pd.read_csv(path)
        feature_columns = [
            "hour_of_day", "day_of_week", "cpu_usage_lag_1", "cpu_usage_lag_5",
            "cpu_roll_mean_10", "mem_usage", "req_rate"
        ]
        X = df[feature_columns].values
        y = df["decision"].map({"scale_down": 0, "no_change": 1, "scale_up": 2}).values
        return X, y

    def train_all(self):
        X, y = self.load_data()
        if X is None or len(X) < 50:
            print(f"[INFO] Skipping {self.pod_name}: not enough data")
            return

        self.train_scaler(X)
        self.train_gpr(X, y)
        self.train_xgb(X, y)
        self.train_lstm(X, y)
        print(f"[DONE] Retraining completed for {self.pod_name}")

    def train_scaler(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, os.path.join(self.model_dir, f"scaler_model_{self.pod_name}.pkl"))
        self.X_scaled = X_scaled

    def train_gpr(self, X, y):
        gpr = GaussianProcessRegressor()
        gpr.fit(self.X_scaled, y)
        joblib.dump(gpr, os.path.join(self.model_dir, f"gpr_model_{self.pod_name}.pkl"))

    def train_xgb(self, X, y):
        clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        clf.fit(self.X_scaled, y)
        clf.save_model(os.path.join(self.model_dir, f"xgb_model_{self.pod_name}.json"))

    def train_lstm(self, X, y):
        X_seq = self.X_scaled.reshape((self.X_scaled.shape[0], self.X_scaled.shape[1], 1))
        y_seq = y[:X_seq.shape[0]]

        model = Sequential()
        model.add(LSTM(64, input_shape=(X_seq.shape[1], 1), return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_seq, y_seq, epochs=5, batch_size=16, verbose=0)

        model.save(os.path.join(self.model_dir, f"lstm_model_{self.pod_name}.h5"))

# Entry point for retraining all services
if __name__ == "__main__":
    
    if ENABLE_RETRAINING:
        print(" Starting bulk model retraining...")
        for svc in AUTO_SCALE_SERVICES:
            trainer = ModelTrainer(svc)
            trainer.train_all()
