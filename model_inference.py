import os
import numpy as np
import xgboost as xgb
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler
from config import MODELS_PATH

class ModelManager:
    def __init__(self, service_name):
        self.service_name = service_name
        self.model_path = MODELS_PATH
        self.scaler = None
        self.gpr = None
        self.xgb = None
        self.lstm = None
        self._load_all_models()

    def _load_all_models(self):
        try:
            print(f"🧠 Loading models for {self.service_name}")
            
            # Load Scaler
            scaler_path = os.path.join(self.model_path, "scaler_model.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")

            # Load GPR
            gpr_path = os.path.join(self.model_path, "gpr_model.pkl")
            if os.path.exists(gpr_path):
                self.gpr = joblib.load(gpr_path)
            else:
                raise FileNotFoundError(f"GPR model not found at {gpr_path}")

            # Load XGBoost
            xgb_path = os.path.join(self.model_path, "xgb_model.json")
            if os.path.exists(xgb_path):
                self.xgb = xgb.XGBClassifier()
                self.xgb.load_model(xgb_path)
            else:
                raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")

            # Load LSTM
            lstm_path = os.path.join(self.model_path, "lstm_model.h5")
            if os.path.exists(lstm_path):
                self.lstm = tf.keras.models.load_model(lstm_path)
            else:
                raise FileNotFoundError(f"LSTM model not found at {lstm_path}")

            print(f"✅ Models loaded for {self.service_name}")

        except Exception as e:
            print(f"[{self.service_name}] Failed to load models: {e}")
            self.scaler = StandardScaler()
            self.gpr = None
            self.xgb = None
            self.lstm = None


    def predict(self, input_row, seq_input=None):
        """
        Predict scaling action using hybrid voting system.
        Returns: 'scale_down', 'no_change', or 'scale_up'
        """
        if not all([self.scaler, self.gpr, self.xgb, self.lstm]):
            print(f"[Fallback] Not all models loaded → using threshold prediction")
            return self.fallback_predict(input_row)

        try:
            X_input = np.array([input_row])
            X_scaled = self.scaler.transform(X_input)

            # GPR Prediction
            gpr_pred, std = self.gpr.predict(X_scaled, return_std=True)
            gpr_score = gpr_pred[0] if isinstance(gpr_pred, np.ndarray) else gpr_pred
            threshold_up = gpr_score + std
            threshold_down = gpr_score - std

            gpr_class = 0 if gpr_score < threshold_down else (2 if gpr_score > threshold_up else 1)

            # XGBoost Prediction
            xgb_pred = self.xgb.predict(X_scaled)[0]
            xgb_class = int(xgb_pred)

            # LSTM Prediction
            if seq_input is not None:
                lstm_input = seq_input.reshape((1, -1, 1))
                lstm_score = self.lstm.predict(lstm_input, verbose=0)[0][0]
                lstm_class = 0 if lstm_score < 0.3 else (2 if lstm_score > 0.7 else 1)
            else:
                lstm_class = 1  # Default to no change

            # Fuse predictions
            votes = [gpr_class, xgb_class, lstm_class]
            vote_counts = {v: votes.count(v) for v in set(votes)}
            majority = max(vote_counts, key=vote_counts.get)
            target_names = ['scale_down', 'no_change', 'scale_up']
            decision = target_names[majority]

            print(f"[{self.service_name}] Hybrid decision used → {decision}")
            return decision

        except Exception as e:
            print(f"[{self.service_name}] Hybrid prediction error: {e}")
            return self.fallback_predict(input_row)

    def fallback_predict(self, features):
        cpu_usage = features[2]  # Assuming index 2 is cpu_usage_lag_1
        if cpu_usage > 0.8:
            return 'scale_up'
        elif cpu_usage < 0.2:
            return 'scale_down'
        else:
            return 'no_change'


def predict_scaling_action(input_row, seq_input, service_name=None):
    """
    Predict scaling action using hybrid model or fallback strategy
    """
    pod_name = service_name or os.getenv("TARGET_DEPLOYMENT", "adservice")
    try:
        manager = ModelManager(pod_name)
        decision = manager.predict(input_row, seq_input)
        print(f"[{pod_name}] Hybrid decision used")
        return decision
    except Exception as e:
        print(f"[{pod_name}] Failed to load hybrid model: {e}")
        return fallback_predict(input_row)


def fallback_predict(features):
    cpu_usage = features[2]  # cpu_usage_lag_1
    if cpu_usage > 0.8:
        return 'scale_up'
    elif cpu_usage < 0.2:
        return 'scale_down'
    return 'no_change'