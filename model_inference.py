import os
import numpy as np
import tensorflow as tf
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler


from config import (
MODELS_PATH
)

# Enable Intel optimizations
tf.config.optimizer.set_jit(True)

print("Using optimized TF:", tf.__version__)


# ======== Model Manager with Cache ========
class ModelManager:
    def __init__(self, service_name):
        self.service_name = service_name
        self.model_path = os.getenv("", MODELS_PATH)
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
                print(f"⚠️ Failed to load hybrid model: {e}")
                self.gpr = self.xgb = self.lstm = None
                self.scaler = StandardScaler()  # Use fallback scaler if odel.h5"))
            
            except Exception as e:
                print(f" Failed to load models for {self.service_name}: {e}")
                self.gpr = self.xgb = self.lstm = None

    def predict(self, input_row, seq_input=None):
            """
            Predict scaling action using hybrid voting system.
            Returns: 'scale_down', 'no_change', or 'scale_up'
            """
            if not all([self.scaler, self.gpr, self.xgb, self.lstm]):
                print(f"[{self.service_name}] Hybrid model failed — falling back to threshold prediction")
                return self.fallback_predict(input_row)

            try:
                X_input = np.array([input_row])
                X_scaled = self.scaler.transform(X_input)

                # GPR Prediction
                gpr_pred, std = self.gpr.predict(X_scaled, return_std=True)
                threshold_up = np.percentile(gpr_pred + std, 90)
                threshold_down = np.percentile(gpr_pred - std, 10)
                gpr_class = 0 if gpr_pred > threshold_up else (2 if gpr_pred < threshold_down else 1)

                # XGBoost Prediction
                xgb_class = int(self.xgb.predict(X_scaled)[0])

                # LSTM Prediction
                if seq_input is not None:
                    lstm_input = X_scaled.reshape((1, -1, 1))
                    lstm_score = self.lstm.predict(lstm_input, verbose=0)[0][0]
                    lstm_class = 0 if lstm_score > threshold_up else (2 if lstm_score < threshold_down else 1)
                else:
                    lstm_class = 1  # Default to no change

                # Fuse decisions
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
        """
        Fallback strategy based on simple thresholds
        """
        cpu_usage = features[2]  # Assuming index 2 is cpu_usage_lag_1
        if cpu_usage > 0.8:
            return 'scale_up'
        elif cpu_usage < 0.2:
            return 'scale_down'
        else:
            return 'no_change'
    @staticmethod
    def classify_gpr(pred, std):
        threshold_up = pred + std * 0.5
        if threshold_up > 0.7:
            return 2
        elif pred < 0.3:
            return 0
        return 1

    @staticmethod
    def classify_lstm(score):
        if score > 0.6:
            return 2
        elif score < 0.3:
            return 0
        return 1


# ======== Fallback Strategy (No Models Found) ========
def fallback_predict(features):
    """
    Fallback prediction based on raw CPU usage.
    Use this if models are not available.
    """
    cpu_usage = features[2]  # cpu_usage_lag_1
    if cpu_usage > 0.8:
        return 'scale_up'
    elif cpu_usage < 0.2:
        return 'scale_down'
    return 'no_change'


# ======== Prediction Fusion Logic ========
def fuse_predictions(gpr_class, xgb_class, lstm_class):
    decision_map = {0: "scale_down", 1: "no_change", 2: "scale_up"}
    votes = [gpr_class, xgb_class, lstm_class]
    vote_counts = {i: votes.count(i) for i in set(votes)}
    majority = [k for k, v in vote_counts.items() if v == max(vote_counts.values())]

    if len(majority) == 1:
        return decision_map[majority[0]]
    else:
        # Tie-breaker by model priority
        for vote in [gpr_class, xgb_class, lstm_class]:
            if vote in majority:
                return decision_map[vote]
            
def predict_scaling_action(input_row, seq_input, service_name=None):
    """
    Predict scaling action using hybrid model or fallback strategy.
    
    Args:
        input_row (np.ndarray): Current feature vector
        seq_input (np.ndarray): Time-series sequence for LSTM
        service_name (str): Optional — use specific model for this service
    
    Returns:
        str: 'scale_down', 'no_change', or 'scale_up'
    """
    pod_name = service_name or os.getenv("TARGET_DEPLOYMENT", "adservice")
    try:
        manager = ModelManager(pod_name)
        decision = manager.predict(input_row)
        print(f"[{pod_name}] Hybrid decision used")
        return decision
    except Exception as e:
        print(f" [{pod_name}] Failed to load hybrid model: {e}")
        return fallback_predict(input_row)