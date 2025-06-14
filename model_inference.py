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
        self.model_path = os.getenv("MODELS_PATH", MODELS_PATH)
        self.scaler = None
        self.gpr = None
        self.xgb = None
        self.lstm = None
        self._load_all_models()
        
        
    def _load_all_models(self):
        try:
            # Load Scaler
            scaler_path = os.path.join(self.model_path, "scaler_model.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                raise FileNotFoundError("Scaler not found")

            # Load GPR
            gpr_path = os.path.join(self.model_path, "gpr_model.pkl")
            if os.path.exists(gpr_path):
                self.gpr = joblib.load(gpr_path)
            else:
                raise FileNotFoundError("GPR model not found")

            # Load XGBoost
            xgb_path = os.path.join(self.model_path, "xgb_model.json")
            if os.path.exists(xgb_path):
                self.xgb = xgb.XGBClassifier()
                self.xgb.load_model(xgb_path)
            else:
                raise FileNotFoundError("XGBoost model not found")

            # Load LSTM
            lstm_path = os.path.join(self.model_path, "lstm_model.h5")
            if os.path.exists(lstm_path):
                self.lstm = tf.keras.models.load_model(lstm_path)
            else:
                raise FileNotFoundError("LSTM model not found")

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
            print(f"[{self.service_name}] Hybrid model failed — falling back to threshold prediction")
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
            if seq_input is not None and len(seq_input) >= self.seq_length:
                lstm_input = seq_input.reshape((1, -1, 1))
                lstm_score = self.lstm.predict(lstm_input, verbose=0)[0][0]
                lstm_class = 0 if lstm_score < 0.3 else (2 if lstm_score > 0.7 else 1)
            else:
                lstm_class = 1  # Default to no change

            # Fuse decisions
            votes = [gpr_class, xgb_class, lstm_class]
            vote_counts = {v: votes.count(v) for v in set(votes)}
            majority_votes = max(vote_counts.values())
            candidates = [k for k, v in vote_counts.items() if v == majority_votes]

            # Tie-breaker: use first valid vote (prioritize GPR > XGB > LSTM)
            for vote in [gpr_class, xgb_class, lstm_class]:
                if vote in candidates:
                    decision = {0: "scale_down", 1: "no_change", 2: "scale_up"}[vote]
                    print(f"[{self.service_name}] Hybrid decision used → {decision}")
                    return decision

        except Exception as e:
            print(f"[{self.service_name}] Hybrid prediction error: {e}")
            return self.fallback_predict(input_row)
        
  
    def fallback_predict(self, features):
        """
        Fallback strategy based on raw CPU usage
        """
        try:
            cpu_usage = float(features[2])  # Ensure numeric value
            if cpu_usage > 0.8:
                return 'scale_up'
            elif cpu_usage < 0.2:
                return 'scale_down'
            else:
                return 'no_change'
        except Exception as e:
            print(f"[Fallback] Invalid feature vector: {e}")
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

# ======== Prediction Fusion Logic ========

def fuse_predictions(self, gpr_class: int, xgb_class: int, lstm_class: int, input_row) -> str:
    """
    Fuse predictions from GPR, XGBoost, and LSTM using majority vote.
    Tie-breaker uses model priority: GPR > XGB > LSTM
    """
    decision_map = {0: "scale_down", 1: "no_change", 2: "scale_up"}
    votes = [gpr_class, xgb_class, lstm_class]
    
    # Ensure votes are valid integers
    valid_votes = [v for v in votes if isinstance(v, int)]
    
    if not valid_votes:
        print("⚠️ No valid votes — falling back to threshold")
        return self.fallback_predict(input_row)
    
    vote_counts = {v: valid_votes.count(v) for v in set(valid_votes)}
    max_votes = max(vote_counts.values())
    candidates = [k for k, v in vote_counts.items() if v == max_votes]

    # Tie-breaker: use first valid vote (or prioritize certain models)
    for vote in votes:
        if vote in candidates:
            return decision_map.get(vote, "no_change")

    return "no_change"

def predict_scaling_action(input_row, seq_input=None, service_name=None):
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