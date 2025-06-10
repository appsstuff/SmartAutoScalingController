import os
import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


MODELS_PATH = os.getenv("MODEL_PATH", "data/models")

# ======== Model Manager with Cache ========
class ModelManager:
    def __init__(self, service_name):
        self.service_name = service_name
        self.model_path = os.getenv("MODEL_PATH", MODELS_PATH)
        self.scaler = None
        self.gpr = None
        self.xgb = None
        self.lstm = None
        self._load_all_models()

    def _load_all_models(self):
        try:
            self.scaler = joblib.load(os.path.join(self.model_path, "scaler_model.pkl"))
            self.gpr = joblib.load(os.path.join(self.model_path, "gpr_model.pkl"))
            self.xgb = xgb.XGBClassifier()
            self.xgb.load_model(os.path.join(self.model_path, "xgb_model.json"))
            self.lstm = tf.keras.models.load_model(os.path.join(self.model_path, "lstm_model.h5"))
        except Exception as e:
            print(f" Failed to load models for {self.service_name}: {e}")
            self.gpr = self.xgb = self.lstm = None

    def predict(self, features):
        """
        Predict action using hybrid voting system
        Returns: 'scale_down', 'no_change', or 'scale_up'
        """
        if any(model is None for model in [self.scaler, self.gpr, self.xgb, self.lstm]):
            return fallback_predict(features)

        X_input = np.array([features])
        X_scaled = self.scaler.transform(X_input)

        # GPR prediction
        gpr_pred, std = self.gpr.predict(X_scaled, return_std=True)
        gpr_class = self.classify_gpr(gpr_pred[0], std[0])

        # XGBoost prediction
        xgb_class = self.xgb.predict(X_scaled)[0]

        # LSTM prediction
        lstm_input = X_scaled.reshape((1, -1, 1))
        lstm_score = self.lstm.predict(lstm_input, verbose=0)[0][0]
        lstm_class = self.classify_lstm(lstm_score)

        # Fuse decisions
        decision_map = {0: "scale_down", 1: "no_change", 2: "scale_up"}
        votes = [gpr_class, xgb_class, lstm_class]
        vote_counts = {i: votes.count(i) for i in set(votes)}
        majority = [k for k, v in vote_counts.items() if v == max(vote_counts.values())]

        if len(majority) == 1:
            final = majority[0]
        else:
            # Tie-breaker by model priority
            final = next(v for v in votes if v in majority)

        return decision_map.get(final, "no_change")

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