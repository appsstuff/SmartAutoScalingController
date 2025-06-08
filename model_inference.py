import os
import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
from models.gpr_model import predict_with_gpr
from models.xgb_model import predict_with_xgb
from models.lstm_model import predict_with_lstm
from sklearn.preprocessing import StandardScaler

# ======== Model Manager with Cache ========
class ModelManager:
    def __init__(self, service_name):
        self.service_name = service_name
        self.model_path = os.path.join("models", service_name)
        self._load_all_models()

    def _load_all_models(self):
        self.scaler = joblib.load(os.path.join(self.model_path, "scaler_model.pkl"))
        self.gpr = joblib.load(os.path.join(self.model_path, "gpr_model.pkl"))
        self.clf = xgb.XGBClassifier()
        self.clf.load_model(os.path.join(self.model_path, "xgb_model.json"))
        self.lstm = tf.keras.models.load_model(os.path.join(self.model_path, "lstm_model.h5"))

    def predict(self, features):
        X_input = np.array([features])
        X_scaled = self.scaler.transform(X_input)

        # GPR Prediction
        gpr_pred, std = self.gpr.predict(X_scaled, return_std=True)
        gpr_class = self.classify_gpr(gpr_pred[0], std[0])

        # XGBoost Prediction
        xgb_pred = self.clf.predict(X_scaled)[0]

        # LSTM Prediction
        lstm_input = X_scaled.reshape((1, X_scaled.shape[0], 1))
        lstm_score = self.lstm.predict(lstm_input)[0][0]
        lstm_class = self.classify_lstm(lstm_score)

        return fuse_predictions(gpr_class, xgb_pred, lstm_class)

    @staticmethod
    def classify_gpr(pred, std):
        if pred > 0.6:
            return 2  # scale_up
        elif pred < 0.3:
            return 0  # scale_down
        else:
            return 1  # no_change

    @staticmethod
    def classify_lstm(score):
        if score > 0.6:
            return 2
        elif score < 0.3:
            return 0
        else:
            return 1

# ======== Prediction Fusion Logic ========
def fuse_predictions(gpr_pred, xgb_pred, lstm_pred):
    decision_map = {0: "scale_down", 1: "no_change", 2: "scale_up"}
    votes = [gpr_pred, xgb_pred, lstm_pred]
    vote_counts = {i: votes.count(i) for i in set(votes)}
    majority = [k for k, v in vote_counts.items() if v == max(vote_counts.values())]

    if len(majority) == 1:
        final = majority[0]
    else:
        # Tie-breaker by model priority
        for model_vote in [gpr_pred, xgb_pred, lstm_pred]:
            if model_vote in majority:
                final = model_vote
                break

    return decision_map[final]

# ======== Main Entry Point ========
def predict_scaling_action(features, service_name):
    manager = ModelManager(service_name)
    return manager.predict(features)

def run_gpr_model(features):
    return predict_with_gpr(features)

def run_xgb_model(features):
    return predict_with_xgb(features)

def run_lstm_model(features):
    return predict_with_lstm(features)
