import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model

target_names = ["scale_down", "no_change", "scale_up"]

def load_models(pod_name):
    """Load pod-specific models"""
    gpr = joblib.load(f"models/gpr_model_{pod_name}.pkl")
    clf = XGBClassifier()
    clf.load_model(f"models/xgb_model_{pod_name}.json")
    model_lstm = load_model(f"models/lstm_model_{pod_name}.h5")
    scaler = joblib.load(f"models/scaler_{pod_name}.pkl")
    return gpr, clf, model_lstm, scaler

def predict_scaling_action(input_row, seq_input):
    """
    Predict scaling action using hybrid model
    """
    pod_name = os.getenv("TARGET_DEPLOYMENT", "adservice")
    gpr, clf, model_lstm, scaler = load_models(pod_name)

    input_scaled = scaler.transform([input_row])
    seq_scaled = scaler.transform(seq_input.reshape(-1, seq_input.shape[-1])).reshape(seq_input.shape)

    # GPR Prediction
    gpr_pred, std = gpr.predict(input_scaled, return_std=True)
    threshold_up = np.percentile(gpr_pred, 90)
    threshold_down = np.percentile(gpr_pred, 10)
    gpr_class = np.where(gpr_pred > threshold_up, 0,
                         np.where(gpr_pred < threshold_down, 2, 1))

    # XGBoost Prediction
    xgb_pred = clf.predict(input_scaled)[0]

    # LSTM Prediction
    lstm_pred = model_lstm.predict(seq_scaled, verbose=0).flatten()
    lstm_class = np.where(lstm_pred.mean() > threshold_up, 0,
                          np.where(lstm_pred.mean() < threshold_down, 2, 1))

    final_class = int(np.round((gpr_class[0] + xgb_pred + lstm_class) / 3))
    final_decision = fuse_predictions(gpr_class, xgb_pred, lstm_class)

    return target_names[final_decision]


def fuse_predictions(gpr_pred, xgb_pred, lstm_pred):
    """
    Combines predictions from GPR, XGB, and LSTM models
    Returns: final_decision (str)
    """
    decision_map = {0: "scale_down", 1: "no_change", 2: "scale_up"}
    votes = [gpr_pred, xgb_pred, lstm_pred]
    
    # Count occurrences
    vote_counts = {i: votes.count(i) for i in set(votes)}
    majority = [k for k, v in vote_counts.items() if v == max(vote_counts.values())]

    if len(majority) == 1:
        final = majority[0]
    else:
        # Tie: use model priority fallback
        priority = [gpr_pred, xgb_pred, lstm_pred]
        for model_vote in priority:
            if model_vote in majority:
                final = model_vote
                break

    return decision_map[final]
