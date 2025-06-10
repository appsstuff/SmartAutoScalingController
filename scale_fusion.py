def fuse_predictions(gpr_class: float, xgb_pred: float, lstm_class: float) -> int:
    """
    Fuse predictions from multiple models to make a scaling decision.
    Returns: 1 for scale up, -1 for scale down, 0 for no change
    """
    # Simple weighted average of predictions
    weighted_decision = (0.4 * gpr_class + 0.4 * xgb_pred + 0.2 * lstm_class)
    
    # Threshold-based decision
    if weighted_decision > 0.6:
        return 1  # Scale up
    elif weighted_decision < 0.4:
        return -1  # Scale down
    return 0  # No change 