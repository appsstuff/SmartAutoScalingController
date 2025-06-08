import numpy as np

def preprocess_metrics(raw_metrics: dict) -> np.ndarray:
    """
    Convert raw Prometheus metrics into ML features.
    Args:
        raw_metrics: Dictionary containing raw metrics (cpu_usage, memory_usage, request_rate)
    Returns:
        numpy array of preprocessed features
    """
    # Extract and normalize features
    features = np.array([
        raw_metrics['cpu_usage'],
        raw_metrics['memory_usage'],
        raw_metrics['request_rate']
    ])
    
    # Normalize features to [0, 1] range
    features = (features - features.min()) / (features.max() - features.min() + 1e-8)
    
    return features 