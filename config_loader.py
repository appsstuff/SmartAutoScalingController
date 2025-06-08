# import os

# # Load global config
# PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090/api/v1/query")
# CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))
# RETRAIN_INTERVAL_MINUTES = int(os.getenv("RETRAIN_INTERVAL_MINUTES", "1440"))

# # Load service-specific config
# NAMESPACE = os.getenv("NAMESPACE", "default")
# TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "adservice")
# MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "2"))
# MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "10"))
# SCALING_STEP = int(os.getenv("SCALING_STEP", "1"))
# SEQ_LENGTH = int(os.getenv("SEQ_LENGTH", "10"))

# FEATURE_NAMES = os.getenv("FEATURE_NAMES", "").split(",") or [
#     "hour_of_day",
#     "day_of_week",
#     "cpu_usage_lag_1",
#     "cpu_usage_lag_5",
#     "cpu_roll_mean_10",
#     "mem_usage",
#     "req_rate"
# ]