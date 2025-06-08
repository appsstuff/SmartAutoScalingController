# config_multi.py
AUTO_SCALE_SERVICES = [
    {
        "pod_name": "adservice",
        "namespace": "default",
        "min_replicas": 2,
        "max_replicas": 10,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "day_of_week", "cpu_usage_lag_1", "cpu_usage_lag_5", "cpu_roll_mean_10", "mem_usage", "req_rate"]
    },
    {
        "pod_name": "cartservice",
        "namespace": "default",
        "min_replicas": 2,
        "max_replicas": 8,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "cpu_usage_lag_1", "cpu_roll_mean_10", "mem_usage", "req_rate"]
    },
    {
        "pod_name": "checkoutservice",
        "namespace": "default",
        "min_replicas": 3,
        "max_replicas": 10,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "cpu_usage_lag_5", "cpu_roll_mean_10", "req_rate"]
    },
    {
        "pod_name": "currencyservice",
        "namespace": "default",
        "min_replicas": 2,
        "max_replicas": 8,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "cpu_usage_lag_1", "cpu_usage_lag_5", "req_rate"]
    },
    {
        "pod_name": "emailservice",
        "namespace": "default",
        "min_replicas": 2,
        "max_replicas": 6,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "cpu_usage_lag_1", "cpu_roll_mean_10", "req_rate"]
    },
    {
        "pod_name": "frontend",
        "namespace": "default",
        "min_replicas": 3,
        "max_replicas": 10,
        "scaling_step": 2,
        "seq_length": 15,
        "feature_names": ["hour_of_day", "day_of_week", "cpu_usage_lag_1", "cpu_usage_lag_5", "cpu_roll_mean_10", "mem_usage", "req_rate"]
    },
    {
        "pod_name": "paymentservice",
        "namespace": "default",
        "min_replicas": 2,
        "max_replicas": 8,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "cpu_usage_lag_1", "cpu_roll_mean_10", "req_rate"]
    },
    {
        "pod_name": "productcatalogservice",
        "namespace": "default",
        "min_replicas": 2,
        "max_replicas": 10,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "cpu_usage_lag_1", "cpu_usage_lag_5", "cpu_roll_mean_10", "mem_usage"]
    },
    {
        "pod_name": "recommendationservice",
        "namespace": "default",
        "min_replicas": 2,
        "max_replicas": 8,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "cpu_usage_lag_5", "cpu_roll_mean_10", "mem_usage", "req_rate"]
    },
    {
        "pod_name": "redis-cart",
        "namespace": "default",
        "min_replicas": 1,
        "max_replicas": 3,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "cpu_usage_lag_1", "mem_usage", "req_rate"]
    },
    {
        "pod_name": "shippingservice",
        "namespace": "default",
        "min_replicas": 2,
        "max_replicas": 8,
        "scaling_step": 1,
        "seq_length": 10,
        "feature_names": ["hour_of_day", "cpu_usage_lag_1", "cpu_roll_mean_10", "req_rate"]
    }
]

if __name__ == "__main__":
    print("📋 AUTO_SCALE_SERVICES:")
    for svc in AUTO_SCALE_SERVICES:
        print(svc)