import asyncio
from prometheus_api_client import PrometheusConnect
from shared_config import AUTO_SCALE_SERVICES

async def fetch_service_metrics(pod_name: str, namespace: str) -> dict:
    # TODO: Implement actual Prometheus metrics fetching
    # This is a placeholder that returns dummy metrics
    return {
        "cpu_usage": 0.0,
        "memory_usage": 0.0,
        "request_rate": 0.0
    }

async def run_all_services() -> list:
    AUTO_SCALE_SERVICES
    
    tasks = [fetch_service_metrics(svc["pod_name"], svc["namespace"]) for svc in AUTO_SCALE_SERVICES]
    return await asyncio.gather(*tasks) 