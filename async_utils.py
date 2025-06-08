import asyncio
import aiohttp
from datetime import datetime
import numpy as np

# http://victoriametrics.monitoring.svc.cluster.local:8428/api/v1/query

async def async_query_vm(session: aiohttp.ClientSession, query):
    """Async PromQL query"""
    try:
        async with session.get("http://prometheus.yassein.com/query/api/v1/query", params={'query': query}, ssl=False) as response:
            if response.status == 200:
                json_data = await response.json()
                result = json_data.get('data', {}).get('result', [])
                if result:
                    return float(result[0]['value'][1])
    except Exception as e:
        print(f" Async VM query failed: {e}")
    return np.random.uniform(0.1, 0.9)

async def fetch_pod_metrics_async(session: aiohttp.ClientSession, pod_name="adservice", namespace="default"):
    """
    Fetch live metrics asynchronously
    """
    queries = {
        "hour_of_day": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
        "cpu_usage_lag_1": f'container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}}',
        "cpu_usage_lag_5": f'container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}} offset 5m',
        "cpu_roll_mean_10": f'avg_over_time(container_cpu_usage_seconds_total{{namespace="{namespace}", container_name="{pod_name}"}}[10m])',
        "mem_usage": f'container_memory_usage_bytes{{namespace="{namespace}", container_name="{pod_name}"}}',
        "req_rate": f'rate(http_requests_total{{namespace="{namespace}", pod=~"{pod_name}.*"}}[1m])'
    }

    results = await asyncio.gather(*[async_query_vm(session, q) for q in queries.values()])
    return dict(zip(queries.keys(), results))