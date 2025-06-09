# import asyncio
# import aiohttp
# import numpy as np
# from async_utils import fetch_pod_metrics_async
# from pod_config import AUTO_SCALE_SERVICES


# async def run_all_services():
#     async with aiohttp.ClientSession() as session:
#         tasks = [fetch_pod_metrics_async(session, svc["pod_name"], svc["namespace"]) for svc in AUTO_SCALE_SERVICES]
#         return await asyncio.gather(*tasks)

# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     raw_metrics_list = loop.run_until_complete(run_all_services())

#     # Print results
#     for idx, metrics in enumerate(raw_metrics_list):
#         print(f"\n Metrics for service {idx+1}:")
#         for key, val in metrics.items():
#             print(f"  {key}: {val}")