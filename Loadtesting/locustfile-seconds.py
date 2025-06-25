import os
import random
from locust import HttpUser, task, between, TaskSet, LoadTestShape

# ----------------------------------------
# Define service weights
# ----------------------------------------

SERVICES = [
    {"name": "adservice", "path": "/api/ad", "weight": 1},
    {"name": "cartservice", "path": "/api/cart", "weight": 2},
    {"name": "checkoutservice", "path": "/api/checkout", "weight": 3},
    {"name": "currencyservice", "path": "/api/currency", "weight": 1},
    {"name": "emailservice", "path": "/api/email", "weight": 1},
    {"name": "frontend", "path": "/", "weight": 5},
    {"name": "paymentservice", "path": "/api/payment", "weight": 1},
    {"name": "productcatalogservice", "path": "/api/products", "weight": 2},
    {"name": "recommendationservice", "path": "/api/recommend", "weight": 2},
    {"name": "redis-cart", "path": "/api/redis", "weight": 1},
    {"name": "shippingservice", "path": "/api/shipping", "weight": 2},
]


def weighted_random_service():
    total_weight = sum(service["weight"] for service in SERVICES)
    rand = random.uniform(0, total_weight)
    cumulative_weight = 0
    for service in SERVICES:
        cumulative_weight += service["weight"]
        if rand < cumulative_weight:
            return service


class UserBehavior(TaskSet):
    @task
    def perform_request(self):
        service = weighted_random_service()
        try:
            self.client.get(service["path"], name=service["name"])
        except Exception as e:
            print(f"Request failed: {service['name']} - {e}")


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    host = os.getenv("TARGET_HOST", "http://35.190.133.77")
    wait_time = between(0, 0)  # no wait time for better RPS emulation


if __name__ == "__main__":
    # os.system("locust -f locustfile.py")
    print('Started')
