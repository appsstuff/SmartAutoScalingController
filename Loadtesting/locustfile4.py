from locust import HttpUser , task, between, TaskSet
import random
import os
import time

# Define service weights
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

class UserBehavior(TaskSet):
    
    def request_service(self, service):
        try:
            self.client.get(service["path"], name=service["name"])
        except Exception as e:
            print(f"Request failed: {service['name']} - {e}")

    def weighted_random_service(self):
        total_weight = sum(service["weight"] for service in SERVICES)
        rand = random.uniform(0, total_weight)
        cumulative_weight = 0
        for service in SERVICES:
            cumulative_weight += service["weight"]
            if rand < cumulative_weight:
                return service

    @task(1)
    def normal_time(self):
        service = self.weighted_random_service()
        self.request_service(service)

    @task(2)
    def peak_time(self):
        service = self.weighted_random_service()
        for _ in range(2):
            self.request_service(service)

    @task(1)
    def workload(self):
        service = self.weighted_random_service()
        for _ in range(5):
            self.request_service(service)

    @task(1)
    def heavy_load(self):
        service = self.weighted_random_service()
        for _ in range(10):
            self.request_service(service)

    @task(1)
    def abnormal_time(self):
        service = self.weighted_random_service()
        try:
            self.client.get(service["path"], name=service["name"], headers={"X-Abnormal-Load": "true"})
        except Exception as e:
            print(f"Abnormal load request failed: {service['name']} - {e}")

class WebsiteUser (HttpUser ):
    host = os.getenv("TARGET_HOST", "http://35.190.133.77")
    tasks = [UserBehavior]
    wait_time = between(1, 3)

if __name__ == "__main__":
    os.system("locust -f locustfile.py")
