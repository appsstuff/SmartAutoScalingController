from locust import HttpUser, task, between, constant_pacing
import random
import time
import os

# Define your services and paths
SERVICES = [
    {"name": "adservice", "path": "/api/ad"},
    {"name": "cartservice", "path": "/api/cart"},
    {"name": "checkoutservice", "path": "/api/checkout"},
    {"name": "currencyservice", "path": "/api/currency"},
    {"name": "emailservice", "path": "/api/email"},
    {"name": "frontend", "path": "/"},
    {"name": "paymentservice", "path": "/api/payment"},
    {"name": "productcatalogservice", "path": "/api/products"},
    {"name": "recommendationservice", "path": "/api/recommend"},
    {"name": "redis-cart", "path": "/api/redis"},
    {"name": "shippingservice", "path": "/api/shipping"},
]

# Simulate time of day
def get_current_phase():
    hour = time.localtime().tm_hour
    if 9 <= hour <= 11 or 17 <= hour <= 20:
        return "peak"
    elif 0 <= hour <= 6:
        return "low"
    else:
        return "normal"

class ServiceUser(HttpUser):
    wait_time = between(0.5, 2)  # base delay between requests

    @task
    def call_services(self):
        phase = get_current_phase()
        service = random.choice(SERVICES)
        path = service["path"]

        # Phase-based behavior
        if phase == "peak":
            repeat = random.randint(5, 10)
        elif phase == "normal":
            repeat = random.randint(2, 4)
        else:
            repeat = 1

        for _ in range(repeat):
            with self.client.get(path, name=f"{service['name']}{path}", catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"{service['name']} failed with {response.status_code}")

        # Heavy load simulation (random)
        if random.random() < 0.1:  # 10% chance
            for _ in range(20):
                self.client.get(path, name=f"heavy_{service['name']}{path}")
