from locust import HttpUser, task, between, TaskSet
import random
import os

# add weight for each service
# number of Pods when gather data
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

class UserBehavior(TaskSet):
    def request_service(self, service):
        try:
            self.client.get(service["path"], name=service["name"])
        except Exception as e:
            print(f"Request failed: {service['name']} - {e}")

    @task(1)
    def normal_time(self):
        service = random.choice(SERVICES)
        self.request_service(service)
    
    @task(2)
    def peak_time(self):
        service = random.choice(SERVICES)
        for _ in range(2):
            self.request_service(service)

    @task(1)
    def workload(self):
        service = random.choice(SERVICES)
        for _ in range(5):
            self.request_service(service)

    @task(1)
    def heavy_load(self):
        service = random.choice(SERVICES)
        for _ in range(10):
            self.request_service(service)

    @task(1)
    def abnormal_time(self):
        service = random.choice(SERVICES)
        try:
            self.client.get(service["path"], name=service["name"], headers={"X-Abnormal-Load": "true"})
        except Exception as e:
            print(f"Abnormal load request failed: {service['name']} - {e}")

class WebsiteUser(HttpUser):
    host = os.getenv("TARGET_HOST", "http://35.190.133.77")
    tasks = [UserBehavior]
    wait_time = between(1, 3)

if __name__ == "__main__":
    os.system("locust -f locustfile.py")
