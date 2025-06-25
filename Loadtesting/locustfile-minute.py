import os
import csv
import random
import time
from locust import HttpUser, task, between, TaskSet, LoadTestShape


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


          
# LoadTestShape using CSV profile with loop over minutes

class CSVLoadShape(LoadTestShape):
    def __init__(self):
        super().__init__()
        self.load_profile = load_csv_profile("usask.sec.min.csv")
        self.minute = 0

    def tick(self):
        rpm = self.load_profile[self.minute]
        self.minute = self.minute + 1
        rps = rpm
        users = max(1, int(rps))
        spawn_rate = users
        print(f"Minute: {self.minute}, Target users={users}, RPM={rpm}, RPS={rps}")
        return users, spawn_rate

def load_csv_profile(csv_file):
    profile = []
    previous_total = None

    print(f"Attempting to load CSV profile from: {csv_file}")
    try:
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            row_count = 0
            for row in reader:
                row_count += 1
                clean_row = {k.strip(): v.strip() for k, v in row.items()}
                try:
                    current_total = float(clean_row["TotalNoOfComingRequests"])
                    delta = 0 if previous_total is None else current_total - previous_total
                    previous_total = current_total
                    profile.append(max(0, int(delta)))
                except (ValueError, KeyError) as e: # Catch specific exception
                    print(f"Invalid row skipped: {clean_row} - Error: {e}")
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
        exit(1)

    if not profile:
        print("No valid data found in CSV.")
        print(f"Debug: Profile is empty after processing {row_count} rows.")
        exit(1)

    print(f"Successfully loaded CSV profile with {len(profile)} entries.")
    return profile

if __name__ == "__main__":
    # os.system("locust -f locustfile.py")
    print('Started')
