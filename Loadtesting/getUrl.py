import requests

SERVICES = [
    {"name": "adservice", "path":  "/adservice"},  # Updated path
    {"name": "cartservice", "path":  "/cartservice"},  # Updated path
    {"name": "checkoutservice", "path":  "/checkoutservice"},  # Updated path
    {"name": "currencyservice", "path":  "/currencyservice"},  # Updated path
    {"name": "emailservice", "path":  "/emailservice"},  # Updated path
    {"name": "frontend", "path": "/"},  # This is correct
    {"name": "paymentservice", "path":  "/paymentservice"},  # Updateds path
    {"name": "productcatalogservice", "path":  "/productcatalogservice"},  # Updated path
    {"name": "recommendationservice", "path":  "/recommendationservice"},  # Updated path
    {"name": "redis-cart", "path":  "/redis-cart"},  # Updated path
    {"name": "shippingservice", "path":  "/shippingservice"},  # Updated path
]


host = "http://35.190.133.77" # Replace with your target host

for service in SERVICES:
    url = f"{host}{service['path']}"
    try:
        response = requests.get(url)
        print(f"{service['name']} - Status Code: {response.status_code}")
    except Exception as e:
        print(f"{service['name']} - Error: {e}")
