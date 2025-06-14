import argparse
import subprocess
import sys

# to test
# python k8s-load-tester.py --service all --mode peak  // all services
# python k8s-load-tester.py --service adservice         // one service
# python k8s-load-tester.py --service cartservice --mode custom --requests 2000 --concurrency 100

SUPPORTED_SERVICES = {
    "adservice": "/ads",
    "cartservice": "/cart",
    "productcatalogservice": "/products",
    "paymentservice": "/charge",
    "shippingservice": "/ship",
    "emailservice": "/send",
    "currencyservice": "/currency",
    "frontend": "/"
}

def run_hey(url, requests=1000, concurrency=50, qps=0, timeout=60):
    """
    Run hey load test with specified parameters.
    """
    cmd = ["hey", "-n", str(requests), "-c", str(concurrency), "-z", f"{timeout}s"]
    if qps > 0:
        cmd += ["-q", str(qps)]
    cmd.append(url)

    print(f"🚀 Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)  # Print the output of the hey command
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[⚠️] Error: hey exited with code {e.returncode}\n{e.stderr}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print("Error: 'hey' command not found. Please install 'hey' and make sure it is in your PATH.", file=sys.stderr)
        return -1

def get_service_url(service_name):
    if service_name not in SUPPORTED_SERVICES:
        raise ValueError(f"Unsupported service: {service_name}")
    return f"http://{service_name}.default.svc.cluster.local{SUPPORTED_SERVICES[service_name]}"

def main():
    parser = argparse.ArgumentParser(description="Test microservices with 'hey'")
    parser.add_argument("--service", required=True, choices=list(SUPPORTED_SERVICES.keys()) + ["all"],
                        help="Service to test (or use 'all' for full test)")
    parser.add_argument("--mode", default="normal",
                        choices=["normal", "peak", "abnormal", "custom"],
                        help="Load mode: normal, peak, abnormal, or custom")
    parser.add_argument("--requests", type=int, help="Custom request count")
    parser.add_argument("--concurrency", type=int, help="Custom concurrency level")
    parser.add_argument("--qps", type=int, default=0, help="Queries per second limit")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")

    args = parser.parse_args()

    if args.service == "all":
        services_to_test = list(SUPPORTED_SERVICES.keys())
    else:
        services_to_test = [args.service]

    # Predefined workload scenarios
    workload_profiles = {
        "normal": {"requests": 1000, "concurrency": 50, "timeout": 60},
        "peak": {"requests": 5000, "concurrency": 200, "timeout": 120},
        "abnormal": {"requests": 8000, "concurrency": 300, "timeout": 180},
        "custom": {}
    }

    profile = workload_profiles.get(args.mode, {})
    print(f"🧪 Using '{args.mode}' profile")

    for svc in services_to_test:
        url = get_service_url(svc)
        print(f"\n🔄 Testing {svc} at {url}")

        req = args.requests or profile.get("requests", 1000)
        con = args.concurrency or profile.get("concurrency", 50)
        to = args.timeout or profile.get("timeout", 60)

        print(f"📊 Requests: {req}, Concurrency: {con}, Timeout: {to}s")

        exit_code = run_hey(url, req, con, args.qps, to)
        if exit_code != 0:
            print(f"[⚠️] Load test failed for {svc} — code {exit_code}")
        else:
            print(f"[✅] Load test completed for {svc}")

if __name__ == '__main__':
    main()
