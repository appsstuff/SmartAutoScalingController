from kubernetes import client, config
from datetime import datetime
import os

def initialize_k8s_client():
    try:
        config.load_incluster_config()
        print("🧠 Loaded in-cluster config")
    except Exception as e:
        print(f"⚠️ Failed to load in-cluster config: {e}")
        config.load_kube_config()
        print("🧠 Loaded local kube config")
    return client.AppsV1Api(), client.CoreV1Api()


def apply_k8s_scaling(pod_name, namespace, decision):
    apps_v1, core_v1 = initialize_k8s_client()
    deployment_name = pod_name  # Assuming pod name matches deployment name

    try:
        deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        current_replicas = deployment.spec.replicas
        min_replicas = os.getenv("MIN_REPLICAS", 2)
        max_replicas = os.getenv("MAX_REPLICAS", 10)
        step = os.getenv("SCALING_STEP", 1)

        if decision == "scale_up":
            new_replicas = min(current_replicas + step, max_replicas)
        elif decision == "scale_down":
            new_replicas = max(current_replicas - step, min_replicas)
        else:
            new_replicas = current_replicas

        if new_replicas != current_replicas:
            body = {"spec": {"replicas": new_replicas}}
            apps_v1.patch_namespaced_deployment(deployment_name, namespace, body)
            print(f"[SUCCESS] Updated replicas to {new_replicas} for {pod_name} in {namespace}")
        else:
            print(f"[INFO] {pod_name} already has {current_replicas} replicas — no change")

    except client.rest.ApiException as e:
        print(f"[ERROR] Kubernetes API exception: {e.reason}")
    except Exception as e:
        print(f"[ERROR] Unexpected exception during scaling: {e}")