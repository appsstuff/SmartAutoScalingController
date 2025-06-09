import os
from kubernetes import client, config

# ===== K8s Client Setup =====
def initialize_k8s_client():
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
        print(" Loaded local kube config")

    return client.AppsV1Api(), client.CoreV1Api()

# ===== Scaler Class =====
class K8sScaler:
    def __init__(self, min_replicas=2, max_replicas=10, scaling_step=1):
        self.apps_v1, _ = initialize_k8s_client()
        self.min_replicas = int(os.getenv("MIN_REPLICAS", min_replicas))
        self.max_replicas = int(os.getenv("MAX_REPLICAS", max_replicas))
        self.scaling_step = int(os.getenv("SCALING_STEP", scaling_step))

    def scale(self, deployment_name, namespace, decision):
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
            current_replicas = deployment.spec.replicas

            print(f"[INFO] {deployment_name} current replicas: {current_replicas}")

            # Determine new replica count
            if decision == "scale_up":
                new_replicas = min(current_replicas + self.scaling_step, self.max_replicas)
            elif decision == "scale_down":
                new_replicas = max(current_replicas - self.scaling_step, self.min_replicas)
            else:
                new_replicas = current_replicas

            if new_replicas != current_replicas:
                self._patch_replicas(deployment_name, namespace, new_replicas)
            else:
                print(f"[INFO] No change: replicas remain at {current_replicas}")

        except client.rest.ApiException as e:
            print(f"[ERROR] Kubernetes API exception: {e.reason}")
        except Exception as e:
            print(f"[ERROR] Unexpected exception during scaling: {e}")

    def _patch_replicas(self, name, namespace, replicas):
        body = {"spec": {"replicas": replicas}}
        self.apps_v1.patch_namespaced_deployment_scale(name=name, namespace=namespace, body=body)
        print(f"[SUCCESS] Updated replicas to {replicas} for {name} in {namespace}")

# ===== Standalone Function (optional backward compatibility) =====
def apply_k8s_scaling(pod_name, namespace, decision):
    scaler = K8sScaler()
    scaler.scale(pod_name, namespace, decision)