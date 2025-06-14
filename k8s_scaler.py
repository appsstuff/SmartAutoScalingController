from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException
from kubernetes.client.rest import ApiException
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_k8s_client():
    """
    Initialize Kubernetes clients for deployment scaling
    Returns (apps_v1, core_v1) clients or None
    """
    try:
        config.load_incluster_config()
        logger.info("🧠 Loaded in-cluster config")
        return client.AppsV1Api(), client.CoreV1Api()
    except ConfigException as ce:
        logger.warning(f"⚠️ In-cluster config failed: {ce}")
        try:
            config.load_kube_config()
            logger.info("🧠 Loaded local kubeconfig")
            return client.AppsV1Api(), client.CoreV1Api()
        except Exception as e2:
            logger.error(f"🚫 Failed to load any Kubernetes config: {e2}")
            return None, None


class K8sScaler:
    def __init__(self, min_replicas=2, max_replicas=10, scaling_step=1):
        self.apps_v1, _ = initialize_k8s_client()
        self.min_replicas = int(os.getenv("MIN_REPLICAS", min_replicas))
        self.max_replicas = int(os.getenv("MAX_REPLICAS", max_replicas))
        self.scaling_step = int(os.getenv("SCALING_STEP", scaling_step))

    def scale(self, deployment_name, namespace, decision):
        """
        Main scaling method — uses decision to adjust replica count
        """
        if not self.apps_v1:
            logger.error("🚫 Kubernetes client not initialized — skipping scaling")
            return False

        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
            current_replicas = deployment.spec.replicas
            logger.info(f"[{deployment_name}] Current replicas: {current_replicas}")

            # Calculate new replica count
            if decision == "scale_up":
                new_replicas = min(current_replicas + self.scaling_step, self.max_replicas)
            elif decision == "scale_down":
                new_replicas = max(current_replicas - self.scaling_step, self.min_replicas)
            else:
                new_replicas = current_replicas

            # Only patch if change is needed
            if new_replicas != current_replicas:
                return self._patch_replicas(deployment_name, namespace, new_replicas)
            else:
                logger.info(f"[{deployment_name}] No change in replicas ({current_replicas})")
                return True

        except ApiException as e:
            logger.error(f"[{deployment_name}] Kubernetes API exception: {e.reason}")
            return False
        except Exception as e:
            logger.error(f"[{deployment_name}] Unexpected error during scaling: {e}")
            return False


    def _patch_replicas(self, name, namespace, replicas):
        """
        Uses scale_subresource to update replica count
        """
        try:
            scale = client.V1Scale(
                metadata=client.V1ObjectMeta(name=name),
                spec=client.V1ScaleSpec(replicas=replicas),
                status=client.V1ScaleStatus(replicas=replicas)
            )

            self.apps_v1.patch_namespaced_deployment_scale(
                name=name,
                namespace=namespace,
                body=scale
            )
            print(f"✅ [K8s] Replicas updated to {replicas} for {name}")
            return True
        except Exception as e:
            print(f"🪲 Failed to scale deployment via scale_subresource: {e}")
            try:
                # Fallback to standard patch if scale_subresource fails
                body = {"spec": {"replicas": replicas}}
                self.apps_v1.patch_namespaced_deployment(
                    name=name,
                    namespace=namespace,
                    body=body
                )
                print(f"✅ Replicas updated via direct patch for {name}")
                return True
            except Exception as fallback_e:
                print(f"🚫 Fallback patch also failed: {fallback_e}")
                return False