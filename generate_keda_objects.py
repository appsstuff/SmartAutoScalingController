import os
from jinja2 import Template
from pod_config import AUTO_SCALE_SERVICES
# Generate KEDA yaml files


# Define template
TEMPLATE = """
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: smart-autoscaler-keda-{{ pod_name }}
  namespace: default
  Labels:
    keda-scaling="true"
    metrics-enabled="true"  
spec:
  scaleTargetRef:
    name: {{ pod_name }}
    kind: Deployment
  minReplicaCount: {{ min_replicas }}
  maxReplicaCount: {{ max_replicas }}
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://victoriametrics.monitoring.svc.cluster.local:8428
        metricName: autoscaler_decision
        threshold: "1.5"
        query: 'autoscaler_decision{service="{{ pod_name }}"}'
"""

t = Template(TEMPLATE)

for svc in AUTO_SCALE_SERVICES:
    rendered = t.render(**svc)
    filename = f"keda-autoscaler-{svc['pod_name']}.yaml"
    with open(filename, "w") as f:
        f.write(rendered)
    print(f"📄 Generated {filename}")