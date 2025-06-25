#!/bin/bash

# Get all services in the default namespace
services=$(kubectl get services -n default -o jsonpath='{.items[*].metadata.name}')

# Loop through each service
for service in $services; do
    # Get the service port (assuming the first port is the one to test)
    port=$(kubectl get service "$service" -n default -o jsonpath='{.spec.ports[0].port}')

    # Construct the URL
    url="http://$service.default.svc.cluster.local:$port"

    # Run hey for the service
    echo "Testing $url"
    hey -n 1000 -c 100 "$url"
done

#chmod +x load_test.sh
#./load_test.sh