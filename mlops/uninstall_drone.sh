#!/bin/bash
# uninstall_drone.sh: Delete all Drone CI Kubernetes resources

set -e

YAML_DIR="k8s_yamls/drone"

# Delete all YAMLs in reverse order
for yaml in "$YAML_DIR"/*.yaml; do
    echo "Deleting $yaml..."
    kubectl delete -f "$yaml" --ignore-not-found=true
    sleep 1
done

# Kill port-forward if running
screen -S drone-port -X quit 2>/dev/null || true

# Delete namespace (will delete any remaining resources)
echo "Deleting drone namespace..."
kubectl delete namespace drone --ignore-not-found=true

echo "Drone CI uninstalled successfully."