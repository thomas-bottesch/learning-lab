#!/bin/bash
# uninstall_mlflow.sh: Remove MLflow from Kubernetes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

YAML_DIR="$SCRIPT_DIR/k8s_yamls/mlflow"

echo "=========================================="
echo "Uninstalling MLflow"
echo "=========================================="

# Kill port-forward if running
echo "1. Stopping MLflow port-forward..."
screen -S mlflow-port -X quit 2>/dev/null || true

# Delete all MLflow resources
echo "2. Deleting MLflow Kubernetes resources..."
for yaml in "$YAML_DIR"/*.yaml; do
    echo "  Deleting $yaml..."
    kubectl delete -f "$yaml" --ignore-not-found || true
done

# Delete namespace (this will delete everything in the namespace)
echo "3. Deleting MLflow namespace..."
kubectl delete namespace mlflow --ignore-not-found || true

echo ""
echo "=========================================="
echo "MLflow Uninstallation Complete!"
echo "=========================================="
echo "All MLflow resources have been removed."
echo ""
echo "Note: PersistentVolumeClaims will remain unless manually deleted."
echo "To delete PVCs: kubectl delete pvc -n mlflow --all"
echo "=========================================="