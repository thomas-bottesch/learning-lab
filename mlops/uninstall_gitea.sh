#!/bin/bash
# uninstall_gitea.sh: Remove all Gitea-related Kubernetes resources

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="gitea"
YAML_DIR="$SCRIPT_DIR/k8s_yamls/gitea"

echo "=========================================="
echo "Uninstalling Gitea..."
echo "=========================================="

# Stop port-forward if running
if screen -list | grep -q "gitea-port"; then
    echo "Stopping gitea-port screen session..."
    screen -S gitea-port -X quit 2>/dev/null || true
fi

# Delete resources in reverse order of creation (service -> deployment -> pvc -> secret -> namespace)
echo "Deleting Gitea resources..."

# Delete the deployment first to stop using the PVC
kubectl delete -f "$YAML_DIR/04-deployment.yaml" --ignore-not-found=true

# Wait for pod to terminate
echo "Waiting for Gitea pod to terminate..."
kubectl wait --for=delete pod -l app=gitea -n "$NAMESPACE" --timeout=60s 2>/dev/null || true

# Delete the PVC to remove the database (this is key for fresh reinstall)
echo "Deleting Gitea data (PVC)..."
kubectl delete -f "$YAML_DIR/03-pvc.yaml" --ignore-not-found=true

# Delete remaining resources
kubectl delete -f "$YAML_DIR/05-service.yaml" --ignore-not-found=true
kubectl delete -f "$YAML_DIR/02-secret.yaml" --ignore-not-found=true
kubectl delete -f "$YAML_DIR/01-namespace.yaml" --ignore-not-found=true

echo ""
echo "=========================================="
echo "Gitea has been uninstalled successfully!"
echo "=========================================="
echo ""
echo "All data has been removed, including:"
echo "  - Gitea deployment and pods"
echo "  - Persistent Volume Claim (database)"
echo "  - Services, secrets, and namespace"
echo ""
echo "You can now reinstall Gitea with:"
echo "  ./install_gitea.sh"
echo ""
echo "The admin user will be created automatically on fresh install."
echo "=========================================="
