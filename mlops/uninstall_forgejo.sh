#!/bin/bash
# uninstall_forgejo.sh: Remove all Forgejo-related Kubernetes resources

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="forgejo"
YAML_DIR="$SCRIPT_DIR/k8s_yamls/forgejo"

echo "=========================================="
echo "Uninstalling Forgejo..."
echo "=========================================="

# Delete resources in reverse order of creation (service -> deployment -> pvc -> secret -> namespace)
echo "Deleting Forgejo resources..."
kubectl delete -f "$YAML_DIR/01-namespace.yaml" --ignore-not-found=true


# Delete runner first
echo "Deleting Forgejo Runner..."
kubectl delete -f "$YAML_DIR/06-runner-deployment.yaml" --ignore-not-found=true

# Wait for runner pod to terminate
echo "Waiting for Forgejo Runner pod to terminate..."
kubectl wait --for=delete pod -l app=forgejo-runner -n "$NAMESPACE" --timeout=60s 2>/dev/null || true

# Delete deployment first to stop using PVC
kubectl delete -f "$YAML_DIR/04-deployment.yaml" --ignore-not-found=true

# Wait for pod to terminate
echo "Waiting for Forgejo pod to terminate..."
kubectl wait --for=delete pod -l app=forgejo -n "$NAMESPACE" --timeout=60s 2>/dev/null || true

# Delete PVC to remove database (this is key for fresh reinstall)
echo "Deleting Forgejo data (PVC)..."
kubectl delete -f "$YAML_DIR/03-pvc.yaml" --ignore-not-found=true

# Delete remaining resources
kubectl delete -f "$YAML_DIR/05-service.yaml" --ignore-not-found=true
kubectl delete -f "$YAML_DIR/02-secret.yaml" --ignore-not-found=true


echo ""
echo "=========================================="
echo "Forgejo has been uninstalled successfully!"
echo "=========================================="
echo ""
echo "All data has been removed, including:"
echo "  - Forgejo deployment and pods"
echo "  - Persistent Volume Claim (database)"
echo "  - Services, secrets, and namespace"
echo ""
echo "You can now reinstall Forgejo with:"
echo "  ./install_forgejo.sh"
echo ""
echo "The admin user will be created automatically on fresh install."
echo "=========================================="
