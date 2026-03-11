#!/bin/bash
# uninstall_zot.sh: Remove all Zot-related Kubernetes resources

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="zot"
YAML_DIR="$SCRIPT_DIR/k8s_yamls/zot"

echo "=========================================="
echo "Uninstalling Zot..."
echo "=========================================="

# Delete resources in reverse order of creation (service -> deployment -> pvc -> configmap -> namespace)
echo "Deleting Zot resources..."

# Delete deployment first to stop using PVC
kubectl delete -f "$YAML_DIR/04-deployment.yaml" --ignore-not-found=true

# Wait for pod to terminate
echo "Waiting for Zot pod to terminate..."
kubectl wait --for=delete pod -l app=zot -n "$NAMESPACE" --timeout=60s 2>/dev/null || true

# Delete PVC to remove registry data (this is key for fresh reinstall)
echo "Deleting Zot data (PVC)..."
kubectl delete -f "$YAML_DIR/03-pvc.yaml" --ignore-not-found=true

# Delete remaining resources
kubectl delete -f "$YAML_DIR/01-namespace.yaml" --ignore-not-found=true

echo ""
echo "=========================================="
echo "Zot has been uninstalled successfully!"
echo "=========================================="
echo ""
echo "All data has been removed, including:"
echo "  - Zot deployment and pods"
echo "  - Persistent Volume Claim (registry data)"
echo "  - Services, configmap, and namespace"
echo ""
echo "You can now reinstall Zot with:"
echo "  ./install_zot.sh"
echo "=========================================="