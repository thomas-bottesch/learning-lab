#!/bin/bash
# uninstall_minio.sh: Remove all MinIO-related Kubernetes resources

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="minio"
YAML_DIR="$SCRIPT_DIR/k8s_yamls/minio"

echo "=========================================="
echo "Uninstalling MinIO..."
echo "=========================================="

# Delete resources in reverse order of creation (job -> deployment -> pvc -> namespace)
echo "Deleting MinIO resources..."

# Delete job first
kubectl delete -f "$YAML_DIR/06-create-bucket-job.yaml" --ignore-not-found=true

# Delete deployment to stop using PVC
kubectl delete -f "$YAML_DIR/04-deployment.yaml" --ignore-not-found=true

# Wait for pod to terminate
echo "Waiting for MinIO pod to terminate..."
kubectl wait --for=delete pod -l app=minio -n "$NAMESPACE" --timeout=60s 2>/dev/null || true

# Delete PVC to remove registry data (this is key for fresh reinstall)
echo "Deleting MinIO data (PVC)..."
kubectl delete -f "$YAML_DIR/03-pvc.yaml" --ignore-not-found=true

# Delete remaining resources
kubectl delete -f "$YAML_DIR/01-namespace.yaml" --ignore-not-found=true

echo ""
echo "=========================================="
echo "MinIO has been uninstalled successfully!"
echo "=========================================="
echo ""
echo "All data has been removed, including:"
echo "  - MinIO deployment and pods"
echo "  - Persistent Volume Claim (storage data)"
echo "  - Services, secret, and namespace"
echo ""
echo "You can now reinstall MinIO with:"
echo "  ./install_minio.sh"
echo "=========================================="
