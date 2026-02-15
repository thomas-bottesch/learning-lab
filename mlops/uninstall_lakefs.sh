#!/bin/bash
# uninstall_lakefs.sh: Completely remove lakeFS from the Kubernetes cluster

set -e

NAMESPACE="lakefs"

echo "=========================================="
echo "Uninstalling lakeFS..."
echo "=========================================="

# Stop port-forward if running
if screen -list | grep -q "lakefs-port"; then
    echo "Stopping lakeFS port-forward..."
    screen -S lakefs-port -X quit 2>/dev/null || true
fi

# Delete the setup job if it exists (ignore errors if not found)
echo "Removing setup job..."
kubectl delete job lakefs-setup -n $NAMESPACE --ignore-not-found=true

# Delete all lakeFS YAMLs in reverse order
echo "Removing lakeFS Kubernetes resources..."
YAML_DIR="k8s_yamls/lakefs"

# Find and delete all YAML files in reverse order (highest number first)
for yaml in $(ls -r "$YAML_DIR"/*.yaml 2>/dev/null || true); do
    if [ -f "$yaml" ]; then
        echo "Deleting $yaml..."
        kubectl delete -f "$yaml" --ignore-not-found=true
    fi
done

# Explicitly delete the PVC (not deleted by the YAMLs if they don't include it)
echo "Removing lakeFS PVC..."
kubectl delete pvc lakefs-metadata-pvc -n $NAMESPACE --ignore-not-found=true

# Delete the namespace (this removes any remaining resources)
echo "Removing lakeFS namespace..."
kubectl delete namespace $NAMESPACE --ignore-not-found=true

echo ""
echo "=========================================="
echo "lakeFS has been completely uninstalled!"
echo "=========================================="
echo ""
echo "Note: If you had MinIO data that lakeFS was using, you may want to"
echo "clean that up separately by logging into MinIO and removing the lakeFS buckets."
echo ""
