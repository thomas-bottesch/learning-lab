#!/bin/bash
# save_lakefs_db.sh: Copy lakeFS database from PVC to local storage

set -e

NAMESPACE="lakefs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_LOCAL_PATH="$SCRIPT_DIR/lakefs-prepared-db"
TMP_DIR=$(mktemp -d)

echo "=========================================="
echo "Saving lakeFS database..."
echo "=========================================="

# Get the lakeFS pod name
POD_NAME=$(kubectl get pod -n $NAMESPACE -l app=lakefs -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [ -z "$POD_NAME" ]; then
    echo "Error: No lakeFS pod found in namespace $NAMESPACE"
    echo "Make sure lakeFS is running before saving the database."
    rm -rf "$TMP_DIR"
    exit 1
fi

echo "Found lakeFS pod: $POD_NAME"
echo "Using temp directory: $TMP_DIR"

# Copy the database files from the pod to temp
kubectl cp "$NAMESPACE/$POD_NAME:/data/lakefs/metadata" "$TMP_DIR/metadata" || {
    echo "Error: Failed to copy database from pod"
    rm -rf "$TMP_DIR"
    exit 1
}

# Also save the credentials info for reference
kubectl exec "$POD_NAME" -n $NAMESPACE -- wget -qO- http://localhost:8000/api/v1/setup_lakefs 2>/dev/null > "$TMP_DIR/setup_status.json" || true

# Create local backup directory
mkdir -p "$DB_LOCAL_PATH"

# Compress to tar.gz
echo "Compressing database..."
tar czf "$DB_LOCAL_PATH/db.tar.gz" -C "$TMP_DIR" .

# Cleanup temp
rm -rf "$TMP_DIR"

# Show size
SIZE=$(du -h "$DB_LOCAL_PATH/db.tar.gz" | cut -f1)

echo ""
echo "=========================================="
echo "Database saved successfully!"
echo "=========================================="
echo "Location: $DB_LOCAL_PATH/db.tar.gz"
echo "Size: $SIZE"
echo ""
echo "To restore on next install, run:"
echo "  bash install_lakefs.sh"
echo "=========================================="
