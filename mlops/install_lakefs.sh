#!/bin/bash
# install_lakefs.sh: Apply all lakeFS-related Kubernetes YAMLs

set -e

YAML_DIR="k8s_yamls/lakefs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_BACKUP_PATH="$SCRIPT_DIR/lakefs-prepared-db"

# Check for saved database backup first
if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
    echo "Found saved database backup at $DB_BACKUP_PATH/db.tar.gz"
    echo "Preparing PVC and restoring database before deploying lakeFS..."
    
    # Apply only PVC first
    echo "Applying PVC..."
    kubectl apply -f "$YAML_DIR/01-namespace.yaml"
    kubectl apply -f "$YAML_DIR/03-pvc.yaml"
    
    # Create inline restore helper pod
    echo "Starting inline restore helper..."
    cat << 'PODEOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: lakefs-inline-restore
  namespace: lakefs
spec:
  securityContext:
    runAsUser: 0
    runAsGroup: 0
    fsGroup: 0
  containers:
    - name: restore
      image: busybox:latest
      command: ["sh", "-c", "sleep 3600"]
      securityContext:
        runAsUser: 0
        runAsGroup: 0
      volumeMounts:
        - name: metadata
          mountPath: /data/lakefs
  volumes:
    - name: metadata
      persistentVolumeClaim:
        claimName: lakefs-metadata-pvc
PODEOF
    
    # Wait for PVC to be bound
    echo "Waiting for PVC to be bound..."
    kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/lakefs-metadata-pvc -n lakefs --timeout=2m || {
        echo "PVC not bound yet, continuing anyway..."
    }

    # Wait for helper pod to be ready
    echo "Waiting for restore helper to be ready..."
    kubectl wait --for=condition=ready pod lakefs-inline-restore -n lakefs --timeout=60s || {
        echo "Error: Restore helper pod failed to start"
        exit 1
    }
    
    # Copy and extract database
    echo "Restoring database to PVC..."
    kubectl cp "$DB_BACKUP_PATH/db.tar.gz" "lakefs/lakefs-inline-restore:/tmp/db.tar.gz"
    kubectl exec lakefs-inline-restore -n lakefs -- sh -c '
        mkdir -p /tmp/extract
        tar xzf /tmp/db.tar.gz -C /tmp/extract
        mkdir -p /data/lakefs/metadata
        cp -r /tmp/extract/metadata/* /data/lakefs/metadata/
        chmod -R 777 /data/lakefs/metadata
        echo "Database restored to PVC"
        ls -la /data/lakefs/metadata/
    '
    # Copy setup_status.json if it exists
    if [ -f "$DB_BACKUP_PATH/setup_status.json" ]; then
        echo "Copying setup_status.json to PVC..."
        kubectl cp "$DB_BACKUP_PATH/setup_status.json" "lakefs/lakefs-inline-restore:/data/lakefs/setup_status.json"
        kubectl exec lakefs-inline-restore -n lakefs -- sh -c 'chmod 777 /data/lakefs/setup_status.json'
    fi

    # Clean up restore pod
    echo "Cleaning up restore helper..."
    kubectl delete pod lakefs-inline-restore -n lakefs --ignore-not-found=true
    
    echo "Database restored. Now deploying lakeFS..."
    
    # Apply remaining YAMLs (skip PVC and restore helper)
    for yaml in "$YAML_DIR"/*.yaml; do
        echo "Applying $yaml..."
        kubectl apply -f "$yaml"
    done
else
    # No backup - apply all YAMLs normally
    for yaml in "$YAML_DIR"/*.yaml; do
        echo "Applying $yaml..."
        kubectl apply -f "$yaml"
    done
fi

echo "All lakeFS YAMLs applied successfully."

# Wait for deployment
echo "Waiting for lakeFS deployment to be ready..."
kubectl rollout status deployment/lakefs -n lakefs --timeout=5m

# Run setup job only if no backup
if [ ! -f "$DB_BACKUP_PATH/db.tar.gz" ]; then   
    # Wait for setup job to complete
    echo "Waiting for setup job to complete..."
    kubectl wait --for=condition=complete job/lakefs-setup -n lakefs --timeout=5m || true
    
    # Show setup logs
    echo ""
    echo "Setup job logs:"
    kubectl logs job/lakefs-setup -n lakefs --tail=20 || true
fi

# Port-forward lakeFS UI to host
echo "Starting port-forward for lakeFS UI (port 8000)..."
screen -dmS lakefs-port kubectl -n lakefs port-forward svc/lakefs 8000:8000

echo ""
echo "=========================================="
echo "lakeFS is now available!"
echo "=========================================="
echo "UI: http://localhost:8000"
echo ""

# Display credentials based on install type
if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
    echo "Using RESTORED database with saved credentials."
    echo ""
    echo "To save this database for future reinstalls:"
    echo "  ./save_lakefs_db.sh"
else
    echo "Using FRESH install with NEW credentials."
    echo ""
    echo "Credentials (check setup job logs above for generated credentials):"
    echo "  These will be different each time you reinstall."
    echo ""
    echo "To save the database (preserves credentials for next time):"
    echo "  ./save_lakefs_db.sh"
fi

echo ""
echo "To configure lakeFS with your existing MinIO storage:"
echo "  1. Create a repository in the lakeFS UI"
echo "  2. Set storage namespace: s3://your-bucket/lakefs-data"
echo "  3. Start versioning your data!"
echo "=========================================="
