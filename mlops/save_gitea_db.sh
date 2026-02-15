#!/bin/bash
# save_gitea_db.sh: Backup Gitea database including OAuth applications
# Similar to save_lakefs_db.sh for lakeFS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_PATH="$SCRIPT_DIR/gitea-prepared-db"

echo "=========================================="
echo "Backing up Gitea database with OAuth state"
echo "=========================================="
echo ""

# Scale down Gitea to ensure clean backup
echo "1. Scaling down Gitea deployment..."
kubectl scale deployment gitea -n gitea --replicas=0

echo "2. Waiting for Gitea pods to terminate..."
kubectl wait --for=delete pod -l app=gitea -n gitea --timeout=60s || echo "Pods may already be deleted"

echo "3. Creating backup directory..."
rm -rf "$BACKUP_PATH"
mkdir -p "$BACKUP_PATH"

echo "4. Creating inline backup helper pod..."
cat << 'PODEOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gitea-inline-backup
  namespace: gitea
spec:
  securityContext:
    runAsUser: 0
    runAsGroup: 0
    fsGroup: 0
  containers:
    - name: backup
      image: busybox:latest
      command: ["sh", "-c", "sleep 3600"]
      securityContext:
        runAsUser: 0
        runAsGroup: 0
      volumeMounts:
        - name: gitea-data
          mountPath: /data/gitea
  volumes:
    - name: gitea-data
      persistentVolumeClaim:
        claimName: gitea-data-pvc
PODEOF

echo "5. Waiting for backup helper to be ready..."
kubectl wait --for=condition=ready pod gitea-inline-backup -n gitea --timeout=60s || {
    echo "Error: Backup helper pod failed to start"
    exit 1
}

echo "6. Creating database backup..."
kubectl exec gitea-inline-backup -n gitea -- sh -c '
    echo "Backing up Gitea database..."
    mkdir -p /tmp/backup
    cp -r /data/gitea/* /tmp/backup/ 2>/dev/null || true
    ls -la /tmp/backup/
    echo "Compressing backup..."
    cd /tmp
    tar czf backup.tar.gz backup/
    echo "Backup size:"
    du -sh backup.tar.gz
'

echo "7. Copying backup to host..."
kubectl cp "gitea/gitea-inline-backup:/tmp/backup.tar.gz" "$BACKUP_PATH/db.tar.gz"

echo "8. Cleaning up backup helper..."
kubectl delete pod gitea-inline-backup -n gitea --ignore-not-found=true

echo "9. Restoring Gitea deployment..."
kubectl scale deployment gitea -n gitea --replicas=1

echo ""
echo "=========================================="
echo "✅ Backup Complete!"
echo "=========================================="
echo "Backup saved to: $BACKUP_PATH/"
echo "Contents:"
ls -la "$BACKUP_PATH/"
echo ""
echo "This backup includes:"
echo "✅ Gitea database with all users, repos, and OAuth apps"
echo "✅ OAuth credentials for Drone CI"
echo ""
echo "To restore on next install:"
echo "1. Run ./install_gitea.sh"
echo "2. It will automatically detect and restore the backup"
echo ""
echo "To start fresh (delete backup):"
echo "rm -rf $BACKUP_PATH/"
echo "=========================================="