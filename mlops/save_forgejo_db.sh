#!/bin/bash
# save_forgejo_db.sh: Backup forgejo database including OAuth applications
# Similar to save_lakefs_db.sh for lakeFS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_PATH="$SCRIPT_DIR/forgejo-prepared-db"

echo "=========================================="
echo "Backing up forgejo database with OAuth state"
echo "=========================================="
echo ""

# Scale down forgejo to ensure clean backup
echo "1. Scaling down forgejo deployment..."
kubectl scale deployment forgejo -n forgejo --replicas=0

echo "2. Waiting for forgejo pods to terminate..."
kubectl wait --for=delete pod -l app=forgejo -n forgejo --timeout=60s || echo "Pods may already be deleted"

echo "3. Creating backup directory..."
rm -rf "$BACKUP_PATH"
mkdir -p "$BACKUP_PATH"

echo "4. Creating inline backup helper pod..."
cat << 'PODEOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: forgejo-inline-backup
  namespace: forgejo
spec:
  securityContext:
    runAsUser: 0
    runAsGroup: 0
    fsGroup: 0
  containers:
    - name: backup
      image: busybox:latest
      command: ["sh", "-c", "trap 'exit 0' TERM; while true; do sleep 1; done"]
      securityContext:
        runAsUser: 0
        runAsGroup: 0
      volumeMounts:
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: forgejo-data-pvc
PODEOF

echo "5. Waiting for backup helper to be ready..."
kubectl wait --for=condition=ready pod forgejo-inline-backup -n forgejo --timeout=60s || {
    echo "Error: Backup helper pod failed to start"
    exit 1
}

echo "6. Creating database backup..."
kubectl exec forgejo-inline-backup -n forgejo -- sh -c '
    echo "Backing up forgejo database..."
    mkdir -p /tmp/backup
    cp -r /data/forgejo/* /tmp/backup/ 2>/dev/null || true
    ls -la /tmp/backup/
    echo "Compressing backup..."
    cd /tmp
    tar czf backup.tar.gz backup/
    echo "Backup size:"
    du -sh backup.tar.gz
'

echo "7. Copying backup to host..."
kubectl cp "forgejo/forgejo-inline-backup:/tmp/backup.tar.gz" "$BACKUP_PATH/db.tar.gz"

echo "8. Cleaning up backup helper..."
kubectl delete pod forgejo-inline-backup -n forgejo --ignore-not-found=true

echo "9. Restoring forgejo deployment..."
kubectl scale deployment forgejo -n forgejo --replicas=1

echo ""
echo "=========================================="
echo "✅ Backup Complete!"
echo "=========================================="
echo "Backup saved to: $BACKUP_PATH/"
echo "Contents:"
ls -la "$BACKUP_PATH/"
echo ""
echo "This backup includes:"
echo "✅ forgejo database with all users"
echo ""
echo "To restore on next install:"
echo "1. Run ./install_forgejo.sh"
echo "2. It will automatically detect and restore the backup"
echo ""
echo "To start fresh (delete backup):"
echo "rm -rf $BACKUP_PATH/"
echo "Before a backup is created the token needs to be taken from webui and updated in k8s_yamls/forgejo/06-runner-deployment.yaml"
echo "Taken can be found in forgejo admin settings -> actions -> runners -> registration token"
echo "=========================================="