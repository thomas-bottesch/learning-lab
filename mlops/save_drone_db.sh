#!/bin/bash
# save_drone_db.sh: Backup Drone CI database including user registration state
# Similar to save_gitea_db.sh for Gitea

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_PATH="$SCRIPT_DIR/drone-prepared-db"

echo "=========================================="
echo "Backing up Drone CI database"
echo "=========================================="
echo ""

# Check if Drone deployment exists
if ! kubectl get deployment/drone -n drone &>/dev/null; then
    echo "❌ Drone deployment not found"
    echo "Please install Drone first with: ./install_drone.sh"
    exit 1
fi

# Scale down Drone to ensure clean backup
echo "1. Scaling down Drone deployment..."
kubectl scale deployment drone -n drone --replicas=0

echo "2. Waiting for Drone pods to terminate..."
kubectl wait --for=delete pod -l app=drone -n drone --timeout=60s || echo "Pods may already be deleted"

echo "3. Creating backup directory..."
rm -rf "$BACKUP_PATH"
mkdir -p "$BACKUP_PATH"

echo "4. Creating inline backup helper pod..."
cat << 'PODEOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: drone-inline-backup
  namespace: drone
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
        - name: drone-data
          mountPath: /data
  volumes:
    - name: drone-data
      persistentVolumeClaim:
        claimName: drone-data-pvc
PODEOF

echo "5. Waiting for backup helper to be ready..."
kubectl wait --for=condition=ready pod drone-inline-backup -n drone --timeout=60s || {
    echo "Error: Backup helper pod failed to start"
    exit 1
}

echo "6. Creating database backup..."
kubectl exec drone-inline-backup -n drone -- sh -c '
    echo "Backing up Drone database..."
    mkdir -p /tmp/backup
    cp -r /data/* /tmp/backup/ 2>/dev/null || true
    echo "Database files:"
    ls -la /tmp/backup/
    echo "Compressing backup..."
    cd /tmp
    tar czf backup.tar.gz backup/
    echo "Backup size:"
    du -sh backup.tar.gz
'

echo "7. Copying backup to host..."
kubectl cp "drone/drone-inline-backup:/tmp/backup.tar.gz" "$BACKUP_PATH/db.tar.gz"

echo "8. Creating metadata file..."
cat > "$BACKUP_PATH/metadata.json" << EOF
{
  "backup_date": "$(date -Iseconds)",
  "drone_version": "2",
  "database_file": "database.sqlite",
  "backup_type": "full",
  "notes": "Drone CI database backup including user registrations and pipeline state"
}
EOF

echo "9. Cleaning up backup helper..."
kubectl delete pod drone-inline-backup -n drone --ignore-not-found=true

echo "10. Restoring Drone deployment..."
kubectl scale deployment drone -n drone --replicas=1

echo ""
echo "=========================================="
echo "✅ Backup Complete!"
echo "=========================================="
echo "Backup saved to: $BACKUP_PATH/"
echo "Contents:"
ls -la "$BACKUP_PATH/"
echo ""
echo "This backup includes:"
echo "✅ Drone SQLite database with all user registrations"
echo "✅ Repository activations and pipeline configurations"
echo "✅ All Drone CI state"
echo ""
echo "To restore on next install:"
echo "1. Run ./install_drone.sh"
echo "2. It will automatically detect and restore the backup"
echo ""
echo "To start fresh (delete backup):"
echo "rm -rf $BACKUP_PATH/"
echo "=========================================="