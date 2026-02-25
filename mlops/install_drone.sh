#!/bin/bash
# install_drone.sh: Apply all Drone CI-related Kubernetes YAMLs with backup/restore
# Similar to install_gitea.sh pattern

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_DIR="$SCRIPT_DIR/k8s_yamls/drone"
DB_BACKUP_PATH="$SCRIPT_DIR/drone-prepared-db"

# Variables
DRONE_URL="http://172.17.0.1:3001"

# Functions
restore_drone_backup() {
    if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
        echo "Found saved database backup at $DB_BACKUP_PATH/db.tar.gz"
        echo "Preparing PVC and restoring database before deploying Drone..."
        
        # Apply only namespace and PVC first
        echo "Applying namespace and PVC..."
        kubectl apply -f "$YAML_DIR/01-namespace.yaml"
        kubectl apply -f "$YAML_DIR/03-pvc.yaml"
        
        # Create inline restore helper pod
        echo "Starting inline restore helper..."
        cat << 'PODEOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: drone-inline-restore
  namespace: drone
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
        - name: drone-data
          mountPath: /data
  volumes:
    - name: drone-data
      persistentVolumeClaim:
        claimName: drone-data-pvc
PODEOF

        echo "Waiting for restore helper to be ready..."
        kubectl wait --for=condition=ready pod drone-inline-restore -n drone --timeout=60s || {
            echo "Error: Restore helper pod failed to start"
            exit 1
        }

        echo "Copying backup to pod..."
        kubectl cp "$DB_BACKUP_PATH/db.tar.gz" drone/drone-inline-restore:/tmp/restore.tar.gz

        echo "Restoring database..."
        kubectl exec drone-inline-restore -n drone -- sh -c '
            echo "Cleaning existing data..."
            rm -rf /data/*
            
            echo "Extracting backup..."
            cd /tmp
            tar xzf restore.tar.gz
            
            echo "Restoring files..."
            cp -r backup/* /data/ 2>/dev/null || true
            
            echo "Setting permissions..."
            chmod -R 755 /data/
            
            echo "Restored files:"
            ls -la /data/
        '

        echo "Cleaning up restore helper..."
        kubectl delete pod drone-inline-restore -n drone --ignore-not-found=true
        
        echo "✅ Database restored from backup"
        return 0
    else
        echo "No existing Drone backup found, fresh installation"
        return 1
    fi
}

apply_drone_yamls() {
    # Execute configmap and secret creation scripts
    echo "Creating Drone ConfigMap..."
    "$SCRIPT_DIR/create_drone_configmap.sh"
    
    echo "Creating Drone Secret..."
    "$SCRIPT_DIR/create_drone_additional_secret.sh"
    
    # Apply remaining Drone YAMLs (excluding namespace and PVC which may already be applied)
    for yaml in "$YAML_DIR"/*.yaml; do
        # Skip namespace and PVC files (already applied during backup restore)
        if [[ "$(basename "$yaml")" == "01-namespace.yaml" ]] || [[ "$(basename "$yaml")" == "03-pvc.yaml" ]]; then
            continue
        fi
        echo "Applying $yaml..."
        kubectl apply -f "$yaml"
    done
    echo "All Drone YAMLs applied successfully."
}

wait_for_drone() {
    echo "Waiting for Drone deployment to be ready..."
    kubectl rollout status deployment/drone -n drone --timeout=5m
}

wait_for_drone_web() {
    echo "Waiting for Drone web server to be available..."
    for i in {1..30}; do
        if curl -s --head --fail "$DRONE_URL" > /dev/null; then
            echo "Drone web server is up."
            return 0
        else
            echo "Drone web server not ready yet, retrying ($i)..."
            sleep 2
        fi
    done
    echo "Drone web server did not become available in time."
    exit 1
}

port_forward_drone() {
    echo "Starting port-forward for Drone UI (port 3001)..."
    screen -dmS drone-port kubectl -n drone port-forward --address 0.0.0.0 svc/drone 3001:80
}

show_drone_info() {
    echo ""
    echo "=========================================="
    echo "Drone CI Installation Complete!"
    echo "=========================================="
    echo "UI: $DRONE_URL"
    echo ""
    echo "Drone CI is now fully configured with:"
    echo "✅ OAuth authentication with Gitea"
    echo "✅ Docker runner for pipeline execution"
    echo "✅ Persistent storage"
    echo ""
    if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
        echo "✅ Database restored from backup"
        echo "   User registration and pipeline state preserved"
        echo ""
        echo "To access Drone CI:"
        echo "1. Go to: $DRONE_URL"
        echo "2. Click 'Login with Gitea'"
        echo "3. You should be logged in automatically"
    else
        echo "⚠️  Fresh installation detected"
        echo "   OAuth needs to be set up once"
        echo "   User registration required on first login"
        echo ""
        echo "To access Drone CI:"
        echo "1. Go to: $DRONE_URL"
        echo "2. Click 'Login with Gitea'"
        echo "3. Authorize the application"
        echo "4. Complete registration at: $DRONE_URL/register"
        echo ""
        echo "After setup, backup to preserve state:"
        echo "  ./save_drone_db.sh"
    fi
    echo ""
    echo "To create your first pipeline:"
    echo "1. Go to your Gitea repository"
    echo "2. Add a .drone.yml file"
    echo "3. Activate the repository in Drone UI"
    echo "4. Push changes to trigger pipeline"
    echo ""
    echo "Backup/restore commands:"
    echo "  Backup:    ./save_drone_db.sh"
    echo "  Status:    ls -la $DB_BACKUP_PATH/"
    echo "=========================================="
}

# Main installation logic
echo "=========================================="
echo "Installing Drone CI with backup/restore support"
echo "=========================================="

# Check for saved database backup first and restore if exists
restore_drone_backup

# Apply all YAMLs (if backup was restored, some may already be applied)
apply_drone_yamls
wait_for_drone
port_forward_drone
wait_for_drone_web
show_drone_info