#!/bin/bash
# install_gitea.sh: Apply all Gitea-related Kubernetes YAMLs with backup/restore
# Similar to install_lakefs.sh pattern

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_DIR="$SCRIPT_DIR/k8s_yamls/gitea"
DB_BACKUP_PATH="$SCRIPT_DIR/gitea-prepared-db"

# Variables
GITEA_URL="http://172.17.0.1:3000"
REPO_NAME="my_ml_project"
EXAMPLE_PROJECT_DIR="./example_git_project"

# Extract Gitea credentials from YAML
GITEA_USER=$(grep 'admin-username:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
GITEA_PASS=$(grep 'admin-password:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
GITEA_EMAIL=$(grep 'admin-email:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')

# Functions
apply_gitea_yamls() {
    for yaml in "$YAML_DIR"/*.yaml; do
        echo "Applying $yaml..."
        kubectl apply -f "$yaml"
    done
    echo "All Gitea YAMLs applied successfully."
}

wait_for_gitea() {
    echo "Waiting for Gitea deployment to be ready..."
    kubectl rollout status deployment/gitea -n gitea --timeout=5m
}

create_admin_user() {
    echo "Creating admin user..."
    kubectl exec -n gitea deployment/gitea -- /usr/local/bin/gitea admin user create \
        --username "$GITEA_USER" \
        --password "$GITEA_PASS" \
        --email "$GITEA_EMAIL" \
        --admin \
        --must-change-password=false 2>/dev/null || echo "Admin user may already exist, continuing..."
}

port_forward_gitea() {
    echo "Starting port-forward for Gitea UI (port 3000)..."
    screen -dmS gitea-port kubectl -n gitea port-forward --address 0.0.0.0 svc/gitea 3000:3000
}

wait_for_gitea_web() {
    echo "Waiting for Gitea web server to be available..."
    for i in {1..30}; do
        if curl -s --head --fail "$GITEA_URL" > /dev/null; then
            echo "Gitea web server is up."
            return 0
        else
            echo "Gitea web server not ready yet, retrying ($i)..."
            sleep 2
        fi
    done
    echo "Gitea web server did not become available in time."
    exit 1
}

show_gitea_info() {
    echo ""
    echo "=========================================="
    echo "Gitea is now available!"
    echo "=========================================="
    echo "UI: $GITEA_URL"
    echo "SSH: localhost:30022"
    echo ""
    echo "Default admin credentials:"
    echo "  Username: $GITEA_USER"
    echo "  Password: $GITEA_PASS"
    echo "  Email: $GITEA_EMAIL"
    echo ""
    echo "Note: For production use, please change the default credentials"
    echo "      and secrets in $YAML_DIR/02-secret.yaml"
    echo "=========================================="
}

# Main installation logic
echo "=========================================="
echo "Installing Gitea with backup/restore support"
echo "=========================================="

# Check for saved database backup first
if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
    echo "Found saved database backup at $DB_BACKUP_PATH/db.tar.gz"
    echo "Preparing PVC and restoring database before deploying Gitea..."
    
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
  name: gitea-inline-restore
  namespace: gitea
spec:
  securityContext:
    runAsUser: 0
    runAsGroup: 0
    fsGroup: 0
  containers:
    - name: restore
      image: busybox:latest
      command: ["sh", "-c", "trap 'exit 0' TERM; while true; do sleep 1; done"]
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
    
    # Wait for PVC to be bound
    echo "Waiting for PVC to be bound..."
    kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/gitea-data-pvc -n gitea --timeout=2m || {
        echo "PVC not bound yet, continuing anyway..."
    }

    # Wait for helper pod to be ready
    echo "Waiting for restore helper to be ready..."
    kubectl wait --for=condition=ready pod gitea-inline-restore -n gitea --timeout=60s || {
        echo "Error: Restore helper pod failed to start"
        exit 1
    }
    
    # Copy and extract database
    echo "Restoring database to PVC..."
    kubectl cp "$DB_BACKUP_PATH/db.tar.gz" "gitea/gitea-inline-restore:/tmp/db.tar.gz"
    kubectl exec gitea-inline-restore -n gitea -- sh -c '
        mkdir -p /tmp/extract
        tar xzf /tmp/db.tar.gz -C /tmp/extract
        mkdir -p /data/gitea
        cp -r /tmp/extract/backup/* /data/gitea/ 2>/dev/null || true
        chmod -R 777 /data/gitea
        echo "Database restored to PVC"
        ls -la /data/gitea/
    '

    # Clean up restore pod
    echo "Cleaning up restore helper..."
    kubectl delete pod gitea-inline-restore -n gitea --ignore-not-found=true --timeout=3s || true
    
    echo "Database restored. Now deploying Gitea..."
    
    # Apply remaining YAMLs
    for yaml in "$YAML_DIR"/*.yaml; do
        echo "Applying $yaml..."
        kubectl apply -f "$yaml"
    done
else
    # No backup - apply all YAMLs normally
    echo "No database backup found. Performing fresh install..."
    for yaml in "$YAML_DIR"/*.yaml; do
        echo "Applying $yaml..."
        kubectl apply -f "$yaml"
    done
fi

echo "All Gitea YAMLs applied successfully."

# Wait for deployment
wait_for_gitea

# Port-forward Gitea UI to host
port_forward_gitea

# Wait for web server
wait_for_gitea_web

# Show info
show_gitea_info

# Check if we restored from backup
if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
    echo ""
    echo "✅ RESTORED from backup - skipping admin/user"
    echo "   (Admin user, and OAuth state already exist)"
else
    # Fresh install: create admin user and repo
    echo ""
    echo "✅ FRESH install - creating admin user and repository"
    
    # Create admin user (idempotent)
    create_admin_user

    echo "You must login to:"
    echo "http://localhost:3000/user/settings/applications"
    echo "and create an OAuth application for Drone with:"
    echo "  Name: Drone CI"
    echo "  Redirect URI: http://172.17.0.1:3001/login"

    echo "you also must add the generated Client ID and Secret from the webui to the Kubernetes secret:"
    echo "k8s_yamls/drone/02-secret.yaml"
    echo "drone-gitea-client-id: <client_id_from_gitea>"
    echo "drone-gitea-client-secret: <client_secret_from_gitea>"
    echo "to allow drone access to Gitea for repository management and OAuth authentication"
    echo "then you must install drone and login once to gitea to complete the OAuth flow and save the credentials for future reinstalls"
    echo "Then use save_drone_db.sh to preserve the OAuth credentials for future reinstalls."
fi

# Create and push repo using standalone script
#echo ""
#echo "Creating initial repository..."
#GITEA_URL="$GITEA_URL" \
#REPO_NAME="$REPO_NAME" \
#SOURCE_DIR="$EXAMPLE_PROJECT_DIR" \
#bash "$SCRIPT_DIR/create_gitea_repo.sh"


echo ""
echo "=========================================="
echo "Gitea Installation Complete!"
echo "=========================================="
if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
    echo "✅ RESTORED from backup with preserved OAuth state"
    echo ""
    echo "To save this state for future reinstalls:"
    echo "  ./save_gitea_db.sh"
else
    echo "✅ FRESH install"
    echo ""
    echo "To save this state (preserves OAuth, users, repos):"
    echo "  ./save_gitea_db.sh"
    echo ""
    echo "Note: Run this after setting up Drone OAuth to preserve credentials."
fi
echo "=========================================="