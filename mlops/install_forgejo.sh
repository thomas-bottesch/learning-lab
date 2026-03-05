#!/usr/bin/env bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# URLs
FORGEJO_URL="http://localhost:4000"

# Extract Forgejo credentials from YAML
YAML_DIR="k8s_yamls/forgejo"
FORGEJO_USER=$(grep 'admin-username:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_PASSWORD=$(grep 'admin-password:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_EMAIL=$(grep 'admin-email:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

print_section() {
    echo ""
    echo -e "${BLUE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if running as root or with sudo
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. This might not be necessary."
fi

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DB_BACKUP_PATH="$SCRIPT_DIR/forgejo-prepared-db"

print_header "Forgejo Installation Script"

# Build CI Docker image if it doesn't exist
print_section "Checking CI Docker image"
CI_IMAGE_NAME="forgejo-runner-python3.12:latest"
CI_DOCKERFILE="host_config/docker_images/Dockerfile.ci"

if sudo docker image ls | grep -q "^forgejo-runner-python3.12.*latest"; then
    print_success "CI image $CI_IMAGE_NAME already exists, skipping build"
else
    print_info "Building CI image $CI_IMAGE_NAME..."
    if sudo docker build -t "$CI_IMAGE_NAME" -f "$CI_DOCKERFILE" .; then
        print_success "CI image built successfully"
    else
        print_error "Failed to build CI image"
        exit 1
    fi
fi

# Check for saved database backup first
if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
    print_info "Found saved database backup at $DB_BACKUP_PATH/db.tar.gz"
    print_info "Preparing PVC and restoring database before deploying Forgejo..."
    
    # Apply only namespace and PVC first
    print_section "Creating namespace and PVC"
    kubectl apply -f k8s_yamls/forgejo/01-namespace.yaml
    kubectl apply -f k8s_yamls/forgejo/03-pvc.yaml
    print_success "Namespace and PVC created"
    
    # Create inline restore helper pod
    print_section "Starting inline restore helper"
    cat << 'PODEOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: forgejo-inline-restore
  namespace: forgejo
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
        - name: data
          mountPath: /data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: forgejo-data-pvc
PODEOF
    
    # Wait for PVC to be bound
    print_info "Waiting for PVC to be bound..."
    kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/forgejo-data-pvc -n forgejo --timeout=2m || {
        print_warning "PVC not bound yet, continuing anyway..."
    }

    # Wait for helper pod to be ready
    print_info "Waiting for restore helper to be ready..."
    kubectl wait --for=condition=ready pod forgejo-inline-restore -n forgejo --timeout=60s || {
        print_error "Restore helper pod failed to start"
        exit 1
    }
    
    # Copy and extract database
    print_section "Restoring database to PVC"
    kubectl cp "$DB_BACKUP_PATH/db.tar.gz" "forgejo/forgejo-inline-restore:/tmp/db.tar.gz"
    kubectl exec forgejo-inline-restore -n forgejo -- sh -c '
        mkdir -p /tmp/extract
        tar xzf /tmp/db.tar.gz -C /tmp/extract
        mkdir -p /data/forgejo
        cp -r /tmp/extract/backup/* /data/forgejo/
        chmod -R 777 /data/forgejo
        echo "Database restored to PVC"
        ls -la /data/forgejo/
    '

    # Clean up restore pod
    print_section "Cleaning up restore helper"
    kubectl delete pod forgejo-inline-restore -n forgejo --ignore-not-found=true --timeout=3s || true
    
    print_success "Database restored. Now deploying Forgejo..."
    
    # Apply remaining YAMLs (namespace, pvc already applied)
    print_section "Creating Kubernetes resources"
    kubectl apply -f k8s_yamls/forgejo/02-secret.yaml
    kubectl apply -f k8s_yamls/forgejo/04-deployment.yaml
    kubectl apply -f k8s_yamls/forgejo/05-service.yaml
    kubectl apply -f k8s_yamls/forgejo/06-runner-deployment.yaml
    print_success "Kubernetes resources created"
else
    # No backup - apply all YAMLs normally
    print_section "Creating Kubernetes resources"
    kubectl apply -f k8s_yamls/forgejo/
    print_success "Kubernetes resources created"
fi

print_section "Waiting for Forgejo pod to be ready"
# Wait for deployment to be ready
kubectl wait --for=condition=available --timeout=300s deployment/forgejo -n forgejo

# Wait for web server to start
MAX_RETRIES=60
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if kubectl logs -n forgejo deployment/forgejo | grep -q "Starting new Web server"; then
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

# Check if we restored from backup
if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
    print_section "RESTORED from backup - skipping admin user creation"
    print_success "Admin user and OAuth state already exist from backup"
else
    print_section "Creating admin user"
    kubectl exec -n forgejo deployment/forgejo -- gitea admin user create \
        --username "$FORGEJO_USER" \
        --password "$FORGEJO_PASSWORD" \
        --email "$FORGEJO_EMAIL" \
        --admin \
        --must-change-password=false 2>/dev/null || echo "Admin user may already exist, continuing..."
    print_success "Admin user created"
fi

print_section "Waiting for Forgejo web server"
# Wait for web server to respond
MAX_RETRIES=60
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "$FORGEJO_URL/api/healthz" > /dev/null 2>&1; then
        print_success "Forgejo web server is responding"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_error "Forgejo web server did not respond within expected time"
    exit 1
fi


#print_section "Deploying Forgejo Runner"
# Runner token is pre-configured in k8s_yamls/forgejo/06-runner-deployment.yaml
# No token generation or secret creation needed!
#print_info "Deploying Forgejo Runner with pre-configured token..."
#kubectl apply -f k8s_yamls/forgejo/06-runner-deployment.yaml
#print_success "Runner deployment created"


print_section "Waiting for Runner to register"
# Wait for runner to be ready
kubectl wait --for=condition=ready --timeout=300s pod -l app=forgejo-runner -n forgejo

# Wait a bit more for runner to register
sleep 5

# Check if runner is registered
RUNNER_CHECK=$(curl -s -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
    "$FORGEJO_URL/api/v1/admin/runners" | grep -c "k8s-runner" || echo "0")

if [ "$RUNNER_CHECK" -gt "0" ]; then
    print_success "Runner registered successfully!"
else
    print_warning "Runner may still be registering. Check logs with: kubectl logs -n forgejo -l app=forgejo-runner"
fi

print_header "Installation Complete!"

echo ""
echo "Forgejo is now running:"
echo "  Web UI:      $FORGEJO_URL"
echo "  Username:    $FORGEJO_USER"
echo "  Password:    $FORGEJO_PASSWORD"
echo ""
echo "Next steps:"
echo "  1. Open $FORGEJO_URL in your browser"
echo "  2. Login with the credentials above"
echo "  3. Actions and Runner are enabled by default"
echo "  4. Create a repository and add a .forgejo/workflows/*.yaml file"
echo ""
print_warning "For security, change the default password after first login!"
print_warning "Update k8s_yamls/forgejo/02-secret.yaml to use your own secret keys!"
echo ""

# Show backup information
echo "=========================================="
if [ -f "$DB_BACKUP_PATH/db.tar.gz" ]; then
    echo "✅ RESTORED from backup with preserved OAuth state"
    echo ""
    echo "To save this state for future reinstalls:"
    echo "  ./save_forgejo_db.sh"
else
    echo "✅ FRESH install"
    echo ""
    echo "To save this state (preserves OAuth, users, repos):"
    echo "  ./save_forgejo_db.sh"
fi
echo "=========================================="
