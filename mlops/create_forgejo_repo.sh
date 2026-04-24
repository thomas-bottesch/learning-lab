#!/usr/bin/env bash
set -euo pipefail

# Script to create and push a Git repository to Forgejo
# Usage: ./create_forgejo_repo.sh <ORG_NAME> <REPO_NAME> <SOURCE_DIR>

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_DIR="$SCRIPT_DIR/k8s_yamls/forgejo"
KUBEFLOW_YAML_DIR="$SCRIPT_DIR/k8s_yamls/kubeflow"

# Static Forgejo URL
FORGEJO_URL="http://localhost:4000"

# Extract credentials from YAML
FORGEJO_USER=$(grep 'admin-username:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_PASSWORD=$(grep 'admin-password:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_EMAIL=$(grep 'admin-email:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')

# Display help message
show_help() {
    cat << EOF
Usage: $0 <ORG_NAME> <REPO_NAME> <SOURCE_DIR>

Arguments:
  ORG_NAME     Name of organization (will be created if doesn't exist)
  REPO_NAME    Name of repository to create
  SOURCE_DIR    Path to directory containing files to push

Example:
  $0 my-org my_ml_project ./example_git_project

EOF
    exit 1
}

# Parse command-line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
fi

if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments." >&2
    echo ""
    show_help
fi

ORG_NAME="$1"
REPO_NAME="$2"
SOURCE_DIR="$3"

# Temporary directory for working
TMP_PROJECT_DIR="/tmp/$REPO_NAME"

# Helper functions
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command not found: $1" >&2
    exit 1
  }
}

wait_for_forgejo() {
    echo "Waiting for Forgejo web server to be available..."
    local i
    for i in {1..30}; do
        if curl -s --head --fail "$FORGEJO_URL" > /dev/null; then
            echo "Forgejo web server is up."
            return 0
        else
            echo "Forgejo web server not ready yet, retrying ($i)..."
            sleep 2
        fi
    done
    echo "Error: Forgejo web server did not become available in time." >&2
    exit 1
}

check_org_exists() {
    local org_name="$1"
    
    echo "Checking if organization '$org_name' exists..."
    local org_check
    org_check=$(curl -s -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
        "$FORGEJO_URL/api/v1/orgs/$org_name" 2>/dev/null || echo "{}")
    
    if echo "$org_check" | grep -q '"id"'; then
        echo "✓ Organization '$org_name' already exists."
        return 0
    else
        return 1
    fi
}

create_org() {
    local org_name="$1"
    
    echo "Creating organization '$org_name' via Forgejo API..."
    local response
    response=$(curl -s -X POST "$FORGEJO_URL/api/v1/orgs" \
        -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
        -H "Content-Type: application/json" \
        -d "{\"username\": \"$org_name\", \"visibility\": \"public\"}" 2>/dev/null || echo "{}")
    
    if echo "$response" | grep -q '"message"'; then
        echo "⚠️ API Error creating organization: $response" >&2
        exit 1
    fi
    
    echo "✓ Organization '$org_name' created successfully!"
}

create_repo_via_api() {
    local target_namespace="$1"
    
    echo "Checking if Forgejo repo exists in '$target_namespace/$REPO_NAME'..."
    local repo_check
    repo_check=$(curl -s -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
        "$FORGEJO_URL/api/v1/repos/$target_namespace/$REPO_NAME" 2>/dev/null || echo "{}")
    
    if echo "$repo_check" | grep -q '"name"'; then
        echo "✓ Repository '$target_namespace/$REPO_NAME' already exists."
        return 0
    fi
    
    echo "Creating repository '$target_namespace/$REPO_NAME' via Forgejo API..."
    local response
    response=$(curl -s -X POST "$FORGEJO_URL/api/v1/org/$ORG_NAME/repos" \
        -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$REPO_NAME\"}" 2>/dev/null || echo "{}")
    
    if echo "$response" | grep -q '"message"'; then
        echo "⚠️ API Error: $response" >&2
        exit 1
    fi
    
    echo "✓ Repository created successfully!"
}

setup_actions_secrets() {
    local target_namespace="$1"
    
    echo ""
    echo "Setting up Forgejo Actions secrets and variables..."
    
    # Temporarily disable error checking for this function
    set +e
    
    # Extract Kubeflow username from user-profile.yaml
    KUBEFLOW_USERNAME=$(grep 'name: user@example.com' "$KUBEFLOW_YAML_DIR/user-profile.yaml" 2>/dev/null | awk -F': ' '{print $2}' | tr -d '" ')
    
    # Extract endpoints from configmap
    MINIO_ENDPOINT=$(grep 'MINIO_ENDPOINT:' "$KUBEFLOW_YAML_DIR/mlops-endpoints-configmap.yaml" 2>/dev/null | awk -F': ' '{print $2}' | tr -d '" ')
    LAKEFS_ENDPOINT=$(grep 'LAKEFS_ENDPOINT:' "$KUBEFLOW_YAML_DIR/mlops-endpoints-configmap.yaml" 2>/dev/null | awk -F': ' '{print $2}' | tr -d '" ')
    LAKEFS_BUCKET_NAME=$(grep 'LAKEFS_BUCKET_NAME:' "$KUBEFLOW_YAML_DIR/mlops-endpoints-configmap.yaml" 2>/dev/null | awk -F': ' '{print $2}' | tr -d '" ')
    DVC_BUCKET_NAME=$(grep 'DVC_BUCKET_NAME:' "$KUBEFLOW_YAML_DIR/mlops-endpoints-configmap.yaml" 2>/dev/null | awk -F': ' '{print $2}' | tr -d '" ')
    
    # Extract MLflow and MinIO credentials from their secrets
    MINIO_ACCESS_KEY=$(grep 'MINIO_ROOT_USER:' "$SCRIPT_DIR/k8s_yamls/minio/02-secret.yaml" 2>/dev/null | awk -F': ' '{print $2}' | tr -d '" ')
    MINIO_SECRET_KEY=$(grep 'MINIO_ROOT_PASSWORD:' "$SCRIPT_DIR/k8s_yamls/minio/02-secret.yaml" 2>/dev/null | awk -F': ' '{print $2}' | tr -d '" ')
    
    MLFLOW_TRACKING_URI="http://mlflow.mlflow.svc.cluster.local:5000"
    
    # Extract LakeFS credentials from its secrets
    LAKEFS_ACCESS_KEY=$(grep 'LAKEFS_AUTH_ADMIN_ACCESS_KEY_ID:' "$SCRIPT_DIR/k8s_yamls/lakefs/02-secret.yaml" 2>/dev/null | awk -F': ' '{print $2}' | tr -d '" ')
    LAKEFS_SECRET_KEY=$(grep 'LAKEFS_AUTH_ADMIN_SECRET_ACCESS_KEY:' "$SCRIPT_DIR/k8s_yamls/lakefs/02-secret.yaml" 2>/dev/null | awk -F': ' '{print $2}' | tr -d '" ')
    
    # Internal cluster service URLs (for runners inside k3s)
    KUBEFLOW_HOST="http://172.17.0.1:8080"
    KUBEFLOW_PASSWORD="12341234"
    KUBEFLOW_NAMESPACE="user-example-com"

    echo "  Extracted configuration:"
    echo "    KUBEFLOW_USERNAME: $KUBEFLOW_USERNAME"
    echo "    MINIO_ENDPOINT: $MINIO_ENDPOINT"
    echo "    LAKEFS_ENDPOINT: $LAKEFS_ENDPOINT"
    echo ""
    
    # Set up VARIABLES (non-sensitive data: hostnames, endpoints, bucket names, namespaces, usernames)
    echo "  Setting up variables (non-sensitive data)..."
    PYPI_INDEX_URL="http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/"
    local variables=(
        "KUBEFLOW_HOST:$KUBEFLOW_HOST"
        "KUBEFLOW_USERNAME:$KUBEFLOW_USERNAME"
        "DEX_USERNAME:$KUBEFLOW_USERNAME"
        "MLFLOW_TRACKING_URI:$MLFLOW_TRACKING_URI"
        "MINIO_ENDPOINT:$MINIO_ENDPOINT"
        "LAKEFS_ENDPOINT:$LAKEFS_ENDPOINT"
        "LAKEFS_BUCKET_NAME:$LAKEFS_BUCKET_NAME"
        "DVC_BUCKET_NAME:$DVC_BUCKET_NAME"
        "KUBEFLOW_NAMESPACE:$KUBEFLOW_NAMESPACE"
        "PYPI_INDEX_URL:$PYPI_INDEX_URL"
    )
    
    for variable_pair in "${variables[@]}"; do
        IFS=':' read -r variable_name variable_value <<< "$variable_pair"
        
        echo "  Creating variable: $variable_name"
        local response
        response=$(curl -s -X POST "$FORGEJO_URL/api/v1/repos/$target_namespace/$REPO_NAME/actions/variables/$variable_name" \
            -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
            -H "Content-Type: application/json" \
            -d "{\"value\": \"$variable_value\"}" 2>/dev/null || echo "{}")
        
        if echo "$response" | grep -q '"message"' && ! echo "$response" | grep -q '"updated"\|"created"'; then
            echo "  ⚠️  Warning: Could not set variable '$variable_name'"
            echo "     Response: $response"
        else
            echo "  ✓ Variable '$variable_name' created/updated"
        fi
    done
    
    # Set up SECRETS (sensitive data: passwords, access keys, secret keys)
    echo ""
    echo "  Setting up secrets (sensitive data)..."
    local secrets=(
        "KUBEFLOW_PASSWORD:$KUBEFLOW_PASSWORD"
        "DEX_PASSWORD:$KUBEFLOW_PASSWORD"
        "MINIO_ACCESS_KEY:$MINIO_ACCESS_KEY"
        "MINIO_SECRET_KEY:$MINIO_SECRET_KEY"
        "LAKEFS_ACCESS_KEY:$LAKEFS_ACCESS_KEY"
        "LAKEFS_SECRET_KEY:$LAKEFS_SECRET_KEY"
        "AWS_ACCESS_KEY_ID:$MINIO_ACCESS_KEY"
        "AWS_SECRET_ACCESS_KEY:$MINIO_SECRET_KEY"
    )
    
    for secret_pair in "${secrets[@]}"; do
        IFS=':' read -r secret_name secret_value <<< "$secret_pair"
        
        echo "  Creating secret: $secret_name"
        local response
        response=$(curl -s -X PUT "$FORGEJO_URL/api/v1/repos/$target_namespace/$REPO_NAME/actions/secrets/$secret_name" \
            -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
            -H "Content-Type: application/json" \
            -d "{\"data\": \"$secret_value\"}" 2>/dev/null || echo "{}")
        
        if echo "$response" | grep -q '"message"' && ! echo "$response" | grep -q '"updated"\|"created"'; then
            echo "  ⚠️  Warning: Could not set secret '$secret_name'"
            echo "     Response: $response"
        else
            echo "  ✓ Secret '$secret_name' created/updated"
        fi
    done
    
    echo ""
    echo "✓ Actions secrets and variables configured!"
    echo ""
    echo "  ⚠️  IMPORTANT: Please update these secrets in repository settings:"
    echo "  Repository → Settings → Secrets → Actions"
    echo ""
    echo "  Required updates:"
    echo "  - KUBEFLOW_PASSWORD: Set your actual Kubeflow password"
    echo ""
    echo "  All other secrets and variables have been auto-configured with internal cluster URLs."
    echo "  Non-sensitive data (hostnames, endpoints, bucket names, namespaces) are stored as variables."
    
    # Re-enable error checking
    set -e
}

setup_and_push_repo() {
    echo ""
    echo "Setting up local repository..."
    
    # Prepare temporary directory
    if [ -d "$TMP_PROJECT_DIR" ]; then
        echo "Removing existing temporary directory: $TMP_PROJECT_DIR"
        rm -rf "$TMP_PROJECT_DIR"
    fi
    
    mkdir -p "$TMP_PROJECT_DIR"
    echo "Copying files from '$SOURCE_DIR' to '$TMP_PROJECT_DIR'..."
    cp -r "$SOURCE_DIR"/. "$TMP_PROJECT_DIR/"
    
    cd "$TMP_PROJECT_DIR"
    
    # Initialize git if needed
    if [ ! -d .git ]; then
        git init
        git config user.name "$FORGEJO_USER"
        git config user.email "$FORGEJO_EMAIL"
        echo "✓ Git repository initialized"
    fi
    
    # Stage files
    git add .
    
    # Commit if there are changes
    if git diff --cached --quiet; then
        echo "✓ No changes to commit (working tree clean)"
    else
        git commit -m "Initial commit" || echo "Note: Commit may already exist"
        echo "✓ Changes committed"
    fi
    
    # Determine remote URL (always use organization)
    local remote_url="$FORGEJO_URL/$ORG_NAME/$REPO_NAME.git"
    local remote_url_auth="http://$FORGEJO_USER:$FORGEJO_PASSWORD@localhost:4000/$ORG_NAME/$REPO_NAME.git"
    
    # Configure remote
    if git remote | grep -q origin; then
        git remote set-url origin "$remote_url_auth"
        echo "✓ Updated remote 'origin' to: $remote_url"
    else
        git remote add origin "$remote_url_auth"
        echo "✓ Added remote 'origin': $remote_url"
    fi
    
    # Push to repository
    echo "Pushing to remote repository..."
    if git push -u origin master 2>/dev/null || git push -u origin main 2>/dev/null; then
        echo "✓ Pushed successfully!"
    else
        echo "Note: Push may already be up to date or branch name differs"
    fi
    
    cd - >/dev/null
}

# Validate required commands
require_cmd curl
require_cmd git

# Display configuration
echo "=========================================="
echo "Forgejo Repository Creator"
echo "=========================================="
echo "Forgejo URL:       $FORGEJO_URL"
echo "Username:         $FORGEJO_USER"
echo "Organization:      $ORG_NAME"
echo "Repository Name:  $REPO_NAME"
echo "Source Directory: $SOURCE_DIR"
echo "=========================================="
echo ""

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist." >&2
    exit 1
fi

# Wait for Forgejo to be ready
wait_for_forgejo

# Check and create organization if needed
if ! check_org_exists "$ORG_NAME"; then
    create_org "$ORG_NAME"
fi

# Set target namespace (always is organization)
target_namespace="$ORG_NAME"
echo "Creating repository in organization: $ORG_NAME"

# Create repository via API
create_repo_via_api "$target_namespace"

# Setup Actions secrets
setup_actions_secrets "$target_namespace"

# Setup and push repository
setup_and_push_repo

# Display summary
echo ""
echo "=========================================="
echo "✅ Repository Setup Complete!"
echo "=========================================="
echo "Repository URL: $FORGEJO_URL/$target_namespace/$REPO_NAME"
echo "Local copy:     $TMP_PROJECT_DIR"
echo ""
echo "To work with this repository:"
echo "  cd $TMP_PROJECT_DIR"
echo "  # Make changes..."
echo "  git add ."
echo "  git commit -m 'Your message'"
echo "  git push"
echo "=========================================="