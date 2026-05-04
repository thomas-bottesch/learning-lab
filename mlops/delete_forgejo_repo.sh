#!/usr/bin/env bash
set -euo pipefail

# Script to delete a Git repository from Forgejo
# Usage: ./delete_forgejo_repo.sh <ORG_NAME> <REPO_NAME> [--delete-org]

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_DIR="$SCRIPT_DIR/k8s_yamls/forgejo"

# Static Forgejo URL
FORGEJO_URL="http://localhost:4000"

# Extract credentials from YAML
FORGEJO_USER=$(grep 'admin-username:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_PASSWORD=$(grep 'admin-password:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_EMAIL=$(grep 'admin-email:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')

# Display help message
show_help() {
    cat << EOF
Usage: $0 <ORG_NAME> <REPO_NAME> [--delete-org]

Arguments:
  ORG_NAME      Name of organization containing the repository
  REPO_NAME     Name of repository to delete
  --delete-org  (Optional) Also delete the organization after deleting the repo

Example:
  $0 my-org my_ml_project
  $0 my-org my_ml_project --delete-org

EOF
    exit 1
}

# Parse command-line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
fi

if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments." >&2
    echo ""
    show_help
fi

ORG_NAME="$1"
REPO_NAME="$2"
DELETE_ORG=false

# Check for optional --delete-org flag
if [ "${3:-}" = "--delete-org" ]; then
    DELETE_ORG=true
fi

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

check_repo_exists() {
    local target_namespace="$1"
    local repo_name="$2"
    
    echo "Checking if repository '$target_namespace/$repo_name' exists..."
    local repo_check
    repo_check=$(curl -s -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
        "$FORGEJO_URL/api/v1/repos/$target_namespace/$repo_name" 2>/dev/null || echo "{}")
    
    if echo "$repo_check" | grep -q '"name"'; then
        echo "✓ Repository '$target_namespace/$repo_name' found."
        return 0
    else
        echo "⚠️  Repository '$target_namespace/$repo_name' does not exist."
        return 1
    fi
}

check_org_exists() {
    local org_name="$1"
    
    echo "Checking if organization '$org_name' exists..."
    local org_check
    org_check=$(curl -s -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
        "$FORGEJO_URL/api/v1/orgs/$org_name" 2>/dev/null || echo "{}")
    
    if echo "$org_check" | grep -q '"id"'; then
        echo "✓ Organization '$org_name' found."
        return 0
    else
        echo "⚠️  Organization '$org_name' does not exist."
        return 1
    fi
}

delete_repo() {
    local target_namespace="$1"
    local repo_name="$2"
    
    echo "Deleting repository '$target_namespace/$repo_name' via Forgejo API..."
    local response
    response=$(curl -s -X DELETE "$FORGEJO_URL/api/v1/repos/$target_namespace/$repo_name" \
        -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
        -H "Content-Type: application/json" 2>/dev/null || echo "{}")
    
    if echo "$response" | grep -q '"message"' && ! echo "$response" | grep -q '"deleted"'; then
        echo "⚠️  API Error deleting repository: $response" >&2
        exit 1
    fi
    
    echo "✓ Repository '$target_namespace/$repo_name' deleted successfully!"
}

delete_org() {
    local org_name="$1"
    
    echo "Deleting organization '$org_name' via Forgejo API..."
    local response
    response=$(curl -s -X DELETE "$FORGEJO_URL/api/v1/orgs/$org_name" \
        -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
        -H "Content-Type: application/json" 2>/dev/null || echo "{}")
    
    if echo "$response" | grep -q '"message"' && ! echo "$response" | grep -q '"deleted"'; then
        echo "⚠️  API Error deleting organization: $response" >&2
        exit 1
    fi
    
    echo "✓ Organization '$org_name' deleted successfully!"
}

# Validate required commands
require_cmd curl

# Display configuration
echo "=========================================="
echo "Forgejo Repository Deleter"
echo "=========================================="
echo "Forgejo URL:       $FORGEJO_URL"
echo "Username:          $FORGEJO_USER"
echo "Organization:      $ORG_NAME"
echo "Repository Name:   $REPO_NAME"
echo "Delete Organization: $DELETE_ORG"
echo "=========================================="
echo ""

# Wait for Forgejo to be ready
wait_for_forgejo

# Set target namespace (always is organization)
target_namespace="$ORG_NAME"

# Check and delete repository
if check_repo_exists "$target_namespace" "$REPO_NAME"; then
    delete_repo "$target_namespace" "$REPO_NAME"
else
    echo "Skipping repository deletion (repo does not exist)."
fi

# Optionally delete organization
if [ "$DELETE_ORG" = true ]; then
    echo ""
    if check_org_exists "$target_namespace"; then
        delete_org "$target_namespace"
    else
        echo "Skipping organization deletion (org does not exist)."
    fi
fi

# Display summary
echo ""
echo "=========================================="
echo "✅ Deletion Complete!"
echo "=========================================="
echo "Deleted repository: $FORGEJO_URL/$target_namespace/$REPO_NAME"
if [ "$DELETE_ORG" = true ]; then
    echo "Deleted organization: $FORGEJO_URL/$target_namespace"
fi
echo "=========================================="
