#!/usr/bin/env bash
set -euo pipefail

# Script to create and push a Git repository to Gitea
# Can create repos in user's personal account or in an organization

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_DIR="$SCRIPT_DIR/k8s_yamls/gitea"

# Configuration with defaults
# Gitea URL can be overridden via environment variable
GITEA_URL="${GITEA_URL:-http://localhost:3000}"

# If credentials are not provided via environment, extract them from YAML
if [ -z "${GITEA_USER:-}" ]; then
    GITEA_USER=$(grep 'admin-username:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
fi

if [ -z "${GITEA_PASSWORD:-}" ]; then
    GITEA_PASSWORD=$(grep 'admin-password:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
fi

if [ -z "${GITEA_EMAIL:-}" ]; then
    GITEA_EMAIL=$(grep 'admin-email:' "$YAML_DIR/02-secret.yaml" | awk -F': ' '{print $2}' | tr -d '"')
fi

# Repository configuration
REPO_NAME="${REPO_NAME:-my_ml_project}"
SOURCE_DIR="${SOURCE_DIR:-./example_git_project}"
ORG_NAME="${ORG_NAME:-}"  # Optional: if set, creates repo in organization

# Temporary directory for working
TMP_PROJECT_DIR="/tmp/$REPO_NAME"

# Helper functions
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command not found: $1" >&2
    exit 1
  }
}

wait_for_gitea() {
    echo "Waiting for Gitea web server to be available..."
    local i
    for i in {1..30}; do
        if curl -s --head --fail "$GITEA_URL" > /dev/null; then
            echo "Gitea web server is up."
            return 0
        else
            echo "Gitea web server not ready yet, retrying ($i)..."
            sleep 2
        fi
    done
    echo "Error: Gitea web server did not become available in time." >&2
    exit 1
}

create_repo_via_api() {
    local target_namespace="$1"
    
    echo "Checking if Gitea repo exists in '$target_namespace/$REPO_NAME'..."
    local repo_check
    repo_check=$(curl -s -u "$GITEA_USER:$GITEA_PASSWORD" \
        "$GITEA_URL/api/v1/repos/$target_namespace/$REPO_NAME" 2>/dev/null || echo "{}")
    
    if echo "$repo_check" | grep -q '"name"'; then
        echo "✓ Repository '$target_namespace/$REPO_NAME' already exists."
        return 0
    fi
    
    echo "Creating repository '$target_namespace/$REPO_NAME' via Gitea API..."
    local api_endpoint
    local payload
    
    if [ -n "$ORG_NAME" ]; then
        api_endpoint="$GITEA_URL/api/v1/org/$ORG_NAME/repos"
        payload="{\"name\": \"$REPO_NAME\"}"
    else
        api_endpoint="$GITEA_URL/api/v1/user/repos"
        payload="{\"name\": \"$REPO_NAME\"}"
    fi
    
    local response
    response=$(curl -s -X POST "$api_endpoint" \
        -u "$GITEA_USER:$GITEA_PASSWORD" \
        -H "Content-Type: application/json" \
        -d "$payload" 2>/dev/null || echo "{}")
    
    if echo "$response" | grep -q '"message"'; then
        echo "⚠️ API Error: $response" >&2
        exit 1
    fi
    
    echo "✓ Repository created successfully!"
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
    cp -r "$SOURCE_DIR"/* "$TMP_PROJECT_DIR/"
    
    cd "$TMP_PROJECT_DIR"
    
    # Initialize git if needed
    if [ ! -d .git ]; then
        git init
        git config user.name "$GITEA_USER"
        git config user.email "$GITEA_EMAIL"
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
    
    # Determine remote URL
    local target_namespace
    if [ -n "$ORG_NAME" ]; then
        target_namespace="$ORG_NAME"
    else
        target_namespace="$GITEA_USER"
    fi
    
    local remote_url="$GITEA_URL/$target_namespace/$REPO_NAME.git"
    local remote_url_auth="http://$GITEA_USER:$GITEA_PASSWORD@localhost:3000/$target_namespace/$REPO_NAME.git"
    
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
echo "Gitea Repository Creator"
echo "=========================================="
echo "Gitea URL:        $GITEA_URL"
echo "Username:         $GITEA_USER"
echo "Repository Name:  $REPO_NAME"
echo "Source Directory: $SOURCE_DIR"
if [ -n "$ORG_NAME" ]; then
    echo "Organization:     $ORG_NAME"
else
    echo "Organization:     (personal account)"
fi
echo "=========================================="
echo ""

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist." >&2
    exit 1
fi

# Wait for Gitea to be ready
wait_for_gitea

# Determine target namespace for the repo
if [ -n "$ORG_NAME" ]; then
    target_namespace="$ORG_NAME"
    echo "Creating repository in organization: $ORG_NAME"
else
    target_namespace="$GITEA_USER"
    echo "Creating repository in personal account of: $GITEA_USER"
fi

# Create repository via API
create_repo_via_api "$target_namespace"

# Setup and push repository
setup_and_push_repo

# Display summary
echo ""
echo "=========================================="
echo "✅ Repository Setup Complete!"
echo "=========================================="
echo "Repository URL: $GITEA_URL/$target_namespace/$REPO_NAME"
echo "Local copy:     $TMP_PROJECT_DIR"
echo ""
echo "To work with this repository:"
echo "  cd $TMP_PROJECT_DIR"
echo "  # Make changes..."
echo "  git add ."
echo "  git commit -m 'Your message'"
echo "  git push"
echo "=========================================="