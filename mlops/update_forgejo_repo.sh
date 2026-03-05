#!/usr/bin/env bash
set -euo pipefail

# Script to update an existing Git repository in Forgejo
# Usage: ./update_forgejo_repo.sh <ORG_NAME> <REPO_NAME> <SOURCE_DIR>

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
Usage: $0 <ORG_NAME> <REPO_NAME> <SOURCE_DIR>

Arguments:
  ORG_NAME     Name of organization
  REPO_NAME    Name of existing repository to update
  SOURCE_DIR    Path to directory containing files to copy

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
TMP_PROJECT_DIR="/tmp/${REPO_NAME}-update"

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
    local org_name="$1"
    local repo_name="$2"
    
    echo "Checking if repository '$org_name/$repo_name' exists..."
    local repo_check
    repo_check=$(curl -s -u "$FORGEJO_USER:$FORGEJO_PASSWORD" \
        "$FORGEJO_URL/api/v1/repos/$org_name/$repo_name" 2>/dev/null || echo "{}")
    
    if echo "$repo_check" | grep -q '"name"'; then
        echo "✓ Repository '$org_name/$repo_name' exists."
        return 0
    else
        echo "✗ Repository '$org_name/$repo_name' does not exist." >&2
        return 1
    fi
}

update_repo() {
    echo ""
    echo "Updating repository..."
    
    # Prepare temporary directory
    if [ -d "$TMP_PROJECT_DIR" ]; then
        echo "Removing existing temporary directory: $TMP_PROJECT_DIR"
        rm -rf "$TMP_PROJECT_DIR"
    fi
    
    mkdir -p "$TMP_PROJECT_DIR"
    
    # Clone the existing repository
    echo "Cloning repository from Forgejo..."
    local remote_url="http://$FORGEJO_USER:$FORGEJO_PASSWORD@localhost:4000/$ORG_NAME/$REPO_NAME.git"
    
    if ! git clone "$remote_url" "$TMP_PROJECT_DIR"; then
        echo "Error: Failed to clone repository." >&2
        exit 1
    fi
    
    echo "✓ Repository cloned successfully"
    
    # Copy files from source directory
    echo ""
    echo "Copying files from '$SOURCE_DIR' to repository..."
    
    # Get absolute path of source directory
    local abs_source_dir
    abs_source_dir=$(cd "$SOURCE_DIR" && pwd)
    
    cp -r "$abs_source_dir"/. "$TMP_PROJECT_DIR/"
    echo "✓ Files copied"
    
    cd "$TMP_PROJECT_DIR"
    
    # Configure git user if needed
    git config user.name "$FORGEJO_USER"
    git config user.email "$FORGEJO_EMAIL"
    
    # Stage all changes
    echo ""
    echo "Staging changes..."
    git add .
    
    # Check if there are changes to commit
    if git diff --cached --quiet; then
        echo "✓ No changes to commit (working tree clean)"
        cd - >/dev/null
        return 0
    fi
    
    # Show what changed
    echo ""
    echo "Changes to be committed:"
    git status --short
    
    # Commit changes
    echo ""
    echo "Committing changes..."
    local commit_message="Update repository content - $(date '+%Y-%m-%d %H:%M:%S')"
    git commit -m "$commit_message"
    echo "✓ Changes committed"
    
    # Push changes
    echo ""
    echo "Pushing changes to remote repository..."
    local branch_name=$(git rev-parse --abbrev-ref HEAD)
    
    if git push -u origin "$branch_name"; then
        echo "✓ Pushed successfully!"
    else
        echo "Error: Failed to push changes." >&2
        exit 1
    fi
    
    cd - >/dev/null
}

# Validate required commands
require_cmd curl
require_cmd git

# Display configuration
echo "=========================================="
echo "Forgejo Repository Updater"
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

# Check if repository exists
if ! check_repo_exists "$ORG_NAME" "$REPO_NAME"; then
    echo ""
    echo "Error: Repository does not exist." >&2
    echo "Please create it first using: ./create_forgejo_repo.sh $ORG_NAME $REPO_NAME $SOURCE_DIR" >&2
    exit 1
fi

# Update repository
update_repo

# Display summary
echo ""
echo "=========================================="
echo "✅ Repository Update Complete!"
echo "=========================================="
echo "Repository URL: $FORGEJO_URL/$ORG_NAME/$REPO_NAME"
echo "Local copy:     $TMP_PROJECT_DIR"
echo ""
echo "To work with this repository:"
echo "  cd $TMP_PROJECT_DIR"
echo "  # Make changes..."
echo "  git add ."
echo "  git commit -m 'Your message'"
echo "  git push"
echo "=========================================="