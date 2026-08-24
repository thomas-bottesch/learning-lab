#!/usr/bin/env bash
set -euo pipefail

# Build NInfer Docker image from GitHub source
# This script:
#   1. Clones the ninfer repository to /tmp/ninfer-build
#   2. Builds the Docker image as 'ninfer:local'
#   3. Cleans up /tmp/ninfer-build

REPO_URL="https://github.com/Neroued/ninfer.git"
BUILD_DIR="/tmp/ninfer-build"
IMAGE_TAG="ninfer:local"

echo "=== NInfer Docker Build Script ==="
echo ""

# Clean up any previous build directory
if [ -d "$BUILD_DIR" ]; then
    echo "Removing previous build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# Clone the repository
echo "Cloning NInfer repository to $BUILD_DIR ..."
git clone "$REPO_URL" "$BUILD_DIR"

echo "Repository cloned successfully."

# Build the Docker image
echo ""
echo "Building Docker image '$IMAGE_TAG' ..."
cd "$BUILD_DIR"
docker build --tag "$IMAGE_TAG" .

echo ""
echo "Docker image '$IMAGE_TAG' built successfully."

# Clean up
echo ""
echo "Cleaning up build directory: $BUILD_DIR ..."
rm -rf "$BUILD_DIR"

echo "Build complete. Image '$IMAGE_TAG' is available."
echo "You can now use 'scripts/download-model.sh' to download a model,"
echo "then 'docker compose up' to start the HTTP service."
