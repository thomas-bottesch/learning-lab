#!/bin/bash
# install_minio.sh: Apply all MinIO-related Kubernetes YAMLs

set -e

YAML_DIR="k8s_yamls/minio"

for yaml in "$YAML_DIR"/*.yaml; do
    echo "Applying $yaml..."
    kubectl apply -f "$yaml"
done

echo "All MinIO YAMLs applied successfully."

# Wait for MinIO deployment to be ready
echo "Waiting for MinIO deployment to be ready..."
kubectl rollout status deployment/minio -n minio --timeout=5m

# Port-forward MinIO Console to host
echo "Starting port-forward for MinIO Console (port 9001)..."
screen -dmS minio-port kubectl -n minio port-forward svc/minio 9001:9001