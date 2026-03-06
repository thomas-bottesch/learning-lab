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

# MinIO is now accessible via LoadBalancer:
# - API: http://localhost:9000
# - Console: http://localhost:9001
