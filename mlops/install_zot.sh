#!/bin/bash
# install_zot.sh: Apply all Zot-related Kubernetes YAMLs

set -e

YAML_DIR="k8s_yamls/zot"

for yaml in "$YAML_DIR"/*.yaml; do
    echo "Applying $yaml..."
    kubectl apply -f "$yaml"
done

echo "All Zot YAMLs applied successfully."

# Wait for Zot deployment to be ready
echo "Waiting for Zot deployment to be ready..."
kubectl rollout status deployment/zot -n zot --timeout=5m

# Wait for LoadBalancer service to be ready
echo "Waiting for Zot LoadBalancer service to be ready..."
while ! kubectl -n zot get svc zot -o jsonpath='{.status.loadBalancer.ingress[0].ip}' >/dev/null 2>&1; do
    echo "Waiting for LoadBalancer IP..."
    sleep 2
done

LB_IP=$(kubectl -n zot get svc zot -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Zot LoadBalancer is ready at IP: $LB_IP"

# Create ml-components repository using OCI artifact
echo "Creating ml-components repository in Zot..."

# Create a dummy config file for ML components
cat > /tmp/ml-config.json <<'EOF'
{
  "name": "ml-components",
  "description": "ML Components repository for Kubeflow Pipelines",
  "version": "1.0.0"
}
EOF

# Push OCI artifact to create the repository structure
REGISTRY="localhost:8001"
oras push "$REGISTRY/ml-components/config:latest" \
  --config /tmp/ml-config.json:application/vnd.ml.config.v1+json

# Clean up
rm -f /tmp/ml-config.json

echo "ml-components repository created successfully in Zot."

# Zot is now accessible via LoadBalancer:
# - Registry: http://localhost:8001
# - v2 API: http://localhost:8001/v2/
