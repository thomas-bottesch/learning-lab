#!/bin/bash
# install_mlflow.sh: Install MLflow with PostgreSQL backend and MinIO artifact storage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

YAML_DIR="$SCRIPT_DIR/k8s_yamls/mlflow"
MLFLOW_URL="http://localhost:5000"

echo "=========================================="
echo "Installing MLflow for MLOps pipeline"
echo "=========================================="

# Apply namespace first
echo "1. Applying MLflow namespace..."
kubectl apply -f "$YAML_DIR/01-namespace.yaml"

# Apply remaining MLflow YAMLs (excluding namespace)
echo "4. Applying remaining MLflow Kubernetes manifests..."
for yaml in "$YAML_DIR"/*.yaml; do
    # Skip the namespace file
    if [[ "$(basename "$yaml")" == "01-namespace.yaml" ]]; then
        continue
    fi
    echo "  Applying $yaml..."
    kubectl apply -f "$yaml"
done

# Wait for bucket creation job
echo "2. Waiting for MinIO bucket creation..."
kubectl wait --for=condition=complete job/mlflow-create-bucket -n mlflow --timeout=2m || true

# Wait for MLflow deployment to be ready
echo "3. Waiting for MLflow deployment to be ready..."
kubectl rollout status deployment/mlflow -n mlflow --timeout=3m

# Wait for the LoadBalancer service to be ready
echo "Waiting for LoadBalancer service to be ready..."
while ! kubectl -n mlflow get svc mlflow -o jsonpath='{.status.loadBalancer.ingress[0].ip}' >/dev/null 2>&1; do
    echo "Waiting for LoadBalancer IP..."
    sleep 2
done

# Wait for MLflow web server
echo "6. Waiting for MLflow web server to be available..."
for i in {1..30}; do
    if curl -s --head --fail "$MLFLOW_URL" > /dev/null; then
        echo "MLflow web server is up."
        break
    else
        echo "MLflow web server not ready yet, retrying ($i)..."
        sleep 2
    fi
done

echo ""
echo "=========================================="
echo "MLflow Installation Complete!"
echo "=========================================="
echo "MLflow UI: $MLFLOW_URL"
echo ""
echo "MLflow is now configured with:"
echo "✅ SQLite backend database (simplified for local learning)"
echo "✅ MinIO artifact storage (bucket: mlflow-artifacts)"
echo "✅ Persistent storage for metadata"
echo ""
echo "Integration with your MLOps stack:"
echo "• Experiments tracked in SQLite"
echo "• Model artifacts stored in MinIO"
echo "• Can be accessed from Kubeflow pipelines"
echo "• Integrates with existing LakeFS/MinIO setup"
echo ""
echo "To use MLflow in your pipelines:"
echo "1. Set environment variable: MLFLOW_TRACKING_URI=http://mlflow.mlflow.svc.cluster.local:5000"
echo "2. Install mlflow package in your pipeline containers"
echo "3. Use mlflow.log_* functions to track experiments"
echo ""
echo "Example Python code:"
echo "  import mlflow"
echo "  mlflow.set_tracking_uri('http://mlflow.mlflow.svc.cluster.local:5000')"
echo "  mlflow.set_experiment('my-experiment')"
echo "  with mlflow.start_run():"
echo "      mlflow.log_param('learning_rate', 0.01)"
echo "      mlflow.log_metric('accuracy', 0.95)"
echo "      mlflow.log_artifact('model.pkl')"
echo "=========================================="