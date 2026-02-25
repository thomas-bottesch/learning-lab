#!/usr/bin/env bash
set -euo pipefail

# Script to set up Drone ConfigMap for MLOps service endpoints
# This creates a Kubernetes ConfigMap that will be mounted to the Drone runner

DRONE_NAMESPACE="${DRONE_NAMESPACE:-drone}"
CONFIGMAP_NAME="${CONFIGMAP_NAME:-drone-config}"

# MLOps service endpoints
MLFLOW_ENDPOINT="${MLFLOW_ENDPOINT:-http://mlflow.mlflow.svc.cluster.local:5000}"
KUBEFLOW_ENDPOINT="${KUBEFLOW_ENDPOINT:-http://istio-ingressgateway.istio-system.svc.cluster.local}"
LAKEFS_ENDPOINT="${LAKEFS_ENDPOINT:-http://lakefs.lakefs.svc.cluster.local:8000}"
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://minio.minio.svc.cluster.local:9000}"

# Additional configuration
LAKEFS_BUCKET_NAME="${LAKEFS_BUCKET_NAME:-lakefs-data}"
DVC_BUCKET_NAME="${DVC_BUCKET_NAME:-dvc-data}"
MODEL_BUCKET="${MODEL_BUCKET:-models}"
DATA_BUCKET="${DATA_BUCKET:-datasets}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command not found: $1" >&2
    exit 1
  }
}

require_cmd kubectl

echo "Setting up Drone MLOps ConfigMap..."
echo "Namespace: $DRONE_NAMESPACE"
echo "ConfigMap: $CONFIGMAP_NAME"
echo ""

# Create/update the Drone MLOps configmap
echo "Creating/updating Drone MLOps ConfigMap..."
kubectl -n "$DRONE_NAMESPACE" create configmap "$CONFIGMAP_NAME" \
  --from-literal=MLFLOW_ENDPOINT="$MLFLOW_ENDPOINT" \
  --from-literal=KUBEFLOW_ENDPOINT="$KUBEFLOW_ENDPOINT" \
  --from-literal=LAKEFS_ENDPOINT="$LAKEFS_ENDPOINT" \
  --from-literal=MINIO_ENDPOINT="$MINIO_ENDPOINT" \
  --from-literal=LAKEFS_BUCKET_NAME="$LAKEFS_BUCKET_NAME" \
  --from-literal=DVC_BUCKET_NAME="$DVC_BUCKET_NAME" \
  --from-literal=MODEL_BUCKET="$MODEL_BUCKET" \
  --from-literal=DATA_BUCKET="$DATA_BUCKET" \
  --from-literal=MLFLOW_TRACKING_URI="$MLFLOW_ENDPOINT" \
  --from-literal=MLFLOW_S3_ENDPOINT_URL="$MINIO_ENDPOINT" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

cat <<EOF
✅ Drone MLOps ConfigMap configured

ConfigMap: $CONFIGMAP_NAME in namespace '$DRONE_NAMESPACE'
Data keys:
  - MLFLOW_ENDPOINT=$MLFLOW_ENDPOINT
  - KUBEFLOW_ENDPOINT=$KUBEFLOW_ENDPOINT
  - LAKEFS_ENDPOINT=$LAKEFS_ENDPOINT
  - MINIO_ENDPOINT=$MINIO_ENDPOINT
  - LAKEFS_BUCKET_NAME=$LAKEFS_BUCKET_NAME
  - DVC_BUCKET_NAME=$DVC_BUCKET_NAME
  - MODEL_BUCKET=$MODEL_BUCKET
  - DATA_BUCKET=$DATA_BUCKET
  - MLFLOW_TRACKING_URI=$MLFLOW_ENDPOINT
  - MLFLOW_S3_ENDPOINT_URL=$MINIO_ENDPOINT

Verify with:
  kubectl -n $DRONE_NAMESPACE get configmap $CONFIGMAP_NAME -o yaml
EOF