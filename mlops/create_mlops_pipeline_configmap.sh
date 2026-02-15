
#!/usr/bin/env bash
set -euo pipefail

TARGET_NAMESPACE="${1:-user-example-com}"
CONFIGMAP_NAME="${2:-pipeline-configmap}"
LAKEFS_ENDPOINT="${LAKEFS_ENDPOINT:-http://lakefs.lakefs.svc.cluster.local:8000}"
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://minio.minio.svc.cluster.local:9000}"
LAKEFS_BUCKET_NAME="${LAKEFS_BUCKET_NAME:-lakefs-data}"
DVC_BUCKET_NAME="${DVC_BUCKET_NAME:-dvc-data}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command not found: $1" >&2
    exit 1
  }
}

require_cmd kubectl

kubectl get namespace "$TARGET_NAMESPACE" >/dev/null 2>&1 || {
  echo "Creating namespace: $TARGET_NAMESPACE"
  kubectl create namespace "$TARGET_NAMESPACE" >/dev/null
}

kubectl -n "$TARGET_NAMESPACE" create configmap "$CONFIGMAP_NAME" \
  --from-literal=LAKEFS_ENDPOINT="$LAKEFS_ENDPOINT" \
  --from-literal=MINIO_ENDPOINT="$MINIO_ENDPOINT" \
  --from-literal=LAKEFS_BUCKET_NAME="$LAKEFS_BUCKET_NAME" \
  --from-literal=DVC_BUCKET_NAME="$DVC_BUCKET_NAME" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

cat <<EOF
Created/updated ConfigMap '$CONFIGMAP_NAME' in namespace '$TARGET_NAMESPACE'.
Data keys:
  - LAKEFS_ENDPOINT=$LAKEFS_ENDPOINT
  - MINIO_ENDPOINT=$MINIO_ENDPOINT
  - LAKEFS_BUCKET_NAME=$LAKEFS_BUCKET_NAME
  - DVC_BUCKET_NAME=$DVC_BUCKET_NAME

Verify:
  kubectl -n $TARGET_NAMESPACE get configmap $CONFIGMAP_NAME -o yaml
EOF
