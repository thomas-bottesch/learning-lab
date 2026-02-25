#!/usr/bin/env bash
set -euo pipefail

# Script to set up Drone secrets for MLOps services
# This creates Kubernetes secrets that will be mounted to the Drone runner

DRONE_NAMESPACE="${DRONE_NAMESPACE:-drone}"
SECRET_NAME="${SECRET_NAME:-drone-additional-credentials}"

# Source namespaces for existing secrets
LAKEFS_NAMESPACE="${LAKEFS_NAMESPACE:-lakefs}"
LAKEFS_SECRET_NAME="${LAKEFS_SECRET_NAME:-lakefs-credentials}"
MINIO_NAMESPACE="${MINIO_NAMESPACE:-minio}"
MINIO_SECRET_NAME="${MINIO_SECRET_NAME:-minio-root}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command not found: $1" >&2
    exit 1
  }
}

get_secret_key_plaintext() {
  local namespace="$1"
  local secret="$2"
  local key="$3"

  local encoded
  encoded="$(kubectl -n "$namespace" get secret "$secret" -o "jsonpath={.data.${key}}" 2>/dev/null || true)"

  if [[ -z "$encoded" ]]; then
    echo "Error: key '$key' not found in secret '$secret' (namespace '$namespace')." >&2
    exit 1
  fi

  echo "$encoded" | base64 -d
}

require_cmd kubectl
require_cmd base64

echo "Setting up Drone MLOps secrets..."
echo "Namespace: $DRONE_NAMESPACE"
echo "Secret: $SECRET_NAME"
echo ""

# Pull existing credentials from source secrets
echo "Fetching credentials from existing secrets..."
lakefs_access_key="$(get_secret_key_plaintext "$LAKEFS_NAMESPACE" "$LAKEFS_SECRET_NAME" "LAKEFS_AUTH_ADMIN_ACCESS_KEY_ID")"
lakefs_secret_key="$(get_secret_key_plaintext "$LAKEFS_NAMESPACE" "$LAKEFS_SECRET_NAME" "LAKEFS_AUTH_ADMIN_SECRET_ACCESS_KEY")"
minio_access_key="$(get_secret_key_plaintext "$MINIO_NAMESPACE" "$MINIO_SECRET_NAME" "MINIO_ROOT_USER")"
minio_secret_key="$(get_secret_key_plaintext "$MINIO_NAMESPACE" "$MINIO_SECRET_NAME" "MINIO_ROOT_PASSWORD")"

# Create/update the Drone MLOps credentials secret
echo "Creating/updating Drone MLOps credentials secret..."
kubectl -n "$DRONE_NAMESPACE" create secret generic "$SECRET_NAME" \
  --from-literal=LAKEFS_ACCESS_KEY="$lakefs_access_key" \
  --from-literal=LAKEFS_SECRET_KEY="$lakefs_secret_key" \
  --from-literal=MINIO_ACCESS_KEY="$minio_access_key" \
  --from-literal=MINIO_SECRET_KEY="$minio_secret_key" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

cat <<EOF
✅ Drone MLOps credentials secret configured

Secret: $SECRET_NAME in namespace '$DRONE_NAMESPACE'
Keys:
  - LAKEFS_ACCESS_KEY
  - LAKEFS_SECRET_KEY
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY

Verify with:
  kubectl -n $DRONE_NAMESPACE get secret $SECRET_NAME -o jsonpath='{.data}'
EOF