#!/usr/bin/env bash
set -euo pipefail

TARGET_NAMESPACE="${1:-user-example-com}"
TARGET_SECRET_NAME="${2:-mlops-credentials}"
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

# Pull existing values from source secrets.
lakefs_access_key="$(get_secret_key_plaintext "$LAKEFS_NAMESPACE" "$LAKEFS_SECRET_NAME" "LAKEFS_AUTH_ADMIN_ACCESS_KEY_ID")"
lakefs_secret_key="$(get_secret_key_plaintext "$LAKEFS_NAMESPACE" "$LAKEFS_SECRET_NAME" "LAKEFS_AUTH_ADMIN_SECRET_ACCESS_KEY")"
minio_access_key="$(get_secret_key_plaintext "$MINIO_NAMESPACE" "$MINIO_SECRET_NAME" "MINIO_ROOT_USER")"
minio_secret_key="$(get_secret_key_plaintext "$MINIO_NAMESPACE" "$MINIO_SECRET_NAME" "MINIO_ROOT_PASSWORD")"

# Create/update the pipeline secret with the key names expected by the Kubeflow pipeline.
kubectl -n "$TARGET_NAMESPACE" create secret generic "$TARGET_SECRET_NAME" \
  --from-literal=LAKEFS_ACCESS_KEY="$lakefs_access_key" \
  --from-literal=LAKEFS_SECRET_KEY="$lakefs_secret_key" \
  --from-literal=MINIO_ACCESS_KEY="$minio_access_key" \
  --from-literal=MINIO_SECRET_KEY="$minio_secret_key" \
  --from-literal=AWS_ACCESS_KEY_ID="$minio_access_key" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$minio_secret_key" \
  --dry-run=client -o yaml | kubectl apply -f - >/dev/null

cat <<EOF
Created/updated secret '$TARGET_SECRET_NAME' in namespace '$TARGET_NAMESPACE'.
Keys:
  - LAKEFS_ACCESS_KEY
  - LAKEFS_SECRET_KEY
  - MINIO_ACCESS_KEY
  - MINIO_SECRET_KEY
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY

Verify:
  kubectl -n $TARGET_NAMESPACE get secret $TARGET_SECRET_NAME -o jsonpath='{.data}'
EOF

# Apply the PodDefault that injects the secret and configmap into all KFP pipeline pods.
# Without this, every pipeline pod crashes with KeyError on the first os.environ[] access.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kubectl apply -f "${SCRIPT_DIR}/k8s_yamls/kubeflow/mlops-poddefault.yaml"
echo "✓ PodDefault 'mlops-credentials' applied — all pipeline pods will receive service credentials."
