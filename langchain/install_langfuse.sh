#!/usr/bin/env bash

set -euo pipefail

NAMESPACE="langfuse"
RELEASE="langfuse"
REPO_NAME="langfuse"
REPO_URL="https://langfuse.github.io/langfuse-k8s"

LOCAL_PORT="3000"
SERVICE_PORT="3000"

ENV_FILE=".langfuse.env"

# ---------------------------------------------------------------------------
# Local Langfuse user
# ---------------------------------------------------------------------------

LANGFUSE_USER_EMAIL="local@example.com"
LANGFUSE_USER_NAME="Local User"
LANGFUSE_USER_PASSWORD="locallocal"

# ---------------------------------------------------------------------------
# Local Langfuse organization/project
# ---------------------------------------------------------------------------

LANGFUSE_ORG_ID="local-org"
LANGFUSE_ORG_NAME="Local Organization"

LANGFUSE_PROJECT_ID="local-project"
LANGFUSE_PROJECT_NAME="Local Project"

usage() {
    echo "Usage: $0 {install|uninstall}"
    echo
    echo "Commands:"
    echo "  install      Install or upgrade Langfuse"
    echo "  uninstall    Uninstall Langfuse and delete its namespace"
    exit 1
}

install() {
    local values_file
    values_file="$(mktemp)"

    trap 'rm -f "$values_file"' EXIT

    echo "==> Adding Langfuse Helm repository..."
    helm repo add "$REPO_NAME" "$REPO_URL" 2>/dev/null || true
    helm repo update

    echo "==> Creating namespace: $NAMESPACE..."
    kubectl create namespace "$NAMESPACE" 2>/dev/null || true

    echo "==> Generating infrastructure secrets..."

    # Hex-only secrets are safe to use inside database connection URLs.
    SALT="$(openssl rand -hex 32)"
    NEXTAUTH_SECRET="$(openssl rand -hex 32)"
    ENCRYPTION_KEY="$(openssl rand -hex 32)"

    POSTGRES_PASSWORD="$(openssl rand -hex 32)"
    CLICKHOUSE_PASSWORD="$(openssl rand -hex 32)"
    REDIS_PASSWORD="$(openssl rand -hex 32)"
    S3_PASSWORD="$(openssl rand -hex 32)"

    echo "==> Generating Langfuse API keys..."

    LANGFUSE_PUBLIC_KEY="lf_pk_$(openssl rand -hex 16)"
    LANGFUSE_SECRET_KEY="lf_sk_$(openssl rand -hex 32)"

    cat > "$values_file" <<EOF
langfuse:
  salt:
    value: "${SALT}"

  nextauth:
    secret:
      value: "${NEXTAUTH_SECRET}"

  encryptionKey:
    value: "${ENCRYPTION_KEY}"

  additionalEnv:
    - name: LANGFUSE_INIT_ORG_ID
      value: "${LANGFUSE_ORG_ID}"

    - name: LANGFUSE_INIT_ORG_NAME
      value: "${LANGFUSE_ORG_NAME}"

    - name: LANGFUSE_INIT_PROJECT_ID
      value: "${LANGFUSE_PROJECT_ID}"

    - name: LANGFUSE_INIT_PROJECT_NAME
      value: "${LANGFUSE_PROJECT_NAME}"

    - name: LANGFUSE_INIT_PROJECT_PUBLIC_KEY
      value: "${LANGFUSE_PUBLIC_KEY}"

    - name: LANGFUSE_INIT_PROJECT_SECRET_KEY
      value: "${LANGFUSE_SECRET_KEY}"

    - name: LANGFUSE_INIT_USER_EMAIL
      value: "${LANGFUSE_USER_EMAIL}"

    - name: LANGFUSE_INIT_USER_NAME
      value: "${LANGFUSE_USER_NAME}"

    - name: LANGFUSE_INIT_USER_PASSWORD
      value: "${LANGFUSE_USER_PASSWORD}"

postgresql:
  auth:
    username: langfuse
    password: "${POSTGRES_PASSWORD}"

clickhouse:
  auth:
    password: "${CLICKHOUSE_PASSWORD}"

redis:
  auth:
    password: "${REDIS_PASSWORD}"

s3:
  auth:
    rootPassword: "${S3_PASSWORD}"
EOF

    echo "==> Installing/upgrading Langfuse..."

    helm upgrade --install "$RELEASE" "$REPO_NAME/langfuse" \
        --namespace "$NAMESPACE" \
        --values "$values_file"

    echo
    echo "==> Waiting for PostgreSQL..."

    kubectl wait \
        --namespace "$NAMESPACE" \
        --for=condition=ready \
        pod \
        -l app.kubernetes.io/name=postgresql \
        --timeout=10m

    echo "==> PostgreSQL is ready."

    echo
    echo "==> Waiting for Langfuse web..."

    kubectl rollout status \
        deployment/langfuse-web \
        --namespace "$NAMESPACE" \
        --timeout=10m

    echo "==> Langfuse web is ready."

    echo
    echo "==> Waiting for Langfuse worker..."

    kubectl rollout status \
        deployment/langfuse-worker \
        --namespace "$NAMESPACE" \
        --timeout=10m

    echo "==> Langfuse worker is ready."

    echo
    echo "==> Writing credentials to ${ENV_FILE}..."

    cat > "$ENV_FILE" <<EOF
# Langfuse local development environment

# Web UI
LANGFUSE_HOST=http://localhost:${LOCAL_PORT}
LANGFUSE_BASE_URL=http://localhost:${LOCAL_PORT}

# Browser login
LANGFUSE_USER_EMAIL=${LANGFUSE_USER_EMAIL}
LANGFUSE_USER_PASSWORD=${LANGFUSE_USER_PASSWORD}

# Langfuse project API credentials
LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY}
LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}

# Langfuse project
LANGFUSE_ORG_ID=${LANGFUSE_ORG_ID}
LANGFUSE_PROJECT_ID=${LANGFUSE_PROJECT_ID}
EOF

    chmod 600 "$ENV_FILE"

    echo
    echo "=========================================="
    echo " Langfuse is ready!"
    echo "=========================================="
    echo
    echo " Web UI:"
    echo "   http://localhost:${LOCAL_PORT}"
    echo
    echo " Browser login:"
    echo "   Email:    ${LANGFUSE_USER_EMAIL}"
    echo "   Password: ${LANGFUSE_USER_PASSWORD}"
    echo
    echo " API credentials:"
    echo "   Public:   ${LANGFUSE_PUBLIC_KEY}"
    echo "   Secret:   ${LANGFUSE_SECRET_KEY}"
    echo
    echo " Credentials saved to:"
    echo "   ${ENV_FILE}"
    echo
    echo " Load them with:"
    echo "   source ${ENV_FILE}"
    echo
    echo " Press Ctrl+C to stop the port-forward."
    echo

  kubectl port-forward \
      --namespace "$NAMESPACE" \
      "svc/langfuse-web" \
      "${LOCAL_PORT}:${SERVICE_PORT}" \
      > /tmp/langfuse-port-forward.log 2>&1 &
}

uninstall() {
    echo "==> Uninstalling Langfuse..."

    helm uninstall "$RELEASE" \
        --namespace "$NAMESPACE" \
        2>/dev/null || true

    echo "==> Deleting namespace: $NAMESPACE..."

    kubectl delete namespace "$NAMESPACE" \
        --ignore-not-found=true \
        --wait=true

    echo "==> Removing local credentials..."

    rm -f "$ENV_FILE"

    echo
    echo "=========================================="
    echo " Langfuse has been uninstalled."
    echo "=========================================="
}

case "${1:-}" in
    install)
        install
        ;;

    uninstall)
        uninstall
        ;;

    *)
        usage
        ;;
esac