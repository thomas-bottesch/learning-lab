#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

NAMESPACE="langfuse"
RELEASE="langfuse"

REPO_NAME="langfuse"
REPO_URL="https://langfuse.github.io/langfuse-k8s"

LOCAL_PORT="3000"
SERVICE_PORT="3000"

ENV_FILE=".langfuse.env"

# -----------------------------------------------------------------------------
# Cluster-wide prerequisites
#
# These are installed once per Kubernetes cluster.
# -----------------------------------------------------------------------------

CERT_MANAGER_RELEASE="cert-manager"
CERT_MANAGER_NAMESPACE="cert-manager"
CERT_MANAGER_VERSION="v1.20.2"

CLICKHOUSE_OPERATOR_RELEASE="clickhouse-operator"
CLICKHOUSE_OPERATOR_NAMESPACE="clickhouse-operator"
CLICKHOUSE_OPERATOR_VERSION="0.0.5"

# -----------------------------------------------------------------------------
# Local Langfuse user
# -----------------------------------------------------------------------------

LANGFUSE_USER_EMAIL="local@example.com"
LANGFUSE_USER_NAME="Local User"
LANGFUSE_USER_PASSWORD="locallocal"

# -----------------------------------------------------------------------------
# Local Langfuse organization/project
# -----------------------------------------------------------------------------

LANGFUSE_ORG_ID="local-org"
LANGFUSE_ORG_NAME="Local Organization"

LANGFUSE_PROJECT_ID="local-project"
LANGFUSE_PROJECT_NAME="Local Project"

# =============================================================================
# Helpers
# =============================================================================

usage() {
    echo "Usage: $0 {install|uninstall}"
    echo
    echo "Commands:"
    echo "  install      Install or upgrade Langfuse"
    echo "  uninstall    Uninstall Langfuse and delete its namespace"
    exit 1
}

require_command() {
    local command_name="$1"

    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: Required command not found: $command_name" >&2
        exit 1
    fi
}

check_prerequisites() {
    echo "==> Checking local prerequisites..."

    require_command kubectl
    require_command helm
    require_command openssl

    if ! kubectl cluster-info >/dev/null 2>&1; then
        echo "ERROR: kubectl cannot connect to a Kubernetes cluster." >&2
        echo "       Check your kubeconfig/context." >&2
        exit 1
    fi

    echo "    kubectl:  OK"
    echo "    helm:     OK"
    echo "    openssl:  OK"
    echo "    cluster:  OK"
}

# =============================================================================
# cert-manager
# =============================================================================

install_cert_manager() {
    echo
    echo "==> Checking cert-manager..."

    if helm status "$CERT_MANAGER_RELEASE" \
        --namespace "$CERT_MANAGER_NAMESPACE" >/dev/null 2>&1; then

        echo "    cert-manager Helm release already exists."

    else
        echo "==> Installing cert-manager ${CERT_MANAGER_VERSION}..."

        helm install "$CERT_MANAGER_RELEASE" \
            oci://quay.io/jetstack/charts/cert-manager \
            --version "$CERT_MANAGER_VERSION" \
            --namespace "$CERT_MANAGER_NAMESPACE" \
            --create-namespace \
            --set crds.enabled=true
    fi

    echo "==> Waiting for cert-manager..."

    kubectl rollout status \
        deployment/cert-manager \
        --namespace "$CERT_MANAGER_NAMESPACE" \
        --timeout=10m

    kubectl rollout status \
        deployment/cert-manager-webhook \
        --namespace "$CERT_MANAGER_NAMESPACE" \
        --timeout=10m

    kubectl rollout status \
        deployment/cert-manager-cainjector \
        --namespace "$CERT_MANAGER_NAMESPACE" \
        --timeout=10m

    echo "==> cert-manager is ready."
}

# =============================================================================
# ClickHouse Operator
# =============================================================================

install_clickhouse_operator() {
    echo
    echo "==> Checking ClickHouse Operator..."

    if helm status "$CLICKHOUSE_OPERATOR_RELEASE" \
        --namespace "$CLICKHOUSE_OPERATOR_NAMESPACE" >/dev/null 2>&1; then

        echo "    ClickHouse Operator Helm release already exists."

    else
        echo "==> Installing ClickHouse Operator ${CLICKHOUSE_OPERATOR_VERSION}..."

        helm install "$CLICKHOUSE_OPERATOR_RELEASE" \
            oci://ghcr.io/clickhouse/clickhouse-operator-helm \
            --version "$CLICKHOUSE_OPERATOR_VERSION" \
            --namespace "$CLICKHOUSE_OPERATOR_NAMESPACE" \
            --create-namespace
    fi

    echo "==> Waiting for ClickHouse CRDs..."

    kubectl wait \
        --for=condition=Established \
        crd/clickhouseclusters.clickhouse.com \
        --timeout=5m

    kubectl wait \
        --for=condition=Established \
        crd/keeperclusters.clickhouse.com \
        --timeout=5m

    echo "==> Waiting for ClickHouse Operator..."

    # The exact deployment name can vary slightly between operator releases.
    # Wait for the namespace to have a ready deployment if one exists.
    local operator_deployment

    operator_deployment="$(
        kubectl get deployments \
            --namespace "$CLICKHOUSE_OPERATOR_NAMESPACE" \
            -o jsonpath='{.items[0].metadata.name}' \
            2>/dev/null || true
    )"

    if [[ -n "$operator_deployment" ]]; then
        kubectl rollout status \
            "deployment/$operator_deployment" \
            --namespace "$CLICKHOUSE_OPERATOR_NAMESPACE" \
            --timeout=10m
    fi

    echo "==> ClickHouse Operator is ready."
}

# =============================================================================
# Cluster prerequisites
# =============================================================================

install_cluster_prerequisites() {
    install_cert_manager
    install_clickhouse_operator
}

# =============================================================================
# Langfuse
# =============================================================================

install() {
    local values_file
    local port_forward_pid=""

    values_file="$(mktemp)"

    # IMPORTANT:
    # values_file is local to this function, so a normal EXIT trap referring
    # to "$values_file" would fail after the function exits under "set -u".
    #
    # The parameter expansion makes cleanup safe even if the variable is gone.
    cleanup() {
        if [[ -n "${port_forward_pid:-}" ]]; then
            echo
            echo "==> Stopping Langfuse port-forward..."
            kill "$port_forward_pid" 2>/dev/null || true
        fi

        rm -f "${values_file:-}"
    }

    trap cleanup INT TERM

    check_prerequisites

    echo
    echo "==> Installing cluster-wide prerequisites..."
    install_cluster_prerequisites

    echo
    echo "==> Adding Langfuse Helm repository..."

    helm repo add "$REPO_NAME" "$REPO_URL" 2>/dev/null || true
    helm repo update

    echo
    echo "==> Creating namespace: $NAMESPACE..."

    kubectl create namespace "$NAMESPACE" 2>/dev/null || true

    # =========================================================================
    # Generate secrets
    # =========================================================================

    echo
    echo "==> Generating infrastructure secrets..."

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

    # =========================================================================
    # Helm values
    #
    # Current Langfuse chart:
    #
    # - PostgreSQL is bundled by the chart.
    # - Redis/Valkey is bundled by the chart.
    # - S3-compatible object storage is bundled by the chart.
    # - ClickHouse is deployed through the ClickHouse Operator.
    #
    # The ClickHouse Operator and cert-manager were installed above.
    # =========================================================================

    echo
    echo "==> Creating temporary Helm values..."

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

    # =========================================================================
    # Install / upgrade Langfuse
    # =========================================================================

    echo
    echo "==> Installing/upgrading Langfuse..."

    helm upgrade --install "$RELEASE" "$REPO_NAME/langfuse" \
        --namespace "$NAMESPACE" \
        --values "$values_file" \
        --wait \
        --timeout=15m

    # =========================================================================
    # Wait for infrastructure
    # =========================================================================

    echo
    echo "==> Waiting for Langfuse PostgreSQL..."

    if kubectl get pods \
        --namespace "$NAMESPACE" \
        -l app.kubernetes.io/name=postgresql \
        >/dev/null 2>&1; then

        kubectl wait \
            --namespace "$NAMESPACE" \
            --for=condition=ready \
            pod \
            -l app.kubernetes.io/name=postgresql \
            --timeout=10m
    else
        echo "    PostgreSQL pod selector not found; continuing."
    fi

    echo "==> PostgreSQL is ready."

    # =========================================================================
    # Wait for ClickHouse
    # =========================================================================

    echo
    echo "==> Waiting for ClickHouse..."

    if kubectl get clickhousecluster \
        --namespace "$NAMESPACE" \
        "$RELEASE" >/dev/null 2>&1; then

        kubectl wait \
            --namespace "$NAMESPACE" \
            --for=condition=Ready \
            "clickhousecluster/$RELEASE" \
            --timeout=15m

        echo "==> ClickHouse is ready."

    else
        echo "WARNING: ClickHouseCluster '$RELEASE' was not found yet."

        echo "==> Current ClickHouse resources:"
        kubectl get clickhousecluster \
            --namespace "$NAMESPACE" \
            2>/dev/null || true

        echo "==> Current ClickHouse pods:"
        kubectl get pods \
            --namespace "$NAMESPACE" \
            -l "clickhouse.altinity.com/chi=$RELEASE" \
            2>/dev/null || true
    fi

    # =========================================================================
    # Wait for Langfuse web
    # =========================================================================

    echo
    echo "==> Waiting for Langfuse web..."

    kubectl rollout status \
        deployment/langfuse-web \
        --namespace "$NAMESPACE" \
        --timeout=15m

    echo "==> Langfuse web is ready."

    # =========================================================================
    # Wait for Langfuse worker
    # =========================================================================

    echo
    echo "==> Waiting for Langfuse worker..."

    kubectl rollout status \
        deployment/langfuse-worker \
        --namespace "$NAMESPACE" \
        --timeout=15m

    echo "==> Langfuse worker is ready."

    # =========================================================================
    # Save credentials
    # =========================================================================

    echo
    echo "==> Writing credentials to ${ENV_FILE}..."

    cat > "$ENV_FILE" <<EOF
# Langfuse local development environment
#
# Generated by install_langfuse.sh
# DO NOT COMMIT THIS FILE.

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

    # =========================================================================
    # Start port-forward (background, detached from script lifecycle)
    # =========================================================================

    echo
    echo "==> Starting Langfuse port-forward..."

    nohup kubectl port-forward \
        --namespace "$NAMESPACE" \
        "svc/langfuse-web" \
        "${LOCAL_PORT}:${SERVICE_PORT}" \
        > /tmp/langfuse-port-forward.log 2>&1 &

    port_forward_pid="$!"

    # Disown so the process survives after the script exits
    disown "$port_forward_pid" 2>/dev/null || true

    # Give kubectl a moment to establish the connection and catch immediate
    # failures such as "address already in use".
    sleep 2

    if ! kill -0 "$port_forward_pid" 2>/dev/null; then
        echo
        echo "ERROR: Langfuse port-forward failed." >&2
        echo
        echo "Port-forward log:" >&2
        cat /tmp/langfuse-port-forward.log >&2 || true
        exit 1
    fi

    # =========================================================================
    # Finished — exit now, port-forward continues in background
    # =========================================================================

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
    echo " Port-forward PID:"
    echo "   ${port_forward_pid}"
    echo
    echo " To stop Langfuse port-forward, run:"
    echo "   kill ${port_forward_pid}"
    echo
}

# =============================================================================
# Uninstall
# =============================================================================

uninstall() {
    echo "==> Uninstalling Langfuse..."

    helm uninstall "$RELEASE" \
        --namespace "$NAMESPACE" \
        2>/dev/null || true

    echo
    echo "==> Deleting namespace: $NAMESPACE..."

    kubectl delete namespace "$NAMESPACE" \
        --ignore-not-found=true \
        --wait=true

    echo
    echo "==> Removing local credentials..."

    rm -f "$ENV_FILE"

    echo
    echo "=========================================="
    echo " Langfuse has been uninstalled."
    echo "=========================================="
    echo
    echo "NOTE:"
    echo "  cert-manager and the ClickHouse Operator"
    echo "  were intentionally left installed because"
    echo "  they are cluster-wide prerequisites."
    echo
}

# =============================================================================
# Main
# =============================================================================

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