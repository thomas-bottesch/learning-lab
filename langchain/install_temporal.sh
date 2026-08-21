#!/usr/bin/env bash

set -euo pipefail

NAMESPACE="temporal"
TEMPORAL_RELEASE="temporal"
POSTGRES_RELEASE="temporal-postgresql"

PORT_FORWARD_DIR="${TMPDIR:-/tmp}/temporal-k8s"
TEMPORAL_PORT_FORWARD_PID="${PORT_FORWARD_DIR}/temporal.pid"
WEB_PORT_FORWARD_PID="${PORT_FORWARD_DIR}/web.pid"

TEMPORAL_PORT="${TEMPORAL_PORT:-7233}"
WEB_PORT="${WEB_PORT:-8081}"


usage() {
    cat <<EOF

Usage:

    $0 install
    $0 uninstall

Environment variables:

    TEMPORAL_PORT=7233
    WEB_PORT=8081

Examples:

    $0 install
    $0 uninstall

EOF
}


check_dependencies() {
    command -v kubectl >/dev/null 2>&1 || {
        echo "ERROR: kubectl is not installed."
        exit 1
    }

    command -v helm >/dev/null 2>&1 || {
        echo "ERROR: helm is not installed."
        exit 1
    }
}


create_namespace() {
    echo "==> Creating namespace: ${NAMESPACE}"

    kubectl create namespace "${NAMESPACE}" \
        --dry-run=client \
        -o yaml |
        kubectl apply -f -
}


install_postgres() {
    echo
    echo "==> Installing PostgreSQL"

    helm repo add bitnami \
        https://charts.bitnami.com/bitnami \
        >/dev/null 2>&1 || true

    helm repo update >/dev/null

    helm upgrade --install "${POSTGRES_RELEASE}" \
        bitnami/postgresql \
        --namespace "${NAMESPACE}" \
        --set auth.username=temporal \
        --set auth.password=temporal \
        --set auth.database=temporal \
        --set primary.persistence.enabled=true \
        --set primary.persistence.size=2Gi \
        --wait \
        --timeout 10m
}


wait_for_postgres() {
    echo
    echo "==> Waiting for PostgreSQL"

    kubectl wait \
        --namespace "${NAMESPACE}" \
        --for=condition=ready \
        pod \
        -l "app.kubernetes.io/instance=${POSTGRES_RELEASE}" \
        --timeout=180s
}


install_temporal() {
    echo
    echo "==> Adding Temporal Helm repository"

    helm repo add temporal \
        https://go.temporal.io/helm-charts \
        >/dev/null 2>&1 || true

    helm repo update >/dev/null

    echo
    echo "==> Creating Temporal values"

    local values_file

    values_file="$(mktemp)"

    cat > "${values_file}" <<EOF
server:
  config:
    persistence:
      defaultStore: default
      visibilityStore: visibility
      numHistoryShards: 4

      datastores:

        default:
          sql:
            pluginName: postgres12
            driverName: postgres12
            databaseName: temporal
            connectAddr: ${POSTGRES_RELEASE}:5432
            connectProtocol: tcp
            user: temporal
            password: temporal
            createDatabase: false
            manageSchema: true

        visibility:
          sql:
            pluginName: postgres12
            driverName: postgres12
            databaseName: temporal_visibility
            connectAddr: ${POSTGRES_RELEASE}:5432
            connectProtocol: tcp
            user: temporal
            password: temporal
            createDatabase: true
            manageSchema: true

web:
  enabled: true

admintools:
  enabled: true

schema:
  useHelmHooks: true
EOF

    echo
    echo "==> Installing Temporal"

    helm upgrade --install "${TEMPORAL_RELEASE}" \
        temporal/temporal \
        --namespace "${NAMESPACE}" \
        --values "${values_file}" \
        --wait \
        --timeout 10m

    rm -f "${values_file}"
}


wait_for_temporal() {
    echo
    echo "==> Waiting for Temporal components"

    kubectl rollout status \
        deployment/"${TEMPORAL_RELEASE}-frontend" \
        -n "${NAMESPACE}" \
        --timeout=180s

    kubectl rollout status \
        deployment/"${TEMPORAL_RELEASE}-history" \
        -n "${NAMESPACE}" \
        --timeout=180s

    kubectl rollout status \
        deployment/"${TEMPORAL_RELEASE}-matching" \
        -n "${NAMESPACE}" \
        --timeout=180s

    kubectl rollout status \
        deployment/"${TEMPORAL_RELEASE}-worker" \
        -n "${NAMESPACE}" \
        --timeout=180s

    kubectl rollout status \
        deployment/"${TEMPORAL_RELEASE}-web" \
        -n "${NAMESPACE}" \
        --timeout=180s
}


stop_port_forwards() {
    echo
    echo "==> Stopping port forwards"

    for pid_file in \
        "${TEMPORAL_PORT_FORWARD_PID}" \
        "${WEB_PORT_FORWARD_PID}"
    do
        if [[ -f "${pid_file}" ]]; then
            pid="$(cat "${pid_file}" 2>/dev/null || true)"

            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                echo "    Killing port-forward PID ${pid}"
                kill "${pid}" 2>/dev/null || true
            fi

            rm -f "${pid_file}"
        fi
    done

    # Defensive cleanup in case the PID file disappeared.
    pkill -f \
        "kubectl port-forward.*-n ${NAMESPACE}.*${TEMPORAL_RELEASE}-frontend" \
        2>/dev/null || true

    pkill -f \
        "kubectl port-forward.*-n ${NAMESPACE}.*${TEMPORAL_RELEASE}-web" \
        2>/dev/null || true
}

create_temporal_namespace() {
    echo
    echo "==> Creating Temporal namespace"

    kubectl exec \
        -n "${NAMESPACE}" \
        deployment/"${TEMPORAL_RELEASE}-admintools" \
        -- \
        temporal operator namespace create \
        default \
        2>/dev/null || true
}

start_port_forwards() {
    echo
    echo "==> Starting Temporal port forwards"

    mkdir -p "${PORT_FORWARD_DIR}"

    stop_port_forwards

    echo "    Temporal API: localhost:${TEMPORAL_PORT}"

    kubectl port-forward \
        -n "${NAMESPACE}" \
        "svc/${TEMPORAL_RELEASE}-frontend" \
        "${TEMPORAL_PORT}:7233" \
        >"${PORT_FORWARD_DIR}/temporal.log" 2>&1 &

    echo $! > "${TEMPORAL_PORT_FORWARD_PID}"


    echo "    Temporal UI:  http://localhost:${WEB_PORT}"

    kubectl port-forward \
        -n "${NAMESPACE}" \
        "svc/${TEMPORAL_RELEASE}-web" \
        "${WEB_PORT}:8080" \
        >"${PORT_FORWARD_DIR}/web.log" 2>&1 &

    echo $! > "${WEB_PORT_FORWARD_PID}"


    # Give kubectl a moment to establish the tunnels.
    sleep 2

    if ! kill -0 "$(cat "${TEMPORAL_PORT_FORWARD_PID}")" 2>/dev/null; then
        echo
        echo "ERROR: Temporal port-forward failed."
        cat "${PORT_FORWARD_DIR}/temporal.log"
        exit 1
    fi

    if ! kill -0 "$(cat "${WEB_PORT_FORWARD_PID}")" 2>/dev/null; then
        echo
        echo "ERROR: Temporal UI port-forward failed."
        cat "${PORT_FORWARD_DIR}/web.log"
        exit 1
    fi
}


show_status() {
    echo
    echo "=============================================="
    echo " Temporal is running"
    echo "=============================================="

    echo
    echo "Pods:"
    kubectl get pods -n "${NAMESPACE}"

    echo
    echo "Services:"
    kubectl get svc -n "${NAMESPACE}"

    echo
    echo "Temporal frontend:"
    echo "    localhost:${TEMPORAL_PORT}"

    echo
    echo "Temporal UI:"
    echo "    http://localhost:${WEB_PORT}"

    echo
    echo "Port-forward logs:"
    echo "    ${PORT_FORWARD_DIR}/temporal.log"
    echo "    ${PORT_FORWARD_DIR}/web.log"

    echo
}


install() {
    echo "=============================================="
    echo " Installing Temporal"
    echo "=============================================="

    check_dependencies

    kubectl cluster-info >/dev/null

    create_namespace
    install_postgres
    wait_for_postgres
    install_temporal
    wait_for_temporal
    create_temporal_namespace
    start_port_forwards
    show_status
}


uninstall() {
    echo "=============================================="
    echo " Uninstalling Temporal"
    echo "=============================================="

    check_dependencies

    stop_port_forwards

    if kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then

        echo
        echo "==> Uninstalling Temporal Helm release"

        helm uninstall \
            "${TEMPORAL_RELEASE}" \
            --namespace "${NAMESPACE}" \
            2>/dev/null || true


        echo
        echo "==> Uninstalling PostgreSQL Helm release"

        helm uninstall \
            "${POSTGRES_RELEASE}" \
            --namespace "${NAMESPACE}" \
            2>/dev/null || true


        echo
        echo "==> Deleting namespace"

        kubectl delete namespace "${NAMESPACE}" \
            --wait=true \
            --timeout=5m \
            2>/dev/null || true

    else
        echo "Namespace ${NAMESPACE} does not exist."
    fi


    echo
    echo "==> Removing local port-forward state"

    rm -rf "${PORT_FORWARD_DIR}"


    echo
    echo "=============================================="
    echo " Temporal completely removed"
    echo "=============================================="
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
        exit 1
        ;;
esac