#!/usr/bin/env bash
# Print export statements for all credentials and endpoints needed locally.
# Reads from the same Kubernetes YAML files used by install_forgejo.sh
# and create_forgejo_repo.sh — no secrets ever live in this file.
#
# Usage (from /workspace/git_repo/mlops/):
#   eval "$(bash repos/team-tron/pipelines/scripts/get_local_env.sh)"
set -euo pipefail

MLOPS_ROOT="${MLOPS_ROOT:-$(pwd)}"

_yaml_field() {
    grep "${1}:" "${MLOPS_ROOT}/k8s_yamls/${2}" | awk -F': ' '{print $2}' | tr -d '"'
}

MINIO_ACCESS_KEY=$(_yaml_field 'MINIO_ROOT_USER'                      'minio/02-secret.yaml')
MINIO_SECRET_KEY=$(_yaml_field 'MINIO_ROOT_PASSWORD'                  'minio/02-secret.yaml')
LAKEFS_ACCESS_KEY=$(_yaml_field 'LAKEFS_AUTH_ADMIN_ACCESS_KEY_ID'     'lakefs/02-secret.yaml')
LAKEFS_SECRET_KEY=$(_yaml_field 'LAKEFS_AUTH_ADMIN_SECRET_ACCESS_KEY' 'lakefs/02-secret.yaml')

printf 'export MINIO_ACCESS_KEY=%s\n'        "$MINIO_ACCESS_KEY"
printf 'export MINIO_SECRET_KEY=%s\n'        "$MINIO_SECRET_KEY"
printf 'export LAKEFS_ACCESS_KEY=%s\n'       "$LAKEFS_ACCESS_KEY"
printf 'export LAKEFS_SECRET_KEY=%s\n'       "$LAKEFS_SECRET_KEY"
printf 'export MINIO_ENDPOINT=%s\n'          "${MINIO_ENDPOINT:-http://localhost:9000}"
printf 'export LAKEFS_ENDPOINT=%s\n'         "${LAKEFS_ENDPOINT:-http://localhost:8000}"
printf 'export MLFLOW_TRACKING_URI=%s\n'     "${MLFLOW_TRACKING_URI:-http://localhost:5000}"
printf 'export LAKEFS_REPO=%s\n'             "${LAKEFS_REPO:-mlops-data-dev}"
printf 'export BRANCH=%s\n'                  "${BRANCH:-main}"
