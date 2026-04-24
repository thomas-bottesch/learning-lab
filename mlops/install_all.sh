#!/bin/bash
# install_all.sh: Full platform install. Order is critical — some secrets and
# ConfigMaps are derived from previous installations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bash install_lakefs.sh
bash install_minio.sh
bash install_zot.sh
bash install_forgejo.sh
bash install_mlflow.sh
bash install_kubeflow.sh
bash create_mlops_configmap.sh
bash create_mlops_secret.sh