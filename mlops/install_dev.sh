#!/bin/bash
# install_dev.sh: Lightweight dev stack — MinIO, LakeFS, Zot, MLflow only.
# Skips Forgejo and Kubeflow (use KFP SubprocessRunner locally instead).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Installing MLOps dev stack"
echo "=========================================="

bash install_minio.sh
bash install_lakefs.sh
bash install_zot.sh
bash install_mlflow.sh

# Create ConfigMap and Secret in the default namespace for local dev use.
# (kubeflow-user-example-com does not exist without Kubeflow installed.)
bash create_mlops_configmap.sh default mlops-endpoints
bash create_mlops_secret.sh default mlops-credentials

echo ""
echo "=========================================="
echo "Dev stack ready."
echo "  LakeFS:  http://localhost:8000"
echo "  MinIO:   http://localhost:9000  (console: http://localhost:9001)"
echo "  Zot:     http://localhost:8001"
echo "  MLflow:  http://localhost:5000"
echo ""
echo "Copy .env.local.example to .env.local in your pipeline repo and run:"
echo "  python run_local.py"
echo "=========================================="
