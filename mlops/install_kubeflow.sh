#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

patch_kubeflow_profile_manifest() {
    local params_file="common/user-namespace/base/params.env"
    local profile_file="common/user-namespace/base/profile-instance.yaml"
    local profile_name user_name
    local escaped_profile_name escaped_user_name

    profile_name="$(grep '^profile-name=' "$params_file" | head -n 1 | cut -d'=' -f2-)"
    user_name="$(grep '^user=' "$params_file" | head -n 1 | cut -d'=' -f2-)"

    escaped_profile_name="${profile_name//&/\\&}"
    escaped_profile_name="${escaped_profile_name//\//\\/}"
    escaped_user_name="${user_name//&/\\&}"
    escaped_user_name="${escaped_user_name//\//\\/}"

    sed -i \
        -e 's|\$(profile-name)|'"$escaped_profile_name"'|g' \
        -e 's|\$(user)|'"$escaped_user_name"'|g' \
        "$profile_file"
}

rm -rf /tmp/manifests
cd /tmp/
git clone https://github.com/kubeflow/manifests.git
cd manifests
git checkout v1.11-branch
patch_kubeflow_profile_manifest

while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do
    echo "Retrying to apply resources..."
    sleep 20
done

# Expose the kubeflow dashboard via LoadBalancer (more resilient than port-forwarding)
kubectl apply -f "$SCRIPT_DIR/k8s_yamls/kubeflow/ingressgateway-loadbalancer.yaml"

# Wait for the LoadBalancer service to be ready
echo "Waiting for LoadBalancer service to be ready..."
while ! kubectl -n istio-system get svc istio-ingressgateway-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}' >/dev/null 2>&1; do
    echo "Waiting for LoadBalancer IP..."
    sleep 2
done

LB_IP=$(kubectl -n istio-system get svc istio-ingressgateway-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Kubeflow dashboard is accessible at: http://localhost:8080"

# Create example notebooks
# we need to cd into the dir of this script
cd "$SCRIPT_DIR"
kubectl apply -f k8s_yamls/kubeflow/user-profile.yaml