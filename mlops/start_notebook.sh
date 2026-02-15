kubectl apply -f k8s_yamls/kubeflow/notebook.yaml

# Wait for the notebook resource to be Ready
while true; do
    ready=$(kubectl get notebooks -n user-example-com ds-notebook -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')
    if [[ "$ready" == "True" ]]; then
        break
    fi
    echo "Waiting for the notebook resource to be Ready..."
    sleep 10
done

echo "Notebook is ready!"

# Prepare an example notebook and a training script
NOTEBOOK_POD=$(kubectl -n user-example-com get pods -l notebook-name=ds-notebook -o jsonpath='{.items[0].metadata.name}')
kubectl -n user-example-com cp ./projects/kubeflow/simple_train.py $NOTEBOOK_POD:/home/jovyan/simple_train.py
kubectl -n user-example-com cp ./projects/kubeflow/simple_train.ipynb $NOTEBOOK_POD:/home/jovyan/simple_train.ipynb