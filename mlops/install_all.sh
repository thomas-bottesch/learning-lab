# Order is important because some secrets and configmaps are based on the previous installations

bash install_lakefs.sh
bash install_minio.sh
bash install_gitea.sh
bash install_mlflow.sh
bash install_drone.sh

bash create_mlops_configmap.sh
bash create_mlops_secret.sh

bash install_kubeflow.sh