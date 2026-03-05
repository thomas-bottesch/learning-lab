"""
Kubeflow Pipeline Components for Expert MLOps Workflow

Reusable Kubeflow components that can be used across multiple projects.
These components handle data versioning, training, and artifact management.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from kfp import dsl
from kfp import kubernetes
from kfp.dsl import Dataset, Input, Output, Metrics, Model

logger = logging.getLogger(__name__)


@dsl.component(
    # Use a default base image - this will be overridden at runtime
    # The actual image is passed as a parameter and will be used by the pipeline
    base_image="python:3.12-slim",
)
def expert_training_component(
    git_commit: str,
    git_branch: str,
    git_repo: str,
    experiment_name: str = "default-experiment",
    model_bucket: str = "models",
    data_bucket: str = "datasets",
    training_image: str = "",
    image_tag: str = "latest",
    output_metrics: Output[Metrics] = None,
    output_model: Output[Model] = None,
) -> Dict[str, Any]:
    """
    Reusable Kubeflow component for expert training workflow.

    This component uses a custom Docker image built by CI/CD.
    The image contains all dependencies and training code.

    Args:
        git_commit: Git commit SHA
        git_branch: Git branch name
        git_repo: Git repository URL
        experiment_name: MLflow experiment name
        model_bucket: MinIO bucket for models
        data_bucket: MinIO bucket for data
        output_metrics: Kubeflow metrics output
        output_model: Kubeflow model output

    Returns:
        Dictionary with run information
    """
    import sys
    import traceback
    from pathlib import Path
    import json
    from datetime import datetime

    try:
        # Import training module (should be in the Docker image)
        # Note: Different projects will have different training modules
        # This is a template that projects should customize

        print(f"Starting expert training for commit: {git_commit[:8]}")
        print(f"Experiment: {experiment_name}")
        print(f"Git branch: {git_branch}")
        print(f"Using Docker image: {training_image}:{image_tag}")

        # Import MLOps utilities (should be in the Docker image)
        from utils.mlops_utils import MLOpsClient

        # Initialize MLOps client (uses environment variables from ConfigMap/Secret)
        mlops_client = MLOpsClient()

        # Create MLflow experiment
        experiment_id = mlops_client.create_experiment(experiment_name)

        # Start MLflow run with Git metadata
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Log Git information
            mlflow.set_tag("mlflow.source.git.commit", git_commit)
            mlflow.set_tag("mlflow.source.git.repoURL", git_repo)
            mlflow.set_tag("mlflow.source.git.branch", git_branch)
            mlflow.set_tag("mlflow.source.type", "KUBEFLOW_PIPELINE")
            mlflow.set_tag("kubeflow_run_id", os.getenv("KFP_RUN_ID", "unknown"))
            mlflow.set_tag("docker_image", f"{training_image}:{image_tag}")

            # PROJECT-SPECIFIC: Import and run training
            # This should be customized per project
            try:
                # Try to import project-specific training module
                from train import main as train_main

                print("Running project-specific training module...")
                exit_code = train_main()
            except ImportError:
                # No fallback raise this issue to be handled by the project
                raise ImportError(
                    "Project-specific training module not found. "
                    "Please ensure your Docker image includes the training code and dependencies."
                )

            if exit_code != 0:
                raise RuntimeError(f"Training failed with exit code: {exit_code}")

            # Read metrics from training output
            metrics_path = Path("/tmp/training_metrics.json")
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                # Log metrics to Kubeflow output
                if output_metrics:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            output_metrics.log_metric(key, value)
                            mlflow.log_metric(key, value)

            # Save model to Kubeflow output
            model_path = Path("/tmp/model.joblib")
            if model_path.exists() and output_model:
                import joblib

                model = joblib.load(model_path)
                joblib.dump(model, output_model.path)
                print(f"Model saved to: {output_model.path}")

            print(f"Training completed successfully. MLflow Run ID: {run.info.run_id}")

            # Return success
            return {
                "mlflow_run_id": run.info.run_id,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "docker_image": f"{training_image}:{image_tag}",
            }

    except Exception as e:
        print(f"Training component failed: {e}")
        traceback.print_exc()
        raise


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests==2.32.4"],
)
def create_lakefs_branch_component(
    source_branch: str = "main",
    branch_name: str = None,
    lakefs_repository: str = "mlops-data",
) -> str:
    """
    Create a LakeFS branch for an experiment.

    Args:
        source_branch: Source branch to create from
        branch_name: Name of new branch (auto-generated if None)
        lakefs_repository: LakeFS repository name

    Returns:
        Name of the created branch
    """
    import os
    import requests
    from datetime import datetime

    try:
        lakefs_access_key = os.environ["LAKEFS_ACCESS_KEY"]
        lakefs_secret_key = os.environ["LAKEFS_SECRET_KEY"]
        lakefs_endpoint = os.environ["LAKEFS_ENDPOINT"]

        # Generate branch name if not provided
        if not branch_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"experiment-{timestamp}"

        # Check if repository exists
        repo_url = (
            f"{lakefs_endpoint.rstrip('/')}/api/v1/repositories/{lakefs_repository}"
        )
        repo_resp = requests.get(
            repo_url,
            auth=(lakefs_access_key, lakefs_secret_key),
            timeout=30,
        )

        if repo_resp.status_code == 404:
            # Create repository if it doesn't exist
            create_url = f"{lakefs_endpoint.rstrip('/')}/api/v1/repositories"
            create_payload = {
                "name": lakefs_repository,
                "storage_namespace": f"s3://{os.getenv('LAKEFS_BUCKET_NAME', 'datasets')}/lakefs/{lakefs_repository}",
                "default_branch": source_branch,
            }
            create_resp = requests.post(
                create_url,
                json=create_payload,
                auth=(lakefs_access_key, lakefs_secret_key),
                timeout=30,
            )
            create_resp.raise_for_status()
            print(f"Created LakeFS repository: {lakefs_repository}")

        # Create branch
        branch_url = f"{lakefs_endpoint.rstrip('/')}/api/v1/repositories/{lakefs_repository}/branches"
        branch_payload = {
            "name": branch_name,
            "source": source_branch,
        }
        branch_resp = requests.post(
            branch_url,
            json=branch_payload,
            auth=(lakefs_access_key, lakefs_secret_key),
            timeout=30,
        )

        if branch_resp.status_code == 409:
            print(f"Branch {branch_name} already exists, reusing it")
        else:
            branch_resp.raise_for_status()
            print(f"Created LakeFS branch: {branch_name} from {source_branch}")

        return branch_name

    except Exception as e:
        print(f"Failed to create LakeFS branch: {e}")
        raise


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["requests==2.32.4"],
)
def merge_lakefs_branch_component(
    source_branch: str,
    target_branch: str = "main",
    lakefs_repository: str = "mlops-data",
    merge_message: str = "Merge experiment branch",
) -> Dict[str, str]:
    """
    Merge a LakeFS branch back to main if conditions are met.

    Args:
        source_branch: Branch to merge from
        target_branch: Branch to merge into (default: main)
        lakefs_repository: LakeFS repository name
        merge_message: Commit message for the merge

    Returns:
        Dictionary with merge status
    """
    import os
    import requests

    try:
        lakefs_access_key = os.environ["LAKEFS_ACCESS_KEY"]
        lakefs_secret_key = os.environ["LAKEFS_SECRET_KEY"]
        lakefs_endpoint = os.environ["LAKEFS_ENDPOINT"]

        merge_url = f"{lakefs_endpoint.rstrip('/')}/api/v1/repositories/{lakefs_repository}/merges"
        merge_payload = {
            "source_ref": source_branch,
            "destination_branch": target_branch,
            "message": merge_message,
        }

        merge_resp = requests.post(
            merge_url,
            json=merge_payload,
            auth=(lakefs_access_key, lakefs_secret_key),
            timeout=30,
        )

        if merge_resp.status_code == 409:
            print(f"Merge conflict for {source_branch} -> {target_branch}")
            return {"status": "conflict", "message": "Merge conflict detected"}
        else:
            merge_resp.raise_for_status()
            print(f"Merged {source_branch} into {target_branch}")
            return {"status": "merged", "message": "Branch successfully merged"}

    except Exception as e:
        print(f"Failed to merge LakeFS branch: {e}")
        return {"status": "error", "message": str(e)}


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["minio==7.2.6"],
)
def upload_to_minio_component(
    file_path: str,
    bucket_name: str,
    object_name: str = None,
) -> str:
    """
    Upload a file to MinIO.

    Args:
        file_path: Local path to file
        bucket_name: MinIO bucket name
        object_name: Object name in bucket (default: basename of file_path)

    Returns:
        S3 URL of uploaded object
    """
    import os
    from pathlib import Path
    from minio import Minio

    try:
        minio_endpoint = os.environ["MINIO_ENDPOINT"]
        minio_access_key = os.environ["MINIO_ACCESS_KEY"]
        minio_secret_key = os.environ["MINIO_SECRET_KEY"]

        # Generate object name if not provided
        if not object_name:
            object_name = Path(file_path).name

        # Initialize MinIO client
        endpoint_host = minio_endpoint.replace("http://", "").replace("https://", "")
        minio_client = Minio(
            endpoint_host,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
        )

        # Ensure bucket exists
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        # Upload file
        minio_client.fput_object(bucket_name, object_name, file_path)

        s3_url = f"s3://{bucket_name}/{object_name}"
        print(f"Uploaded to MinIO: {s3_url}")

        return s3_url

    except Exception as e:
        print(f"Failed to upload to MinIO: {e}")
        raise
