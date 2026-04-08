"""
MLOps Utilities for Expert Workflow

Common utilities for data versioning, experiment tracking, and artifact management.
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class MLOpsClient:
    """Client for MLOps operations across all services."""

    def __init__(self):
        self.config = self._load_config()
        self.mlflow_client = None
        self.minio_client = None
        self.s3_client = None

    def _load_config(self) -> Dict[str, str]:
        """Load configuration from environment with validation."""
        config = {}

        # Required configuration - raise error if missing
        required_vars = [
            "MLFLOW_TRACKING_URI",
            "LAKEFS_ENDPOINT",
            "LAKEFS_ACCESS_KEY",
            "LAKEFS_SECRET_KEY",
            "MINIO_ENDPOINT",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
        ]

        for var in required_vars:
            value = os.getenv(var)
            if not value:
                raise ValueError(
                    f"Required environment variable {var} is not set. "
                    f"This typically means the MLOps ConfigMap and Secret are not mounted to your pod.\n"
                    f"Please ensure your Kubernetes deployment includes:\n"
                    f"  - envFrom:\n"
                    f"    - configMapRef:\n"
                    f"        name: pipeline-configmap\n"
                    f"    - secretRef:\n"
                    f"        name: mlops-credentials"
                )
            config[var.lower()] = value

        # Optional configuration with defaults
        optional_vars = {
            "MODEL_BUCKET": "models",
            "DATA_BUCKET": "datasets",
            "LAKEFS_BUCKET_NAME": "datasets",  # Use datasets bucket for LakeFS
            "LAKEFS_REPOSITORY": "mlops-data",
            "LAKEFS_BRANCH": "main",
        }

        for var, default in optional_vars.items():
            config[var.lower()] = os.getenv(var, default)

        logger.info(f"MLOps configuration loaded successfully")
        logger.info(f"  MLflow: {config.get('mlflow_tracking_uri')}")
        logger.info(f"  LakeFS: {config.get('lakefs_endpoint')}")
        logger.info(f"  MinIO: {config.get('minio_endpoint')}")

        return config

    def get_mlflow_client(self):
        """Get or create MLflow client with lazy imports."""
        if self.mlflow_client is None:
            import mlflow

            # Set tracking URI for this client instance
            mlflow.set_tracking_uri(self.config["mlflow_tracking_uri"])
            self.mlflow_client = mlflow
        return self.mlflow_client

    def get_minio_client(self):
        """Get or create MinIO client with lazy imports."""
        if self.minio_client is None:
            from minio import Minio

            endpoint = self.config["minio_endpoint"]
            access_key = self.config["minio_access_key"]
            secret_key = self.config["minio_secret_key"]

            # Remove protocol prefix for MinIO client
            endpoint_host = endpoint.replace("http://", "").replace("https://", "")

            logger.info(f"Initializing MinIO client for {endpoint}")
            self.minio_client = Minio(
                endpoint_host,
                access_key=access_key,
                secret_key=secret_key,
                secure=False,
            )
        return self.minio_client

    def get_s3_client(self):
        """Get or create S3 client (for LakeFS compatibility) with lazy imports."""
        if self.s3_client is None:
            import boto3
            from botocore.client import Config

            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.config["minio_endpoint"],
                aws_access_key_id=self.config["minio_access_key"],
                aws_secret_access_key=self.config["minio_secret_key"],
                config=Config(signature_version="s3v4"),
            )
        return self.s3_client

    def create_experiment(self, experiment_name: str) -> str:
        """Create or get MLflow experiment."""
        mlflow = self.get_mlflow_client()
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name} ({experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing experiment: {experiment_name} ({experiment_id})"
                )
            return experiment_id
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise  # Re-raise the exception - don't silently fail

    def start_run(
        self, experiment_name: str, run_name: str = None, tags: Dict[str, str] = None
    ):
        """Start a new MLflow run."""
        mlflow = self.get_mlflow_client()
        mlflow.set_experiment(experiment_name)
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_metric(self, key: str, value: float, step: int = None):
        """Log a metric to the current MLflow run."""
        mlflow = self.get_mlflow_client()
        mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: str):
        """Log a parameter to the current MLflow run."""
        mlflow = self.get_mlflow_client()
        mlflow.log_param(key, value)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log an artifact to the current MLflow run."""
        mlflow = self.get_mlflow_client()
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, artifact_path: str, **kwargs):
        """Log a model to MLflow (generic, not just sklearn)."""
        mlflow = self.get_mlflow_client()
        # Try different model logging methods
        try:
            # Try sklearn first
            import sklearn

            if isinstance(model, sklearn.base.BaseEstimator):
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
                return
        except (ImportError, AttributeError):
            pass

        # Generic model logging
        import joblib
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            joblib.dump(model, tmp.name)
            mlflow.log_artifact(tmp.name, artifact_path)
            os.unlink(tmp.name)

    def upload_to_minio(
        self, file_path: str, bucket_name: str, object_name: str = None
    ) -> str:
        """Upload a file to MinIO bucket."""
        try:
            minio_client = self.get_minio_client()

            # Use filename as object name if not provided
            if object_name is None:
                object_name = Path(file_path).name

            # Ensure bucket exists
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)

            # Upload to MinIO
            minio_client.fput_object(bucket_name, object_name, file_path)

            s3_url = f"s3://{bucket_name}/{object_name}"
            logger.info(f"Uploaded to MinIO: {s3_url}")

            return s3_url

        except Exception as e:
            logger.error(f"Failed to upload to MinIO: {e}")
            raise

    def generate_run_tags(self, git_info: Dict[str, str] = None) -> Dict[str, str]:
        """Generate tags for MLflow run."""
        tags = {
            "mlflow.source.type": "JOB",
            "mlflow.source.name": "Expert MLOps Pipeline",
            "timestamp": datetime.now().isoformat(),
            "pipeline.run_id": os.getenv("KFP_RUN_ID", str(uuid.uuid4())),
        }

        if git_info:
            tags.update(
                {
                    "mlflow.source.git.commit": git_info.get("commit", ""),
                    "mlflow.source.git.repoURL": git_info.get("repo_url", ""),
                    "mlflow.source.git.branch": git_info.get("branch", ""),
                }
            )

        return tags

    def download_from_minio(self, bucket_name: str, object_name: str, file_path: str):
        """Download a file from MinIO bucket."""
        try:
            minio_client = self.get_minio_client()
            minio_client.fget_object(bucket_name, object_name, file_path)
            logger.info(
                f"Downloaded from MinIO: {bucket_name}/{object_name} -> {file_path}"
            )
        except Exception as e:
            logger.error(f"Failed to download from MinIO: {e}")
            raise

    def list_minio_objects(self, bucket_name: str, prefix: str = "") -> list:
        """List objects in MinIO bucket."""
        try:
            minio_client = self.get_minio_client()
            objects = minio_client.list_objects(
                bucket_name, prefix=prefix, recursive=True
            )
            return [obj.object_name for obj in objects]
        except Exception as e:
            logger.error(f"Failed to list MinIO objects: {e}")
            raise


def get_git_info() -> Dict[str, str]:
    """Extract Git information from environment (for CI/CD)."""
    return {
        "commit": os.getenv("GITHUB_SHA", os.getenv("GIT_COMMIT", "")),
        "branch": os.getenv("GITHUB_REF_NAME", os.getenv("GIT_BRANCH", "")),
        "repo_url": os.getenv("GITHUB_SERVER_URL", os.getenv("GIT_REPO_URL", "")),
        "author": os.getenv("GITHUB_ACTOR", os.getenv("GIT_AUTHOR", "")),
    }
