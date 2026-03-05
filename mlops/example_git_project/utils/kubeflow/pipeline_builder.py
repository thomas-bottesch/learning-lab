"""
Kubeflow Pipeline Builder for Expert MLOps Workflow

Build reusable Kubeflow pipelines using the pipeline components.
This module provides factory functions to create pipelines for different projects.
"""

from typing import Dict, Any, Optional, Callable
from kfp import dsl
from kfp import kubernetes

from .pipeline_components import (
    expert_training_component,
    create_lakefs_branch_component,
    merge_lakefs_branch_component,
    upload_to_minio_component,
)


def create_expert_mlops_pipeline(
    pipeline_name: str = "expert-mlops-training-pipeline",
    pipeline_description: str = "Expert MLOps training pipeline with data versioning",
    configmap_name: str = "pipeline-configmap",
    secret_name: str = "mlops-credentials",
    enable_data_versioning: bool = True,
    enable_auto_merge: bool = False,
    enable_artifact_upload: bool = True,
    custom_training_component: Optional[Callable] = None,
) -> Callable:
    """
    Create a reusable expert MLOps pipeline.

    Args:
        pipeline_name: Name of the pipeline
        pipeline_description: Description of the pipeline
        configmap_name: Name of ConfigMap with service endpoints
        secret_name: Name of Secret with credentials
        enable_data_versioning: Whether to create LakeFS branches
        enable_auto_merge: Whether to auto-merge branches after success
        enable_artifact_upload: Whether to upload artifacts to MinIO
        custom_training_component: Custom training component (uses default if None)

    Returns:
        A Kubeflow pipeline function
    """

    # Use custom training component if provided, otherwise use default
    training_component = custom_training_component or expert_training_component

    @dsl.pipeline(name=pipeline_name, description=pipeline_description)
    def expert_pipeline(
        git_commit: str = "unknown",
        git_branch: str = "main",
        git_repo: str = "unknown",
        experiment_name: str = "default-experiment",
        training_image: str = "${GITHUB_REPOSITORY_OWNER}/${GITHUB_REPOSITORY_NAME}",
        image_tag: str = "latest",
        model_bucket: str = "models",
        data_bucket: str = "datasets",
        lakefs_repository: str = "mlops-data",
    ) -> None:
        """
        Expert MLOps pipeline with data versioning and experiment tracking.

        Args:
            git_commit: Git commit SHA
            git_branch: Git branch name
            git_repo: Git repository URL
            experiment_name: MLflow experiment name
            training_image: Docker image for training
            image_tag: Docker image tag
            model_bucket: MinIO bucket for models
            data_bucket: MinIO bucket for data
            lakefs_repository: LakeFS repository name
        """

        # Step 1: Create LakeFS branch for data versioning (optional)
        lakefs_branch_task = None
        if enable_data_versioning:
            lakefs_branch_task = create_lakefs_branch_component(
                source_branch="main",
                branch_name="",  # Empty string - component will generate name
                lakefs_repository=lakefs_repository,
            )

            # Inject credentials for LakeFS task
            kubernetes.use_secret_as_env(
                task=lakefs_branch_task,
                secret_name=secret_name,
                secret_key_to_env={
                    "LAKEFS_ACCESS_KEY": "LAKEFS_ACCESS_KEY",
                    "LAKEFS_SECRET_KEY": "LAKEFS_SECRET_KEY",
                },
            )

        # Step 2: Run expert training
        training_task = training_component(
            git_commit=git_commit,
            git_branch=git_branch,
            git_repo=git_repo,
            experiment_name=experiment_name,
            model_bucket=model_bucket,
            data_bucket=data_bucket,
            training_image=training_image,
            image_tag=image_tag,
        )

        # Inject ConfigMap and Secret into training task
        kubernetes.use_config_map_as_env(
            task=training_task,
            config_map_name=configmap_name,
            config_map_key_to_env={
                "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
                "LAKEFS_ENDPOINT": "LAKEFS_ENDPOINT",
                "MINIO_ENDPOINT": "MINIO_ENDPOINT",
            },
        )
        kubernetes.use_secret_as_env(
            task=training_task,
            secret_name=secret_name,
            secret_key_to_env={
                "LAKEFS_ACCESS_KEY": "LAKEFS_ACCESS_KEY",
                "LAKEFS_SECRET_KEY": "LAKEFS_SECRET_KEY",
                "MINIO_ACCESS_KEY": "MINIO_ACCESS_KEY",
                "MINIO_SECRET_KEY": "MINIO_SECRET_KEY",
            },
        )

        # Configure training task resources
        training_task.set_cpu_limit("2")
        training_task.set_memory_limit("4Gi")

        # Optional: Set dependency if using data versioning
        if enable_data_versioning and lakefs_branch_task:
            training_task.after(lakefs_branch_task)

        # Step 3: Upload artifacts to MinIO (optional)
        if enable_artifact_upload:
            upload_task = upload_to_minio_component(
                file_path="/tmp/model.joblib",
                bucket_name=model_bucket,
                object_name="model.joblib",  # Simple name - can be customized per project
            )

            # Upload depends on training success
            upload_task.after(training_task)

            # Inject credentials for upload task
            kubernetes.use_secret_as_env(
                task=upload_task,
                secret_name=secret_name,
                secret_key_to_env={
                    "MINIO_ACCESS_KEY": "MINIO_ACCESS_KEY",
                    "MINIO_SECRET_KEY": "MINIO_SECRET_KEY",
                },
            )

        # Step 4: Conditionally merge LakeFS branch (optional)
        if enable_data_versioning and enable_auto_merge and lakefs_branch_task:
            merge_task = merge_lakefs_branch_component(
                source_branch=lakefs_branch_task.output,
                target_branch="main",
                lakefs_repository=lakefs_repository,
                merge_message=f"Merge training results from commit {git_commit[:8]}",
            )

            # Merge depends on training success
            merge_task.after(training_task)

            # Inject credentials for merge task
            kubernetes.use_secret_as_env(
                task=merge_task,
                secret_name=secret_name,
                secret_key_to_env={
                    "LAKEFS_ACCESS_KEY": "LAKEFS_ACCESS_KEY",
                    "LAKEFS_SECRET_KEY": "LAKEFS_SECRET_KEY",
                },
            )

    return expert_pipeline


def create_simple_training_pipeline(
    pipeline_name: str = "simple-training-pipeline",
    configmap_name: str = "pipeline-configmap",
    secret_name: str = "mlops-credentials",
) -> Callable:
    """
    Create a simple training pipeline without data versioning.

    Args:
        pipeline_name: Name of the pipeline
        configmap_name: Name of ConfigMap
        secret_name: Name of Secret

    Returns:
        A simple Kubeflow pipeline function
    """
    return create_expert_mlops_pipeline(
        pipeline_name=pipeline_name,
        pipeline_description="Simple training pipeline without data versioning",
        enable_data_versioning=False,
        enable_auto_merge=False,
        enable_artifact_upload=True,
    )


def create_full_mlops_pipeline(
    pipeline_name: str = "full-mlops-pipeline",
    configmap_name: str = "pipeline-configmap",
    secret_name: str = "mlops-credentials",
) -> Callable:
    """
    Create a full MLOps pipeline with all features enabled.

    Args:
        pipeline_name: Name of the pipeline
        configmap_name: Name of ConfigMap
        secret_name: Name of Secret

    Returns:
        A full-featured Kubeflow pipeline function
    """
    return create_expert_mlops_pipeline(
        pipeline_name=pipeline_name,
        pipeline_description="Full MLOps pipeline with data versioning and auto-merge",
        enable_data_versioning=True,
        enable_auto_merge=True,
        enable_artifact_upload=True,
    )


def compile_pipeline(
    pipeline_func: Callable,
    output_path: str = "compiled_pipeline.yaml",
) -> str:
    """
    Compile a Kubeflow pipeline to a YAML file.

    Args:
        pipeline_func: Pipeline function to compile
        output_path: Path to save compiled pipeline

    Returns:
        Path to compiled pipeline
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=output_path,
    )

    return output_path
