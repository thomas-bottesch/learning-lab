"""
Kubeflow Pipeline Submitter for Expert MLOps Workflow

Submit pipelines to Kubeflow with proper configuration and Git integration.
This module can be used across multiple projects.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_kfp_client():
    """
    Get Kubeflow Pipelines client.

    Returns:
        kfp.Client instance
    """
    try:
        from kfp import Client

        # Get Kubeflow endpoint from environment or use default
        kubeflow_host = os.getenv(
            "KUBEFLOW_HOST",
            "http://istio-ingressgateway.istio-system.svc.cluster.local",
        )

        client = Client(host=f"{kubeflow_host}/pipeline")
        logger.info(f"Connected to Kubeflow at {kubeflow_host}")
        return client

    except ImportError as e:
        logger.error(f"Failed to import kfp: {e}")
        logger.error("Install with: pip install kfp kfp-server-api")
        raise
    except Exception as e:
        logger.error(f"Failed to connect to Kubeflow: {e}")
        raise


def get_git_info() -> Dict[str, str]:
    """
    Extract Git information from environment or Git commands.

    Returns:
        Dictionary with Git information
    """
    git_info = {
        "commit": os.getenv("GITHUB_SHA", os.getenv("GIT_COMMIT", "unknown")),
        "branch": os.getenv("GITHUB_REF_NAME", os.getenv("GIT_BRANCH", "main")),
        "repo": os.getenv("GITHUB_REPOSITORY", os.getenv("GIT_REPO", "unknown")),
        "author": os.getenv("GITHUB_ACTOR", os.getenv("GIT_AUTHOR", "unknown")),
        "message": os.getenv(
            "GITHUB_HEAD_COMMIT_MESSAGE", os.getenv("GIT_MESSAGE", "")
        ),
        "training_image": os.getenv("TRAINING_IMAGE", ""),
        "image_tag": os.getenv("IMAGE_TAG", "latest"),
    }

    # Try to get commit from git command if not set
    if git_info["commit"] == "unknown":
        try:
            import subprocess

            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            git_info["commit"] = commit
            # Use commit as image tag
            git_info["image_tag"] = commit[:8]
        except:
            pass

    # If we have a commit but no specific image tag, use commit as tag
    if git_info["commit"] != "unknown" and git_info["image_tag"] == "latest":
        git_info["image_tag"] = git_info["commit"][:8]

    return git_info


def upload_pipeline(
    client,
    pipeline_path: str,
    pipeline_name: str,
    description: str = "Expert MLOps training pipeline",
) -> str:
    """
    Upload pipeline to Kubeflow.

    Args:
        client: Kubeflow client
        pipeline_path: Path to pipeline YAML
        pipeline_name: Name of the pipeline
        description: Pipeline description

    Returns:
        Pipeline ID
    """
    try:
        logger.info(f"Uploading pipeline: {pipeline_name}")

        # Upload to Kubeflow
        pipeline = client.upload_pipeline(
            pipeline_package_path=pipeline_path,
            pipeline_name=pipeline_name,
            description=description,
        )

        logger.info(f"Pipeline uploaded with ID: {pipeline.id}")
        return pipeline.id

    except Exception as e:
        logger.error(f"Failed to upload pipeline: {e}")
        raise


def create_experiment(
    client,
    experiment_name: str,
    description: str = "Expert MLOps workflow experiments",
) -> str:
    """
    Create or get experiment in Kubeflow.

    Args:
        client: Kubeflow client
        experiment_name: Name of experiment
        description: Experiment description

    Returns:
        Experiment ID
    """
    try:
        # Try to get existing experiment
        try:
            experiment = client.get_experiment(experiment_name=experiment_name)
            logger.info(f"Using existing experiment: {experiment_name}")
            return experiment.id
        except:
            # Create new experiment
            experiment = client.create_experiment(
                name=experiment_name, description=description
            )
            logger.info(f"Created new experiment: {experiment_name}")
            return experiment.id

    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise


def run_pipeline(
    client,
    pipeline_id: str,
    experiment_id: str,
    git_info: Dict[str, str],
    run_name: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Run a pipeline with Git information.

    Args:
        client: Kubeflow client
        pipeline_id: ID of pipeline to run
        experiment_id: ID of experiment
        git_info: Git information dictionary
        run_name: Custom run name
        additional_params: Additional pipeline parameters

    Returns:
        Run ID
    """
    try:
        if not run_name:
            run_name = f"expert-mlops-{git_info.get('commit', 'unknown')[:8]}"

        # Prepare pipeline parameters
        params = {
            "git_commit": git_info.get("commit", "unknown"),
            "git_branch": git_info.get("branch", "main"),
            "git_repo": git_info.get("repo", "unknown"),
            "experiment_name": git_info.get("experiment_name", "default-experiment"),
            "training_image": git_info.get("training_image", ""),
            "image_tag": git_info.get("image_tag", "latest"),
            "model_bucket": git_info.get("model_bucket", "models"),
            "data_bucket": git_info.get("data_bucket", "datasets"),
            "lakefs_repository": git_info.get("lakefs_repository", "mlops-data"),
        }

        # Add additional parameters if provided
        if additional_params:
            params.update(additional_params)

        logger.info(f"Starting pipeline run: {run_name}")
        logger.info(f"Parameters: {params}")
        logger.info(f"Docker image: {params['training_image']}:{params['image_tag']}")

        # Run pipeline
        run = client.run_pipeline(
            experiment_id=experiment_id,
            job_name=run_name,
            pipeline_id=pipeline_id,
            params=params,
        )

        logger.info(f"Pipeline run started with ID: {run.id}")
        logger.info(f"Run URL: {client._get_url_prefix()}/#/runs/details/{run.id}")

        return run.id

    except Exception as e:
        logger.error(f"Failed to run pipeline: {e}")
        raise


def submit_pipeline(
    pipeline_path: str,
    pipeline_name: str,
    experiment_name: str = "expert-mlops-experiments",
    run_name: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Submit a pipeline to Kubeflow (complete workflow).

    Args:
        pipeline_path: Path to pipeline YAML
        pipeline_name: Name of the pipeline
        experiment_name: Name of experiment
        run_name: Custom run name
        additional_params: Additional pipeline parameters

    Returns:
        Dictionary with run information
    """
    try:
        # Step 1: Connect to Kubeflow
        logger.info("Step 1: Connecting to Kubeflow...")
        client = get_kfp_client()

        # Step 2: Upload pipeline
        logger.info("Step 2: Uploading pipeline...")
        pipeline_id = upload_pipeline(
            client=client,
            pipeline_path=pipeline_path,
            pipeline_name=pipeline_name,
        )

        # Step 3: Create experiment
        logger.info("Step 3: Setting up experiment...")
        experiment_id = create_experiment(
            client=client,
            experiment_name=experiment_name,
        )

        # Step 4: Get Git information
        logger.info("Step 4: Gathering Git information...")
        git_info = get_git_info()
        logger.info(f"Git info: {git_info}")

        # Step 5: Run pipeline
        logger.info("Step 5: Running pipeline...")
        run_id = run_pipeline(
            client=client,
            pipeline_id=pipeline_id,
            experiment_id=experiment_id,
            git_info=git_info,
            run_name=run_name,
            additional_params=additional_params,
        )

        return {
            "pipeline_id": pipeline_id,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "git_commit": git_info.get("commit", "unknown"),
            "docker_image": f"{git_info.get('training_image', 'unknown')}:{git_info.get('image_tag', 'latest')}",
        }

    except Exception as e:
        logger.error(f"Failed to submit pipeline: {e}")
        raise


def main():
    """
    Command-line interface for submitting pipelines.
    """
    parser = argparse.ArgumentParser(
        description="Submit Expert MLOps Pipeline to Kubeflow"
    )
    parser.add_argument("--pipeline-path", required=True, help="Path to pipeline YAML")
    parser.add_argument("--pipeline-name", help="Pipeline name (default: from file)")
    parser.add_argument(
        "--experiment", default="expert-mlops-experiments", help="Experiment name"
    )
    parser.add_argument("--run-name", help="Custom run name")
    parser.add_argument("--model-bucket", default="models", help="Model bucket name")
    parser.add_argument("--data-bucket", default="datasets", help="Data bucket name")
    parser.add_argument("--lakefs-repo", default="mlops-data", help="LakeFS repository")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Use filename as pipeline name if not provided
        pipeline_name = args.pipeline_name or Path(args.pipeline_path).stem

        # Prepare additional parameters
        additional_params = {
            "model_bucket": args.model_bucket,
            "data_bucket": args.data_bucket,
            "lakefs_repository": args.lakefs_repo,
        }

        # Submit pipeline
        result = submit_pipeline(
            pipeline_path=args.pipeline_path,
            pipeline_name=pipeline_name,
            experiment_name=args.experiment,
            run_name=args.run_name,
            additional_params=additional_params,
        )

        logger.info("=" * 60)
        logger.info("✅ Expert MLOps pipeline submitted successfully!")
        logger.info("=" * 60)
        logger.info(f"Pipeline Run ID: {result['run_id']}")
        logger.info(
            f"Git Commit: {result['git_commit'][:8] if result['git_commit'] != 'unknown' else 'unknown'}"
        )
        logger.info(f"Docker Image: {result['docker_image']}")
        logger.info(f"Experiment: {args.experiment}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Monitor pipeline execution in Kubeflow UI")
        logger.info("2. Check MLflow for experiment tracking")
        logger.info("3. View artifacts in MinIO")
        logger.info("4. Check data versions in LakeFS")

    except Exception as e:
        logger.error(f"Failed to submit pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
