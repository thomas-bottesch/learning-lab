"""
Project-specific Kubeflow Integration

This file shows how to use the reusable Kubeflow utilities
for a specific project. Each project can customize this.
"""

import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent / "utils"))

from kfp import dsl
from kfp import kubernetes
from utils.kubeflow.pipeline_components import expert_training_component
from utils.kubeflow.pipeline_builder import create_expert_mlops_pipeline
from utils.kubeflow.pipeline_submitter import submit_pipeline


# Project-specific training component (optional)
# If you need custom training logic, create a custom component
@dsl.component(
    # Use a default base image - this will be overridden at runtime
    # The actual image is passed as a parameter and will be used by the pipeline
    base_image="python:3.12-slim",
)
def iris_training_component(
    git_commit: str,
    git_branch: str,
    git_repo: str,
    experiment_name: str = "iris-classification",
    model_bucket: str = "models",
    data_bucket: str = "datasets",
    training_image: str = "",
    image_tag: str = "latest",
    output_metrics: dsl.Output[dsl.Metrics] = None,
    output_model: dsl.Output[dsl.Model] = None,
):
    """
    Project-specific training component for Iris classification.

    This uses our Docker image and runs the Iris training script.
    """
    import sys
    import traceback
    from pathlib import Path

    try:
        # Our training code is in the Docker image
        from train import main as train_main

        print(f"Starting Iris training for commit: {git_commit[:8]}")
        print(f"Experiment: {experiment_name}")
        print(f"Using Docker image: {training_image}:{image_tag}")

        # Start MLflow run
        import mlflow

        # MLflow configuration is already set via ConfigMap/Secret
        # The environment variables are injected by Kubernetes
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Log Git information
            mlflow.set_tag("mlflow.source.git.commit", git_commit)
            mlflow.set_tag("mlflow.source.git.repoURL", git_repo)
            mlflow.set_tag("mlflow.source.git.branch", git_branch)
            mlflow.set_tag("mlflow.source.type", "KUBEFLOW_PIPELINE")
            mlflow.set_tag("project", "iris-classification")
            mlflow.set_tag("docker_image", f"{training_image}:{image_tag}")

            # Run our project-specific training
            print("Running Iris training...")
            exit_code = train_main()

            if exit_code != 0:
                raise RuntimeError(f"Training failed with exit code: {exit_code}")

            print(f"Iris training completed. MLflow Run ID: {run.info.run_id}")

            # Return success
            return {
                "mlflow_run_id": run.info.run_id,
                "status": "success",
                "project": "iris-classification",
            }

    except Exception as e:
        print(f"Iris training component failed: {e}")
        traceback.print_exc()
        raise


# Create project-specific pipeline using reusable builder
iris_pipeline = create_expert_mlops_pipeline(
    pipeline_name="expert-mlops-pipeline",
    pipeline_description="Expert MLOps pipeline with data versioning",
    enable_data_versioning=True,
    enable_auto_merge=False,  # Set to True for automatic promotion
    custom_training_component=iris_training_component,  # Use our custom component
)


def compile_iris_pipeline(output_path: str = "iris_pipeline.yaml"):
    """
    Compile the Iris classification pipeline.

    Usage:
        python project_kubeflow_integration.py
    """
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path=output_path,
    )

    print(f"Expert MLOps pipeline compiled to: {output_path}")
    print("Upload this to Kubeflow Pipelines UI or use the submitter.")


def submit_iris_pipeline():
    """
    Submit the Iris classification pipeline to Kubeflow.

    Usage:
        python project_kubeflow_integration.py --submit
    """
    # Compile pipeline first
    pipeline_path = "iris_pipeline.yaml"
    compile_iris_pipeline(pipeline_path)

    # Submit using reusable submitter
    result = submit_pipeline(
        pipeline_path=pipeline_path,
        pipeline_name="expert-mlops-pipeline",
        experiment_name="mlops-experiments",
    )

    print("=" * 60)
    print("✅ Expert MLOps pipeline submitted successfully!")
    print("=" * 60)
    print(f"Run ID: {result['run_id']}")
    print(
        f"Git Commit: {result['git_commit'][:8] if result['git_commit'] != 'unknown' else 'unknown'}"
    )
    print(f"Docker Image: {result['docker_image']}")
    print("")
    print("Next steps:")
    print("1. Monitor in Kubeflow UI: http://localhost:8080")
    print("2. Check MLflow: http://localhost:5000")
    print("3. View artifacts in MinIO: http://localhost:9000")
    print("4. Check data versions in LakeFS: http://localhost:8000")

    parser = argparse.ArgumentParser(description="Expert MLOps Kubeflow Integration")
    parser.add_argument("--compile", action="store_true", help="Compile pipeline only")
    parser.add_argument(
        "--submit", action="store_true", help="Compile and submit pipeline"
    )

    args = parser.parse_args()

    if args.submit:
        submit_iris_pipeline()
    elif args.compile:
        compile_iris_pipeline()
    else:
        print("Usage:")
        print("  python project_kubeflow_integration.py --compile  # Compile pipeline")
        print(
            "  python project_kubeflow_integration.py --submit   # Compile and submit"
        )
        print("")
        print("This uses the standard buckets configured in the cluster:")
        print("  - Model bucket: models (from ConfigMap)")
        print("  - Data bucket: datasets (from ConfigMap)")
        print("  - LakeFS repository: mlops-data (default)")
        print("")
        print("Or use the reusable utilities directly:")
        print(
            "  from utils.kubeflow.pipeline_builder import create_expert_mlops_pipeline"
        )
        print("  from utils.kubeflow.pipeline_submitter import submit_pipeline")
