from kfp import dsl
from kfp import kubernetes
from kfp.dsl import Dataset, Input, Output, Metrics, Model
from kfp_client_utils import get_kfp_client


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["requests==2.32.4"],
)
def lakefs_create_branch(
    source_branch: str,
    branch_name: str,
    lakefs_repository: str,
    bucket_name: str,
) -> str:
    """Create a new branch in LakeFS for the experiment."""
    import os, requests, traceback, sys

    try:
        lakefs_access_key = os.environ["LAKEFS_ACCESS_KEY"]
        lakefs_secret_key = os.environ["LAKEFS_SECRET_KEY"]
        lakefs_endpoint = os.environ["LAKEFS_ENDPOINT"]

        desired_storage_namespace = f"s3://{bucket_name}/lakefs/{lakefs_repository}"

        repo_url = (
            f"{lakefs_endpoint.rstrip('/')}/api/v1/repositories/{lakefs_repository}"
        )
        repo_resp = requests.get(
            repo_url,
            auth=(lakefs_access_key, lakefs_secret_key),
            timeout=30,
        )

        if repo_resp.status_code == 404:
            create_repo_url = f"{lakefs_endpoint.rstrip('/')}/api/v1/repositories"
            create_payload = {
                "name": lakefs_repository,
                "storage_namespace": desired_storage_namespace,
                "default_branch": source_branch,
            }
            create_resp = requests.post(
                create_repo_url,
                json=create_payload,
                auth=(lakefs_access_key, lakefs_secret_key),
                timeout=30,
            )
            if create_resp.status_code == 400:
                print("LakeFS repo creation payload:", create_payload)
                print("LakeFS error response:", create_resp.text)
            if create_resp.status_code not in (200, 201, 409):
                create_resp.raise_for_status()
            print(
                f"Created missing repository '{lakefs_repository}' "
                f"with default branch '{source_branch}'."
            )
        else:
            repo_resp.raise_for_status()
            repo_body = repo_resp.json()
            actual_storage_namespace = repo_body.get("storage_namespace")
            if actual_storage_namespace != desired_storage_namespace:
                raise RuntimeError(
                    "Existing lakeFS repository has unexpected storage namespace. "
                    f"repo={lakefs_repository}, expected={desired_storage_namespace}, "
                    f"actual={actual_storage_namespace}. "
                    "Update the pipeline bucket_name/repository values to match, "
                    "or recreate the repository with the expected storage namespace."
                )

        base = lakefs_endpoint.rstrip("/")
        branch_url = (
            f"{base}/api/v1/repositories/{lakefs_repository}/branches/{branch_name}"
        )
        branch_resp = requests.get(
            branch_url, auth=(lakefs_access_key, lakefs_secret_key), timeout=30
        )
        if branch_resp.status_code == 200:
            print(f"Branch {branch_name} already exists. Reusing it.")
            return branch_name
        elif branch_resp.status_code != 404:
            branch_resp.raise_for_status()

        # Branch does not exist, create it
        url = f"{base}/api/v1/repositories/{lakefs_repository}/branches"
        payload = {"name": branch_name, "source": source_branch}
        response = requests.post(
            url, json=payload, auth=(lakefs_access_key, lakefs_secret_key), timeout=30
        )
        if response.status_code == 409:
            print(f"Branch {branch_name} already exists (race condition). Reusing it.")
            return branch_name
        response.raise_for_status()
        print(f"Created branch: {branch_name} from {source_branch}")
        return branch_name
    except Exception:
        traceback.print_exc()
        sys.exit(1)


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "requests==2.32.4",
        "minio==7.2.7",
        "pandas==2.3.1",
        "numpy==2.2.6",
    ],
)
def generate_and_commit_data(
    branch_name: str,
    lakefs_repository: str,
    bucket_name: str,
    object_key: str,
    sample_size: int,
    output_dataset: Output[Dataset],
) -> str:
    """Generate synthetic data and commit it to LakeFS."""
    import io, json, os, traceback, sys
    from datetime import UTC, datetime
    from minio import Minio
    import numpy as np
    import pandas as pd
    import requests

    try:
        lakefs_access_key = os.environ["LAKEFS_ACCESS_KEY"]
        lakefs_secret_key = os.environ["LAKEFS_SECRET_KEY"]
        lakefs_endpoint = os.environ["LAKEFS_ENDPOINT"]
        lakefs_s3_endpoint = lakefs_endpoint.rstrip("/")

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            rng.normal(size=(sample_size, 6)), columns=[f"f{i}" for i in range(6)]
        )
        df["label"] = ((df["f0"] + 0.7 * df["f1"] - 0.4 * df["f2"]) > 0).astype(int)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        output_dataset.path and open(output_dataset.path, "wb").write(csv_bytes)

        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        # Remove .csv extension if present
        base_name = object_key[:-4] if object_key.endswith(".csv") else object_key
        branch_object_key = f"{branch_name}/{base_name}-{ts}.csv"

        endpoint_no_scheme = lakefs_s3_endpoint.replace("https://", "").replace(
            "http://", ""
        )
        s3_client = Minio(
            endpoint_no_scheme,
            access_key=lakefs_access_key,
            secret_key=lakefs_secret_key,
            secure=lakefs_s3_endpoint.startswith("https://"),
        )
        try:
            import io

            s3_client.put_object(
                bucket_name=lakefs_repository,
                object_name=branch_object_key,
                data=io.BytesIO(csv_bytes),
                length=len(csv_bytes),
                content_type="text/csv",
            )
        except Exception:
            print("LakeFS S3 gateway upload failed")
            print("S3 endpoint:", lakefs_s3_endpoint)
            print("Repository:", lakefs_repository)
            print("Branch:", branch_name)
            print("Object key:", branch_object_key)
            raise

        # Commit in lakeFS
        commit_url = f"{lakefs_endpoint.rstrip('/')}/api/v1/repositories/{lakefs_repository}/branches/{branch_name}/commits"
        commit_payload = {
            "message": f"Add generated training data: {branch_object_key}",
            "metadata": {"producer": "kubeflow-pipeline", "bucket": bucket_name},
        }
        commit_resp = requests.post(
            commit_url,
            json=commit_payload,
            auth=(lakefs_access_key, lakefs_secret_key),
            timeout=30,
        )
        commit_resp.raise_for_status()

        body = commit_resp.json()
        commit_id = body.get("id") or body.get("commit_id")
        if not commit_id:
            raise RuntimeError(f"Commit response missing commit id: {json.dumps(body)}")

        print(f"Committed data to {branch_name} at commit {commit_id}")
        print(f"DEBUG: Generated object key: {branch_object_key}")
        print(f"DEBUG: Returning: {branch_object_key}")
        return branch_object_key
    except Exception:
        traceback.print_exc()
        sys.exit(1)


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "requests==2.32.4",
        "minio==7.2.7",
        "pandas==2.3.1",
        "scikit-learn==1.7.1",
        "joblib==1.4.2",
        "mlflow==2.16.1",
        "boto3==1.34.128",
    ],
)
def train_with_mlflow_tracking(
    branch_name: str,
    object_key: str,
    lakefs_repository: str,
    experiment_name: str,
    metrics: Output[Metrics],
    model: Output[Model],
) -> float:
    """Train a model with MLflow experiment tracking."""
    import io, json, os, traceback, sys
    import joblib
    from minio import Minio
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    import mlflow
    import mlflow.sklearn

    try:
        # Debug: print received parameters
        print(f"DEBUG TRAINING: Received parameters:")
        print(f"DEBUG TRAINING: branch_name = {branch_name}")
        print(f"DEBUG TRAINING: object_key = {object_key}")
        print(f"DEBUG TRAINING: lakefs_repository = {lakefs_repository}")
        print(f"DEBUG TRAINING: experiment_name = {experiment_name}")

        # Get environment variables
        lakefs_access_key = os.environ["LAKEFS_ACCESS_KEY"]
        lakefs_secret_key = os.environ["LAKEFS_SECRET_KEY"]
        lakefs_endpoint = os.environ["LAKEFS_ENDPOINT"]

        # Get MinIO credentials for MLflow artifact storage
        minio_access_key = os.environ.get("MINIO_ACCESS_KEY")
        minio_secret_key = os.environ.get("MINIO_SECRET_KEY")

        # MLflow tracking URI (from configmap or secret)
        mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Configure boto3 to use MinIO credentials for S3 artifact storage
        # MLflow uses boto3 internally for S3 operations
        if minio_access_key and minio_secret_key:
            # Set AWS credentials from MinIO credentials
            os.environ["AWS_ACCESS_KEY_ID"] = minio_access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret_key
            print("DEBUG: Set AWS credentials from MinIO credentials")

        # Also configure boto3 to use MinIO endpoint if needed
        # Extract MinIO endpoint from MLflow tracking URI if it's an S3/MinIO URI
        if mlflow_tracking_uri.startswith("http"):
            # This is an HTTP tracking server, artifacts might be in S3/MinIO
            # We need to check if MLflow is configured to use S3/MinIO for artifacts
            print(f"DEBUG: MLflow tracking URI: {mlflow_tracking_uri}")
            print(
                f"DEBUG: AWS_ACCESS_KEY_ID is set: {'AWS_ACCESS_KEY_ID' in os.environ}"
            )
            print(
                f"DEBUG: AWS_SECRET_ACCESS_KEY is set: {'AWS_SECRET_ACCESS_KEY' in os.environ}"
            )

            # Set MLFLOW_S3_ENDPOINT_URL to use MinIO for artifact storage
            # Get MinIO endpoint from configmap (should be available as MINIO_ENDPOINT)
            minio_endpoint = os.environ.get("MINIO_ENDPOINT")
            if minio_endpoint:
                os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_endpoint
                print(f"DEBUG: Set MLFLOW_S3_ENDPOINT_URL to: {minio_endpoint}")
            else:
                print("WARNING: MINIO_ENDPOINT not found in environment variables")
                print("WARNING: MLflow may fail to upload artifacts to MinIO")

        # Load data from LakeFS
        endpoint_no_scheme = (
            lakefs_endpoint.rstrip("/").replace("https://", "").replace("http://", "")
        )
        s3_client = Minio(
            endpoint_no_scheme,
            access_key=lakefs_access_key,
            secret_key=lakefs_secret_key,
            secure=lakefs_endpoint.rstrip("/").startswith("https://"),
        )

        print(f"DEBUG: Trying to load object from LakeFS")
        print(f"DEBUG: Repository/Bucket: {lakefs_repository}")
        print(f"DEBUG: Object key: {object_key}")
        print(f"DEBUG: Full path: {lakefs_repository}/{object_key}")

        response = s3_client.get_object(lakefs_repository, object_key)
        try:
            csv_text = response.read().decode("utf-8")
        finally:
            response.close()
            response.release_conn()

        df = pd.read_csv(io.StringIO(csv_text))
        X = df.drop(columns=["label"])
        y = df["label"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Start MLflow run
        with mlflow.start_run(run_name=f"lakefs-{branch_name}") as run:
            # Log parameters
            mlflow.log_param("branch", branch_name)
            mlflow.log_param("repository", lakefs_repository)
            mlflow.log_param("object_key", object_key)
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 400)

            # Train model
            clf = LogisticRegression(max_iter=400)
            clf.fit(X_train, y_train)

            # Evaluate
            pred = clf.predict(X_test)
            acc = float(accuracy_score(y_test, pred))
            report = classification_report(y_test, pred, output_dict=True)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
            mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])
            mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])

            # Log to Kubeflow metrics
            metrics.log_metric("accuracy", acc)

            # Log model
            mlflow.sklearn.log_model(
                clf,
                "model",
                registered_model_name=f"lakefs-model-{experiment_name}",
                metadata={
                    "framework": "sklearn",
                    "source_branch": branch_name,
                    "source_object": object_key,
                    "run_id": run.info.run_id,
                },
            )

            # Log dataset info as artifact
            dataset_info = {
                "branch": branch_name,
                "repository": lakefs_repository,
                "object_key": object_key,
                "dataset_size": len(df),
                "features": list(X.columns),
                "target": "label",
                "class_distribution": dict(y.value_counts()),
            }
            mlflow.log_dict(dataset_info, "dataset_info.json")

            # Save model locally for Kubeflow output
            joblib.dump(clf, model.path)
            model.metadata["framework"] = "sklearn"
            model.metadata["source_branch"] = branch_name
            model.metadata["source_object"] = object_key
            model.metadata["mlflow_run_id"] = run.info.run_id
            model.metadata["mlflow_experiment_id"] = run.info.experiment_id

            # Print run info
            print(f"MLflow Run ID: {run.info.run_id}")
            print(f"MLflow Experiment ID: {run.info.experiment_id}")
            print(f"Model registered as: lakefs-model-{experiment_name}")
            print(
                json.dumps(
                    {"accuracy": acc, "rows": len(df), "mlflow_run": run.info.run_id},
                    indent=2,
                )
            )

            return acc

    except Exception:
        traceback.print_exc()
        sys.exit(1)


@dsl.pipeline(name="mlflow-integrated-training")
def mlflow_integrated_pipeline(
    source_branch: str,
    feature_branch: str,
    lakefs_repository: str,
    pipeline_secret_name: str,
    pipeline_configmap_name: str,
    bucket_name: str,
    object_key: str,
    sample_size: int,
    min_accuracy: float,
    experiment_name: str = "lakefs-experiments",
) -> None:
    """Complete MLflow-integrated pipeline with data generation."""

    # 1. Create branch in LakeFS
    branch_task = lakefs_create_branch(
        source_branch=source_branch,
        branch_name=feature_branch,
        lakefs_repository=lakefs_repository,
        bucket_name=bucket_name,
    )

    # 2. Generate and commit data to the branch
    data_task = generate_and_commit_data(
        branch_name=branch_task.output,
        lakefs_repository=lakefs_repository,
        bucket_name=bucket_name,
        object_key=object_key,
        sample_size=sample_size,
    )

    # 3. Train model with MLflow tracking
    train_task = train_with_mlflow_tracking(
        branch_name=branch_task.output,
        object_key=data_task.outputs["Output"],
        lakefs_repository=lakefs_repository,
        experiment_name=experiment_name,
    )

    # Configure environment variables and secrets for all tasks
    for task in [branch_task, data_task, train_task]:
        task.set_env_variable("PYTHONFAULTHANDLER", "1")
        kubernetes.use_config_map_as_env(
            task,
            config_map_name=pipeline_configmap_name,
            config_map_key_to_env={
                "LAKEFS_ENDPOINT": "LAKEFS_ENDPOINT",
                "MINIO_ENDPOINT": "MINIO_ENDPOINT",
                "MLFLOW_TRACKING_URI": "MLFLOW_TRACKING_URI",
                "MLFLOW_S3_ENDPOINT_URL": "MLFLOW_S3_ENDPOINT_URL",
            },
        )
        kubernetes.use_secret_as_env(
            task,
            secret_name=pipeline_secret_name,
            secret_key_to_env={
                "LAKEFS_ACCESS_KEY": "LAKEFS_ACCESS_KEY",
                "LAKEFS_SECRET_KEY": "LAKEFS_SECRET_KEY",
                "MINIO_ACCESS_KEY": "MINIO_ACCESS_KEY",
                "MINIO_SECRET_KEY": "MINIO_SECRET_KEY",
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            },
        )


if __name__ == "__main__":
    # Example usage
    pipeline_secret_name = "mlops-credentials"
    pipeline_configmap_name = "pipeline-configmap"
    source_branch = "main"
    lakefs_repository = "ml-data"
    bucket_name = "lakefs-data"
    object_key = "training/data.csv"
    sample_size = 1000
    min_accuracy = 0.80
    experiment_name = "lakefs-kubeflow-integration"

    import uuid

    unique_branch = f"exp-{uuid.uuid4().hex[:8]}"

    client = get_kfp_client()
    run = client.create_run_from_pipeline_func(
        mlflow_integrated_pipeline,
        arguments={
            "source_branch": source_branch,
            "feature_branch": unique_branch,
            "pipeline_secret_name": pipeline_secret_name,
            "pipeline_configmap_name": pipeline_configmap_name,
            "bucket_name": bucket_name,
            "object_key": object_key,
            "sample_size": sample_size,
            "min_accuracy": min_accuracy,
            "experiment_name": experiment_name,
            "lakefs_repository": lakefs_repository,
        },
        experiment_name="MLflow Integration Demo",
    )

    print(f"Pipeline run created: {run.run_id}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"Check MLflow UI at: http://localhost:5000")
