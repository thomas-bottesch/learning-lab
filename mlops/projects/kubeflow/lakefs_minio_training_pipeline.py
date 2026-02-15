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
        branch_object_key = f"{branch_name}/{object_key.rstrip('.csv')}-{ts}.csv"

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
    ],
)
def train_on_lakefs_data(
    branch_name: str,
    object_key: str,
    lakefs_repository: str,
    metrics: Output[Metrics],
    model: Output[Model],
) -> float:
    import io, json, os, traceback, sys
    import joblib
    from minio import Minio
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    try:
        lakefs_access_key = os.environ["LAKEFS_ACCESS_KEY"]
        lakefs_secret_key = os.environ["LAKEFS_SECRET_KEY"]
        lakefs_endpoint = os.environ["LAKEFS_ENDPOINT"]

        endpoint_no_scheme = (
            lakefs_endpoint.rstrip("/").replace("https://", "").replace("http://", "")
        )
        s3_client = Minio(
            endpoint_no_scheme,
            access_key=lakefs_access_key,
            secret_key=lakefs_secret_key,
            secure=lakefs_endpoint.rstrip("/").startswith("https://"),
        )
        response = s3_client.get_object(lakefs_repository, object_key)
        try:
            csv_text = response.read().decode("utf-8")
        finally:
            response.close()
            response.release_conn()

        df = pd.read_csv(io.StringIO(csv_text))
        X = df.drop(columns=["label"])
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = LogisticRegression(max_iter=400)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, pred))

        metrics.log_metric("accuracy", acc)

        model.metadata["framework"] = "sklearn"
        model.metadata["source_branch"] = branch_name
        model.metadata["source_object"] = object_key

        joblib.dump(clf, model.path)
        print(json.dumps({"accuracy": acc, "rows": len(df)}, indent=2))
        return acc
    except Exception:
        traceback.print_exc()
        sys.exit(1)


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["requests==2.32.4"],
)
def maybe_promote_branch(
    source_branch: str,
    target_branch: str,
    accuracy: float,
    min_accuracy: float,
    lakefs_repository: str,
) -> str:
    import os, requests, traceback, sys

    try:
        lakefs_access_key = os.environ["LAKEFS_ACCESS_KEY"]
        lakefs_secret_key = os.environ["LAKEFS_SECRET_KEY"]
        lakefs_endpoint = os.environ["LAKEFS_ENDPOINT"]

        if accuracy < min_accuracy:
            msg = f"Skipping merge: accuracy {accuracy:.4f} below threshold {min_accuracy:.4f}."
            print(msg)
            return msg

        url = f"{lakefs_endpoint.rstrip('/')}/api/v1/repositories/{lakefs_repository}/refs/{target_branch}/merge/{source_branch}"
        response = requests.post(
            url, auth=(lakefs_access_key, lakefs_secret_key), timeout=30
        )

        if response.status_code in (200, 201):
            msg = f"Merged {source_branch} into {target_branch}"
            print(msg)
            return msg

        if response.status_code in (400, 409):
            detail = response.text
            try:
                payload = response.json()
                detail = payload.get("message") or payload.get("error") or detail
            except Exception:
                pass

            detail_lc = str(detail).lower()
            if "no changes" in detail_lc:
                msg = (
                    f"No-op merge: {source_branch} has no changes to merge into "
                    f"{target_branch}."
                )
                print(msg)
                return msg

            msg = (
                f"Skipping merge: lakeFS reported merge precondition/conflict "
                f"(status={response.status_code}). details={detail}"
            )
            print(msg)
            return msg

        response.raise_for_status()
        msg = f"Merged {source_branch} into {target_branch}"
        print(msg)
        return msg
    except Exception:
        traceback.print_exc()
        sys.exit(1)


@dsl.pipeline(name="lakefs-minio-kubeflow-training")
def lakefs_minio_training_pipeline(
    source_branch: str,
    feature_branch: str,
    lakefs_repository: str,
    pipeline_secret_name: str,
    pipeline_configmap_name: str,
    bucket_name: str,
    object_key: str,
    sample_size: int,
    min_accuracy: float,
) -> None:
    branch = lakefs_create_branch(
        source_branch=source_branch,
        branch_name=feature_branch,
        lakefs_repository=lakefs_repository,
        bucket_name=bucket_name,
    )

    prepared = generate_and_commit_data(
        branch_name=branch.output,
        lakefs_repository=lakefs_repository,
        bucket_name=bucket_name,
        object_key=object_key,
        sample_size=sample_size,
    )

    trained = train_on_lakefs_data(
        branch_name=branch.output,
        object_key=prepared.outputs["Output"],
        lakefs_repository=lakefs_repository,
    )

    promoted = maybe_promote_branch(
        source_branch=branch.output,
        target_branch=source_branch,
        accuracy=trained.outputs["Output"],
        min_accuracy=min_accuracy,
        lakefs_repository=lakefs_repository,
    )

    for task in [branch, prepared, trained, promoted]:
        task.set_env_variable("PYTHONFAULTHANDLER", "1")
        kubernetes.use_config_map_as_env(
            task,
            config_map_name=pipeline_configmap_name,
            config_map_key_to_env={
                "LAKEFS_ENDPOINT": "LAKEFS_ENDPOINT",
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
            },
        )


if __name__ == "__main__":
    pipeline_secret_name = "mlops-credentials"
    pipeline_configmap_name = "pipeline-configmap"
    source_branch = "main"
    lakefs_repository = "ml-data"
    bucket_name = "lakefs-data"
    object_key = "training/data.csv"
    sample_size = 1000
    min_accuracy = 0.80

    import uuid

    unique_branch = f"exp-{uuid.uuid4().hex[:8]}"

    client = get_kfp_client()
    run = client.create_run_from_pipeline_func(
        lakefs_minio_training_pipeline,
        arguments={
            "source_branch": source_branch,
            "feature_branch": unique_branch,
            "pipeline_secret_name": pipeline_secret_name,
            "pipeline_configmap_name": pipeline_configmap_name,
            "lakefs_repository": lakefs_repository,
            "bucket_name": bucket_name,
            "object_key": object_key,
            "sample_size": sample_size,
            "min_accuracy": min_accuracy,
        },
        run_name="lakefs-minio-training-run",
        experiment_name="lakefs-minio-experiment",
    )
    run_id = getattr(run, "run_id", getattr(run, "id", None))
    print(f"Pipeline submitted. Run ID: {run_id or 'unknown'}")
