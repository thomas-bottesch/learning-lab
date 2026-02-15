from kfp import dsl
from kfp import kubernetes
from kfp.dsl import Artifact
from kfp.dsl import Input
from kfp.dsl import Output

from kfp_client_utils import get_kfp_client


@dsl.component(
    base_image="python:3.12", packages_to_install=["dvc[s3]", "numpy", "pandas"]
)
def dvc_prepare(repo_tar: Output[Artifact]) -> None:
    import os
    import subprocess
    import tarfile

    import numpy as np
    import pandas as pd

    workdir = "/tmp/dvc-repo"
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    subprocess.run(["dvc", "init", "--no-scm"], check=True)
    subprocess.run(["dvc", "config", "core.analytics", "false"], check=True)
    minio_bucket = os.environ["DVC_BUCKET_NAME"]
    subprocess.run(
        ["dvc", "remote", "add", "-d", "minio", f"s3://{minio_bucket}/demo"],
        check=True,
    )
    subprocess.run(
        [
            "dvc",
            "remote",
            "modify",
            "minio",
            "endpointurl",
            os.environ["MINIO_ENDPOINT"],
        ],
        check=True,
    )

    df = pd.DataFrame(np.random.rand(200, 10), columns=[f"f{i}" for i in range(10)])
    df["label"] = (df["f0"] + df["f1"] > 1.0).astype(int)
    df.to_csv("data.csv", index=False)

    subprocess.run(["dvc", "add", "data.csv"], check=True)
    subprocess.run(["dvc", "push"], check=True)

    with tarfile.open(repo_tar.path, "w:gz") as tar:
        tar.add(workdir, arcname="repo")


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["dvc[s3]", "numpy", "pandas", "scikit-learn"],
)
def dvc_train(repo_tar: Input[Artifact]) -> None:
    import json
    import os
    import subprocess
    import tarfile

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    workdir = "/tmp"
    os.makedirs(workdir, exist_ok=True)

    with tarfile.open(repo_tar.path, "r:gz") as tar:
        tar.extractall(workdir)

    candidate_dirs = [
        os.path.join(workdir, "repo"),
        os.path.join(workdir, "dvc-repo"),
    ]
    for candidate in candidate_dirs:
        if os.path.isdir(os.path.join(candidate, ".dvc")):
            workdir = candidate
            break
    else:
        raise FileNotFoundError(
            f"Could not find extracted DVC repo. Checked: {candidate_dirs}"
        )

    os.chdir(workdir)
    subprocess.run(
        [
            "dvc",
            "remote",
            "modify",
            "minio",
            "endpointurl",
            os.environ["MINIO_ENDPOINT"],
        ],
        check=True,
    )
    subprocess.run(["dvc", "pull"], check=True)

    df = pd.read_csv("data.csv")
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    metrics = {"accuracy": acc}
    print(json.dumps(metrics, indent=2))


@dsl.pipeline(name="dvc-minio-demo")
def pipeline(
    secret_name: str,
    pipeline_configmap_name: str,
) -> None:
    minio_env = {
        "AWS_DEFAULT_REGION": "us-east-1",
        "PIP_ROOT_USER_ACTION": "ignore",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }

    prep = dvc_prepare()
    train = dvc_train(repo_tar=prep.outputs["repo_tar"])

    for task in [prep, train]:
        for key, value in minio_env.items():
            task.set_env_variable(key, value)
        kubernetes.use_config_map_as_env(
            task,
            config_map_name=pipeline_configmap_name,
            config_map_key_to_env={
                "MINIO_ENDPOINT": "MINIO_ENDPOINT",
                "DVC_BUCKET_NAME": "DVC_BUCKET_NAME",
            },
        )
        kubernetes.use_secret_as_env(
            task,
            secret_name=secret_name,
            secret_key_to_env={
                "MINIO_ACCESS_KEY": "AWS_ACCESS_KEY_ID",
                "MINIO_SECRET_KEY": "AWS_SECRET_ACCESS_KEY",
            },
        )


if __name__ == "__main__":
    try:
        import os
        import tempfile
        import warnings
        from datetime import UTC, datetime
        from kfp.compiler import Compiler

        def _first_attr(obj: object, names: list[str]) -> str | None:
            for name in names:
                value = getattr(obj, name, None)
                if value:
                    return value
            return None

        warnings.filterwarnings(
            "ignore",
            message=(
                "This client only works with Kubeflow Pipeline v2.0.0-beta.2 "
                "and later versions."
            ),
            category=FutureWarning,
        )

        client = get_kfp_client()
        pipeline_name = "dvc-minio-demo"
        experiment_name = "dvc-minio-experiment"
        secret_name = "mlops-credentials"
        pipeline_configmap_name = "pipeline-configmap"

        version_name = f"{pipeline_name}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            Compiler().compile(pipeline, tmp.name)
            pipeline_package_path = tmp.name

        # Ensure pipeline exists and create a fresh version for this run.
        try:
            pipeline_id = client.get_pipeline_id(pipeline_name)
            if pipeline_id is None:
                pipeline_obj = client.upload_pipeline(
                    pipeline_package_path, pipeline_name=pipeline_name
                )
                pipeline_id = _first_attr(pipeline_obj, ["pipeline_id", "id"])

            if not pipeline_id:
                raise RuntimeError("Could not determine pipeline_id after upload.")

            version_obj = client.upload_pipeline_version(
                pipeline_package_path=pipeline_package_path,
                pipeline_version_name=version_name,
                pipeline_id=pipeline_id,
            )
            version_id = _first_attr(
                version_obj,
                ["pipeline_version_id", "version_id", "id"],
            )
            if not version_id:
                raise RuntimeError(
                    "Could not determine pipeline version_id after upload."
                )
        finally:
            os.remove(pipeline_package_path)

        # Create or get experiment, then submit run.
        experiment = client.create_experiment(name=experiment_name)
        exp_id = _first_attr(experiment, ["experiment_id", "id"])
        if not exp_id:
            raise RuntimeError(f"Could not determine experiment id from: {experiment}")

        run_name = f"{pipeline_name}-run-{datetime.now(UTC).strftime('%H%M%S')}"

        run = client.run_pipeline(
            experiment_id=exp_id,
            job_name=run_name,
            pipeline_id=pipeline_id,
            version_id=version_id,
            params={
                "secret_name": secret_name,
                "pipeline_configmap_name": pipeline_configmap_name,
            },
        )
        run_id = _first_attr(run, ["run_id", "id"])
        print(f"Pipeline submitted successfully. Run ID: {run_id or 'unknown'}")
    except Exception as err:
        message = str(err)
        if (
            "401" in message
            or "Unauthorized" in message
            or "User identity is empty" in message
        ):
            print("Authentication failed for Kubeflow Pipelines.")
        raise
