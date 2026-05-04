from importlib import metadata
from kfp import dsl
from typing import NamedTuple

_BASE_IMAGE = "ml-components/base:latest"
_PACKAGE_NAME = "ml-components-model-training"
_TARGET_IMAGE = (
    f"ml-components/model-training:{metadata.version('ml-components-model-training')}"
)


@dsl.component(base_image=_BASE_IMAGE, target_image=_TARGET_IMAGE)
def model_training(
    features_path: str,
    mlflow_run_name: str = "tron-training",
    epochs: int = 10,
    max_iter: int = 200,
) -> NamedTuple("Outputs", [("model_artifact", str), ("mlflow_run_id", str)]):
    """Train a LogisticRegression on the feature set and log everything to MLflow."""
    import os
    import pickle
    from io import BytesIO
    import boto3
    import mlflow
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("tron-training")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )
    path = features_path.removeprefix("s3://")
    bucket, key = path.split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))

    # Assume last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        mlflow.log_params({"epochs": epochs, "max_iter": max_iter})
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("train_accuracy", accuracy)

        # Save model to MinIO models bucket
        model_key = f"tron/{run.info.run_id}/model.pkl"
        model_bytes = pickle.dumps(model)
        s3.put_object(Bucket="models", Key=model_key, Body=model_bytes)

        model_artifact = f"s3://models/{model_key}"
        mlflow.log_param("model_artifact", model_artifact)
        run_id = run.info.run_id

    from collections import namedtuple

    Outputs = namedtuple("Outputs", ["model_artifact", "mlflow_run_id"])
    return Outputs(model_artifact=model_artifact, mlflow_run_id=run_id)
