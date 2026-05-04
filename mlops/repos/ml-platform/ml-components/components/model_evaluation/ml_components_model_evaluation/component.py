from importlib import metadata
from kfp import dsl
from typing import NamedTuple

_BASE_IMAGE = "ml-components/base:latest"
_PACKAGE_NAME = "ml-components-model-evaluation"
_TARGET_IMAGE = f"ml-components/model-evaluation:{metadata.version('ml-components-model-evaluation')}"


@dsl.component(base_image=_BASE_IMAGE, target_image=_TARGET_IMAGE)
def model_evaluation(
    model_artifact: str,
    features_path: str,
    min_accuracy: float = 0.80,
) -> NamedTuple("Outputs", [("evaluation_report", str), ("accuracy", float)]):
    """Load model and features, compute accuracy, return pass/fail report."""
    import json
    import os
    import pickle
    from io import BytesIO
    import boto3
    import pandas as pd
    from sklearn.model_selection import train_test_split

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )

    def _read(uri: str) -> bytes:
        path = uri.removeprefix("s3://")
        bucket, key = path.split("/", 1)
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    model = pickle.loads(_read(model_artifact))
    df = pd.read_parquet(BytesIO(_read(features_path)))

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    accuracy = float(model.score(X_test, y_test))
    status = "passed" if accuracy >= min_accuracy else "failed"

    if status == "failed":
        raise ValueError(
            f"Model accuracy {accuracy:.3f} below threshold {min_accuracy:.3f}"
        )

    report = json.dumps(
        {
            "status": status,
            "accuracy": accuracy,
            "min_accuracy": min_accuracy,
            "model_artifact": model_artifact,
        }
    )
    from collections import namedtuple

    Outputs = namedtuple("Outputs", ["evaluation_report", "accuracy"])
    return Outputs(evaluation_report=report, accuracy=accuracy)
