from importlib import metadata
from kfp import dsl
from typing import NamedTuple

_BASE_IMAGE = "ml-components/base:latest"
_PACKAGE_NAME = "ml-components-feature-engineering"
_TARGET_IMAGE = f"ml-components/feature-engineering:{metadata.version('ml-components-feature-engineering')}"


@dsl.component(base_image=_BASE_IMAGE, target_image=_TARGET_IMAGE)
def feature_engineering(
    dataset_path: str,
) -> NamedTuple("Outputs", [("features_path", str)]):
    """Apply StandardScaler to numeric columns and write normalized Parquet to MinIO."""
    import os
    from io import BytesIO
    import boto3
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )
    path = dataset_path.removeprefix("s3://")
    bucket, key = path.split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))

    # Exclude the 'label' column from scaling — it must remain as discrete class labels
    numeric_cols = [c for c in df.select_dtypes("number").columns if c != "label"]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Write to datasets bucket under features/ prefix
    features_key = key.replace("data.parquet", "features.parquet")
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=features_key, Body=buf.getvalue())

    features_path = f"s3://{bucket}/{features_key}"
    from collections import namedtuple

    Outputs = namedtuple("Outputs", ["features_path"])
    return Outputs(features_path=features_path)
