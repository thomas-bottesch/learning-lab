from importlib import metadata
from kfp import dsl
from typing import NamedTuple

_BASE_IMAGE = "ml-components/base:latest"
_PACKAGE_NAME = "ml-components-data-validation"
_TARGET_IMAGE = (
    f"ml-components/data-validation:{metadata.version('ml-components-data-validation')}"
)


@dsl.component(base_image=_BASE_IMAGE, target_image=_TARGET_IMAGE)
def data_validation(
    dataset_path: str,
    lakefs_commit: str,
    min_rows: int = 1000,
    max_null_fraction: float = 0.05,
) -> NamedTuple("Outputs", [("dataset_path", str), ("validation_report", str)]):
    """Validate dataset quality. Raises on failure — aborts all downstream steps."""
    import json
    import os
    import boto3
    import pandas as pd
    from io import BytesIO

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )
    # Parse s3://bucket/key from dataset_path
    path = dataset_path.removeprefix("s3://")
    bucket, key = path.split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))

    errors = []
    if len(df) < min_rows:
        errors.append(f"row count {len(df)} below minimum {min_rows}")
    for col, frac in df.isnull().mean().items():
        if frac > max_null_fraction:
            errors.append(
                f"column '{col}' has {frac:.1%} nulls (limit {max_null_fraction:.1%})"
            )
    for col in df.select_dtypes("number").columns:
        if (df[col] == 0).all():
            errors.append(f"column '{col}' is all zeros")

    if errors:
        raise ValueError(
            "Data validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    report = json.dumps(
        {
            "status": "passed",
            "rows": len(df),
            "columns": list(df.columns),
            "lakefs_commit": lakefs_commit,
        }
    )
    from collections import namedtuple

    Outputs = namedtuple("Outputs", ["dataset_path", "validation_report"])
    return Outputs(dataset_path=dataset_path, validation_report=report)
