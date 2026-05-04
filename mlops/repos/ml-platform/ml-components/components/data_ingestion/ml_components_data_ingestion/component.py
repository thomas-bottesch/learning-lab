from importlib import metadata
from kfp import dsl
from typing import NamedTuple

_BASE_IMAGE = "ml-components/base:latest"
_PACKAGE_NAME = "ml-components-data-ingestion"
_TARGET_IMAGE = (
    f"ml-components/data-ingestion:{metadata.version('ml-components-data-ingestion')}"
)


@dsl.component(base_image=_BASE_IMAGE, target_image=_TARGET_IMAGE)
def data_ingestion(
    lakefs_repo: str,
    branch: str,
) -> NamedTuple("Outputs", [("dataset_path", str), ("lakefs_commit", str)]):
    """Pull a dataset from LakeFS at a pinned commit and return its MinIO path."""
    import os
    import requests
    from collections import namedtuple

    endpoint = os.environ["LAKEFS_ENDPOINT"]
    access_key = os.environ["LAKEFS_ACCESS_KEY"]
    secret_key = os.environ["LAKEFS_SECRET_KEY"]

    # Resolve commit BEFORE reading — prevents TOCTOU race if branch advances.
    resp = requests.get(
        f"{endpoint}/api/v1/repositories/{lakefs_repo}/branches/{branch}",
        auth=(access_key, secret_key),
        timeout=30,
    )
    resp.raise_for_status()
    commit = resp.json()["commit_id"]

    # Dataset lives at a stable S3 path keyed by commit hash.
    dataset_path = f"s3://datasets/{lakefs_repo}/{commit}/data.parquet"
    Outputs = namedtuple("Outputs", ["dataset_path", "lakefs_commit"])
    return Outputs(dataset_path=dataset_path, lakefs_commit=commit)
