#!/usr/bin/env python3
"""Seed initial training data into MinIO and LakeFS.

Run once before the first pipeline execution, or let submit.yaml run it automatically.
Idempotent — skips upload if the dataset already exists at the expected path.

All credentials are required environment variables — there are no defaults.
For local development, populate them from Kubernetes YAML sources:

    eval "$(bash scripts/get_local_env.sh)"
    python scripts/seed_data.py

For CI, credentials are injected automatically from Forgejo secrets
(set by create_forgejo_repo.sh at repo creation time).
"""
import os
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import make_classification


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set.\n"
            "For local development, run:\n"
            "    eval \"$(bash scripts/get_local_env.sh)\"\n"
            "For CI, ensure create_forgejo_repo.sh ran before this job."
        )
    return value


def ensure_bucket(s3, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
        print(f"Created MinIO bucket: {bucket}")


def ensure_lakefs_repo(endpoint: str, auth: tuple, lakefs_repo: str, branch: str) -> str:
    """Create LakeFS repo if missing; return current commit_id for branch."""
    resp = requests.get(
        f"{endpoint}/api/v1/repositories/{lakefs_repo}",
        auth=auth,
        timeout=30,
    )
    if resp.status_code == 404:
        create = requests.post(
            f"{endpoint}/api/v1/repositories",
            auth=auth,
            json={
                "name": lakefs_repo,
                "storage_namespace": f"s3://lakefs/{lakefs_repo}/",
                "default_branch": branch,
            },
            timeout=30,
        )
        create.raise_for_status()
        print(f"Created LakeFS repo: {lakefs_repo}")
    else:
        resp.raise_for_status()

    branch_resp = requests.get(
        f"{endpoint}/api/v1/repositories/{lakefs_repo}/branches/{branch}",
        auth=auth,
        timeout=30,
    )
    branch_resp.raise_for_status()
    return branch_resp.json()["commit_id"]


def seed_dataset(s3, lakefs_repo: str, commit_id: str) -> str:
    """Upload sklearn dataset to MinIO. Returns the dataset_path string."""
    bucket = "datasets"
    key = f"{lakefs_repo}/{commit_id}/data.parquet"
    dataset_path = f"s3://{bucket}/{key}"

    try:
        s3.head_object(Bucket=bucket, Key=key)
        print(f"Dataset already exists at {dataset_path} — skipping upload.")
        return dataset_path
    except Exception:
        pass

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["label"] = y.astype(np.int32)

    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"Seeded {len(df)} rows ({len(feature_cols)} features + label) → {dataset_path}")
    return dataset_path


def main():
    # Secrets — required, no defaults. Raises immediately if any are missing.
    lakefs_access_key = _require("LAKEFS_ACCESS_KEY")
    lakefs_secret_key = _require("LAKEFS_SECRET_KEY")
    minio_access_key  = _require("MINIO_ACCESS_KEY")
    minio_secret_key  = _require("MINIO_SECRET_KEY")

    # Configuration — non-sensitive, safe to have operational defaults.
    lakefs_endpoint = os.environ.get("LAKEFS_ENDPOINT", "http://localhost:8000")
    minio_endpoint  = os.environ.get("MINIO_ENDPOINT",  "http://localhost:9000")
    lakefs_repo     = os.environ.get("LAKEFS_REPO", "mlops-data-dev")
    branch          = os.environ.get("BRANCH", "main")

    auth = (lakefs_access_key, lakefs_secret_key)
    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
    )

    ensure_bucket(s3, "datasets")
    ensure_bucket(s3, "models")

    commit_id = ensure_lakefs_repo(lakefs_endpoint, auth, lakefs_repo, branch)
    print(f"LakeFS repo={lakefs_repo}, branch={branch}, commit={commit_id}")

    seed_dataset(s3, lakefs_repo, commit_id)
    print("Seed complete. Pipelines can now run without manual data setup.")


if __name__ == "__main__":
    main()
