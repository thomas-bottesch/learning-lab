# Task: Implement MLOps Repo Strategy — Team Tron

## Goal

Create two Forgejo repositories implementing the two-tier strategy from `GIT_REPO_STRATEGY.md`:

1. **`ml-platform/ml-components`** — per-component package monorepo with 5 reusable Kubeflow pipeline components
2. **`team-tron/pipelines`** — team repo with 3 KFP pipelines that consume all 5 components

Write all source files under `repos/` relative to the `mlops/` directory, then push to Forgejo using the provided shell scripts. All commands run from `/workspace/git_repo/mlops/`.

---

## Networking Rules (commit to memory before writing any file)

| Context | Use |
|---------|-----|
| From a Kubeflow pipeline Pod (inside cluster) | `<service>.<namespace>.svc.cluster.local:<port>` |
| From a Forgejo CI runner container (Docker on host) | `172.17.0.1:<port>` |
| From the devcontainer / host (your shell) | `localhost:<port>` |

Concretely:
- `MLFLOW_TRACKING_URI` in pipeline Pods → `http://mlflow.mlflow.svc.cluster.local:5000`
- ZOT registry in CI runner → `172.17.0.1:8001`
- Forgejo PyPI in CI runner → `http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/`
- Kubeflow API in CI runner → `http://172.17.0.1:8080`

---

## What to Build

### 5 Components (all in `ml-platform/ml-components`)

Each component is an **independent installable package** with its own `pyproject.toml`, published separately to Forgejo PyPI.

| snake_case dir | Package name | KFP function name | Role |
|----------------|-------------|------------------|------|
| `data_ingestion` | `ml-components-data-ingestion` | `data_ingestion` | Resolve LakeFS commit, return dataset path + commit hash |
| `data_validation` | `ml-components-data-validation` | `data_validation` | Gate: row count, null fraction, all-zero checks |
| `feature_engineering` | `ml-components-feature-engineering` | `feature_engineering` | StandardScaler normalization (excluding label column), write Parquet to MinIO |
| `model_training` | `ml-components-model-training` | `model_training` | Train LogisticRegression, log to MLflow, save artifact |
| `model_evaluation` | `ml-components-model-evaluation` | `model_evaluation` | Compute accuracy, return pass/fail evaluation report |

### 3 Pipelines (all in `team-tron/pipelines`)

| File | Purpose | Components used |
|------|---------|-----------------|
| `training_pipeline.py` | Full supervised training run | all 5 |
| `retraining_pipeline.py` | Hyperparameter-tunable retrain | all 5 |
| `evaluation_pipeline.py` | Quick data + model quality check | all 5 (model_training with epochs=1) |

All 5 components are used across the 3 pipelines. Pipelines differ in their parameter defaults and semantic purpose, not in their DAG shape.

---

## Component Function Signatures

These are the exact signatures the component functions must have. The pipeline wiring depends on the output names being exactly as shown.

```python
# data_ingestion
def data_ingestion(
    lakefs_repo: str,
    branch: str,
) -> NamedTuple("Outputs", [("dataset_path", str), ("lakefs_commit", str)]):
    ...

# data_validation
def data_validation(
    dataset_path: str,
    lakefs_commit: str,
    min_rows: int = 1000,
    max_null_fraction: float = 0.05,
) -> NamedTuple("Outputs", [("dataset_path", str), ("validation_report", str)]):
    ...

# feature_engineering
def feature_engineering(
    dataset_path: str,
) -> NamedTuple("Outputs", [("features_path", str)]):
    ...

# model_training
def model_training(
    features_path: str,
    mlflow_run_name: str = "tron-training",
    epochs: int = 10,
    max_iter: int = 200,
) -> NamedTuple("Outputs", [("model_artifact", str), ("mlflow_run_id", str)]):
    ...

# model_evaluation
def model_evaluation(
    model_artifact: str,
    features_path: str,
    min_accuracy: float = 0.80,
) -> NamedTuple("Outputs", [("evaluation_report", str), ("accuracy", float)]):
    ...
```

---

## Complete Directory Structure to Create

```
repos/
├── ml-platform/
│   └── ml-components/
│       ├── components/
│       │   ├── data_ingestion/
│       │   │   ├── manifest.yaml
│       │   │   ├── pyproject.toml
│       │   │   ├── runtime-requirements.txt
│       │   │   ├── ml_components_data_ingestion/
│       │   │   │   ├── __init__.py
│       │   │   │   └── component.py
│       │   │   └── tests/
│       │   │       └── test_data_ingestion.py
│       │   ├── data_validation/
│       │   │   ├── manifest.yaml
│       │   │   ├── pyproject.toml
│       │   │   ├── runtime-requirements.txt
│       │   │   ├── ml_components_data_validation/
│       │   │   │   ├── __init__.py
│       │   │   │   └── component.py
│       │   │   └── tests/
│       │   │       └── test_data_validation.py
│       │   ├── feature_engineering/
│       │   │   ├── manifest.yaml
│       │   │   ├── pyproject.toml
│       │   │   ├── runtime-requirements.txt
│       │   │   ├── ml_components_feature_engineering/
│       │   │   │   ├── __init__.py
│       │   │   │   └── component.py
│       │   │   └── tests/
│       │   │       └── test_feature_engineering.py
│       │   ├── model_training/
│       │   │   ├── manifest.yaml
│       │   │   ├── pyproject.toml
│       │   │   ├── runtime-requirements.txt
│       │   │   ├── ml_components_model_training/
│       │   │   │   ├── __init__.py
│       │   │   │   └── component.py
│       │   │   └── tests/
│       │   │       └── test_model_training.py
│       │   └── model_evaluation/
│       │       ├── manifest.yaml
│       │       ├── pyproject.toml
│       │       ├── runtime-requirements.txt
│       │       ├── ml_components_model_evaluation/
│       │       │   ├── __init__.py
│       │       │   └── component.py
│       │       └── tests/
│       │           └── test_model_evaluation.py
│       ├── docker/
│       │   └── Dockerfile.base
│       ├── schemas/
│       │   └── manifest-schema.json
│       ├── scripts/
│       │   ├── new_component.sh
│       │   ├── generate_catalog.py
│       │   └── compile_component.py
│       ├── catalog_patterns.json
│       ├── .forgejo/
│       │   └── workflows/
│       │       └── publish.yaml
│       ├── README.md
│       └── AGENTS.md
└── team-tron/
    └── pipelines/
        ├── pipelines/
        │   ├── __init__.py                          # empty
        │   ├── training_pipeline.py
        │   ├── retraining_pipeline.py
        │   └── evaluation_pipeline.py
        ├── config/
        │   ├── dev.yaml
        │   └── prod.yaml
        ├── scripts/
        │   ├── get_local_env.sh
        │   └── seed_data.py
        ├── kfp_utils/
        │   ├── __init__.py                          # empty
        │   └── auth.py
        ├── tests/
        │   └── test_pipeline_compile.py
        ├── .forgejo/
        │   └── workflows/
        │       ├── ci.yaml
        │       └── submit.yaml
        ├── pyproject.toml
        ├── uv.lock                                  # generated by `uv lock`
        ├── .env.local.example
        └── run_local.py
```

---

## File Contents: ml-platform/ml-components

### `docker/Dockerfile.base`

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir \
    "kfp>=2.0" \
    "pandas>=2.0" \
    "scikit-learn>=1.3" \
    "mlflow>=2.16" \
    "boto3>=1.34" \
    "requests>=2.31" \
    "pyarrow>=14.0"
```

### `catalog_patterns.json`

```json
{
  "composition_patterns": [
    {
      "name": "standard-training-pipeline",
      "description": "Full supervised learning pipeline for team tron: ingest → validate → engineer → train → evaluate.",
      "sequence": ["data-ingestion", "data-validation", "feature-engineering", "model-training", "model-evaluation"]
    },
    {
      "name": "quick-evaluation-pipeline",
      "description": "Fast quality check: runs the full DAG with minimal epochs to validate data and model health.",
      "sequence": ["data-ingestion", "data-validation", "feature-engineering", "model-training", "model-evaluation"]
    }
  ]
}
```

### `AGENTS.md`

```markdown
# ml-components Agent Reference

## Quickstart

Pull the catalog to discover available components before writing any pipeline code:

    oras pull --plain-http 172.17.0.1:8001/ml-components/catalog:latest

The catalog JSON contains every published component's description, tags, inputs, outputs, and `typical_downstream` links. Use it to find existing components before authoring new ones.

## Conventions

- **Schema notation** on inputs/outputs: `s3://<bucket>/<path>` — the literal bucket name matches the MinIO bucket created by `install_minio.sh`.
- **`typical_upstream` / `typical_downstream`**: the expected DAG neighbors. Deviating is allowed when justified; document the reason in the PR.
- **Tags**: lowercase, hyphenated. Use existing tags from the catalog before inventing new ones.
- **`type: Dataset`** means a string URI pointing to a Parquet file or directory on MinIO. It is not a KFP `Artifact` type — it is a plain `str` that carries a path.

## Component Pipeline: standard order

    data-ingestion → data-validation → feature-engineering → model-training → model-evaluation

`data-validation` is the hard gate. If it raises, downstream steps do not run.

## Authoring a New Component

1. Pull catalog and check — does an existing component cover this need?
2. If not: `bash scripts/new_component.sh <name> <category>`
3. Implement `component.py` — no Dockerfile, no component.yaml. CI generates both.
4. Fill every field in `manifest.yaml` — blank fields fail CI validation.
5. Write at least one pytest test that calls the function directly (no Docker).
6. Bump `pyproject.toml` version per bump rules (MINOR for new component).
7. Open PR. Do not push directly to main.

## Version Bump Rules

Each component has its own `components/<name>/pyproject.toml` with an independent version. Bump versions per component:

| Change | Bump |
|--------|------|
| New component | MINOR (for that component) |
| Input/output interface change | MINOR (for that component) |
| Change to `components/<name>/` code | MINOR (for that component) |
| Bug fix, interface unchanged | PATCH (for that component) |
| Breaking interface change | MAJOR (for that component) |

> **Key difference from v1**: Updating one component no longer affects others. Team Tron can bump `ml-components-model-training` to 2.0.0 while `ml-components-data-ingestion` stays at 1.0.0. Pipeline cache is preserved for unchanged components.
```

### `schemas/manifest-schema.json`

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["name", "version", "category", "description", "inputs", "outputs", "typical_upstream", "typical_downstream", "tags"],
  "additionalProperties": true,
  "properties": {
    "name": { "type": "string", "pattern": "^[a-z][a-z0-9-]*$" },
    "version": { "type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$" },
    "category": { "type": "string", "minLength": 1 },
    "description": { "type": "string", "minLength": 20 },
    "inputs": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "type", "description"],
        "properties": {
          "name": { "type": "string" },
          "type": { "type": "string" },
          "description": { "type": "string" },
          "schema": { "type": "string" },
          "default": {}
        }
      }
    },
    "outputs": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["name", "type", "description", "schema"],
        "properties": {
          "name": { "type": "string" },
          "type": { "type": "string" },
          "description": { "type": "string" },
          "schema": { "type": "string" }
        }
      }
    },
    "typical_upstream": {
      "type": "array",
      "minItems": 1,
      "items": { "type": "string" }
    },
    "typical_downstream": {
      "type": "array",
      "items": { "type": "string" }
    },
    "resource_profile": {
      "type": "object",
      "properties": {
        "cpu": { "type": "string" },
        "memory": { "type": "string" }
      }
    },
    "tags": {
      "type": "array",
      "minItems": 1,
      "items": { "type": "string" }
    }
  }
}
```

### `scripts/new_component.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
NAME="$1"
CATEGORY="${2:-general}"
DIR="components/${NAME}"
mkdir -p "${DIR}/tests"
touch "${DIR}/__init__.py"
cat > "${DIR}/component.py" <<PYEOF
from pathlib import Path
import yaml
from kfp import dsl
from typing import NamedTuple

_BASE_IMAGE = "ml-components/base:latest"
_manifest = yaml.safe_load((Path(__file__).parent / "manifest.yaml").read_text())
_TARGET_IMAGE = f"ml-components/{_manifest['name']}:{_manifest['version']}"
_PACKAGE_NAME = "ml-components-$(echo "$NAME" | tr '_' '-')"

@dsl.component(base_image=_BASE_IMAGE, target_image=_TARGET_IMAGE)
def ${NAME}() -> str:
    raise NotImplementedError("Implement ${NAME}")
PYEOF
cat > "${DIR}/manifest.yaml" <<YAMLEOF
name: $(echo "$NAME" | tr '_' '-')
version: "1.0.0"
category: ${CATEGORY}
description: |
  TODO: describe what this component does.
inputs: []
outputs:
  - name: output
    type: String
    description: TODO
    schema: "TODO"
typical_upstream:
  - TODO
typical_downstream:
  - TODO
resource_profile:
  cpu: "1"
  memory: "2Gi"
tags: [TODO]
YAMLEOF
cat > "${DIR}/tests/test_${NAME}.py" <<TESTEOF
def test_${NAME}_placeholder():
    pass
TESTEOF
echo "Scaffolded components/${NAME}/ — fill in component.py, manifest.yaml, and tests."
```

### `scripts/generate_catalog.py`

```python
import argparse, datetime, json, subprocess, tempfile, requests, yaml

def search_specs(registry: str) -> list[dict]:
    query = '{ GlobalSearch(query: "-spec") { Images { RepoName Tag } } }'
    r = requests.post(
        f"http://{registry}/v2/_zot/ext/search",
        json={"query": query},
        timeout=30,
    )
    r.raise_for_status()
    images = r.json()["data"]["GlobalSearch"]["Images"]
    images = [img for img in images if img["RepoName"].endswith("-spec")]

    from packaging.version import Version, InvalidVersion
    best: dict[str, dict] = {}
    for img in images:
        repo = img["RepoName"]
        try:
            v = Version(img["Tag"])
        except InvalidVersion:
            continue
        if repo not in best or v > Version(best[repo]["Tag"]):
            best[repo] = img
    return list(best.values())

def pull_manifest(registry: str, repo: str, tag: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                "oras",
                "pull",
                "--plain-http",
                f"{registry}/{repo}:{tag}",
                "--output",
                tmpdir,
            ],
            check=True,
            capture_output=True,
        )
        with open(f"{tmpdir}/manifest.yaml") as f:
            return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    args = parser.parse_args()

    specs = search_specs(args.registry)
    components = []
    for img in specs:
        manifest = pull_manifest(args.registry, img["RepoName"], img["Tag"])
        name = img["RepoName"].removeprefix("ml-components/").removesuffix("-spec")
        components.append({
            "name": name,
            "latest_stable": img["Tag"],
            "spec_ref": f"{args.registry}/{img['RepoName']}:{img['Tag']}",
            "category":           manifest.get("category", ""),
            "tags":               manifest.get("tags", []),
            "description":        manifest.get("description", ""),
            "typical_downstream": manifest.get("typical_downstream", []),
            "outputs":            manifest.get("outputs", []),
        })

    try:
        with open("catalog_patterns.json") as f:
            patterns = json.load(f)
    except FileNotFoundError:
        patterns = {"composition_patterns": []}

    catalog = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "components": components,
        "composition_patterns": patterns.get("composition_patterns", []),
    }
    with open("catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Catalog written: {len(components)} components.")

if __name__ == "__main__":
    main()
```

### `scripts/compile_component.py`

```python
#!/usr/bin/env python3
"""Compile a KFP component directory to component.yaml.

Usage:
    python scripts/compile_component.py <component_name>

Example:
    python scripts/compile_component.py model_training
    # Generates: components/model_training/component.yaml
"""

import sys
import importlib
import pathlib

sys.path.insert(0, ".")

comp = sys.argv[1]
pkg_name = f"ml_components_{comp}"
mod = importlib.import_module(f"components.{comp}.{pkg_name}.component")
fn = getattr(mod, comp)

from kfp import compiler

out = pathlib.Path(f"components/{comp}/component.yaml")
compiler.Compiler().compile(fn, str(out))
print(f"Generated {out}")
```

---

## Component Files

### `data_ingestion`

#### `components/data_ingestion/ml_components_data_ingestion/component.py`

```python
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
```

#### `components/data_ingestion/ml_components_data_ingestion/__init__.py`

```python
from ml_components_data_ingestion.component import data_ingestion
```

#### `components/data_ingestion/pyproject.toml`

```toml
[project]
name = "ml-components-data-ingestion"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "kfp>=2.0",
    "pandas>=2.0",
    "boto3>=1.34",
    "requests>=2.31",
    "pyarrow>=14.0",
]

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["ml_components_data_ingestion*"]

[tool.setuptools.package-data]
"ml_components_data_ingestion" = ["manifest.yaml"]
```

#### `components/data_ingestion/manifest.yaml`

```yaml
name: data-ingestion
version: "1.0.0"
category: ingestion
description: |
  Resolves the current LakeFS commit for a given repo and branch, then
  returns the stable MinIO S3 path for that commit's dataset. The commit
  is resolved before the data path is constructed to avoid TOCTOU races.
  Downstream components always read from an immutable snapshot.

inputs:
  - name: lakefs_repo
    type: String
    description: LakeFS repository name, e.g. "mlops-data-dev".
  - name: branch
    type: String
    description: LakeFS branch to read from, e.g. "main" or "experiment-1".

outputs:
  - name: dataset_path
    type: Dataset
    description: MinIO S3 URI pointing to the Parquet file for this commit.
    schema: "s3://datasets/<repo>/<commit>/data.parquet"
  - name: lakefs_commit
    type: String
    description: LakeFS commit hash resolved at read time. Passed forward for lineage tagging.
    schema: "<40-char sha>"

typical_upstream:
  - pipeline-start
typical_downstream:
  - data-validation

resource_profile:
  cpu: "0.5"
  memory: "512Mi"

tags: [ingestion, lakefs, s3, parquet, lineage]
```

#### `components/data_ingestion/tests/test_data_ingestion.py`

```python
import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

from unittest.mock import patch, MagicMock


def test_returns_correct_path_and_commit():
    from ml_components_data_ingestion.component import data_ingestion

    mock_response = MagicMock()
    mock_response.json.return_value = {"commit_id": "abc123def456"}
    mock_response.raise_for_status = MagicMock()

    env = {
        "LAKEFS_ENDPOINT": "http://lakefs:8000",
        "LAKEFS_ACCESS_KEY": "key",
        "LAKEFS_SECRET_KEY": "secret",
    }
    with patch("requests.get", return_value=mock_response), patch.dict(
        "os.environ", env
    ):
        result = data_ingestion.python_func(lakefs_repo="test-repo", branch="main")

    assert result.lakefs_commit == "abc123def456"
    assert "abc123def456" in result.dataset_path
    assert result.dataset_path.startswith("s3://datasets/")


def test_raises_on_lakefs_error():
    from ml_components_data_ingestion.component import data_ingestion

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404")

    env = {
        "LAKEFS_ENDPOINT": "http://lakefs:8000",
        "LAKEFS_ACCESS_KEY": "key",
        "LAKEFS_SECRET_KEY": "secret",
    }
    with patch("requests.get", return_value=mock_response), patch.dict(
        "os.environ", env
    ):
        try:
            data_ingestion.python_func(lakefs_repo="missing-repo", branch="main")
            assert False, "Expected exception"
        except Exception:
            pass
```

---

### `data_validation`

#### `components/data_validation/ml_components_data_validation/component.py`

```python
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
```

#### `components/data_validation/ml_components_data_validation/__init__.py`

```python
from ml_components_data_validation.component import data_validation
```

#### `components/data_validation/pyproject.toml`

```toml
[project]
name = "ml-components-data-validation"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "kfp>=2.0",
    "pandas>=2.0",
    "boto3>=1.34",
    "requests>=2.31",
    "pyarrow>=14.0",
]

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["ml_components_data_validation*"]

[tool.setuptools.package-data]
"ml_components_data_validation" = ["manifest.yaml"]
```

#### `components/data_validation/manifest.yaml`

```yaml
name: data-validation
version: "1.0.0"
category: validation
description: |
  Validates a dataset from MinIO before feature engineering begins.
  Checks row count, per-column null fraction, and all-zero numeric columns.
  Raises ValueError on any failure — KFP marks the step Failed and all
  downstream steps are skipped. On success, passes dataset_path through
  unchanged so feature-engineering can consume it directly.

inputs:
  - name: dataset_path
    type: Dataset
    description: Output of data-ingestion. Parquet file on MinIO.
    schema: "s3://datasets/<repo>/<commit>/data.parquet"
  - name: lakefs_commit
    type: String
    description: LakeFS commit hash from data-ingestion. Embedded in the validation report for lineage.
    schema: "<40-char sha>"
  - name: min_rows
    type: Integer
    default: 1000
    description: Minimum acceptable row count. Fails if the dataset has fewer rows.
  - name: max_null_fraction
    type: Float
    default: 0.05
    description: Maximum null fraction per column. Fails if any column exceeds this.

outputs:
  - name: dataset_path
    type: Dataset
    description: Same path as input — passed through unchanged on success.
    schema: "s3://datasets/<repo>/<commit>/data.parquet"
  - name: validation_report
    type: String
    description: JSON summary with status, row count, column list, and lakefs_commit.
    schema: '{"status": "passed", "rows": <int>, "columns": [...], "lakefs_commit": "<sha>"}'

typical_upstream:
  - data-ingestion
typical_downstream:
  - feature-engineering

resource_profile:
  cpu: "1"
  memory: "4Gi"

tags: [validation, data-quality, blocking, gate]
```

#### `components/data_validation/tests/test_data_validation.py`

```python
import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import pandas as pd


def make_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


ENV = {
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY": "test-access-key",
    "MINIO_SECRET_KEY": "test-secret-key",
}


def test_valid_dataset_passes():
    from ml_components_data_validation.component import data_validation

    df = pd.DataFrame({"a": range(1500), "b": [1.0] * 1500})
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: make_parquet_bytes(df))
    }

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        result = data_validation.python_func(
            dataset_path="s3://datasets/repo/abc/data.parquet",
            lakefs_commit="abc123",
        )

    report = json.loads(result.validation_report)
    assert report["status"] == "passed"
    assert result.dataset_path == "s3://datasets/repo/abc/data.parquet"


def test_too_few_rows_raises():
    from ml_components_data_validation.component import data_validation

    df = pd.DataFrame({"a": range(5), "b": [1.0] * 5})
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: make_parquet_bytes(df))
    }

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        with pytest.raises(ValueError, match="row count"):
            data_validation.python_func(
                dataset_path="s3://datasets/repo/abc/data.parquet",
                lakefs_commit="abc123",
                min_rows=1000,
            )
```

---

### `feature_engineering`

#### `components/feature_engineering/ml_components_feature_engineering/component.py`

```python
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
```

#### `components/feature_engineering/ml_components_feature_engineering/__init__.py`

```python
from ml_components_feature_engineering.component import feature_engineering
```

#### `components/feature_engineering/pyproject.toml`

```toml
[project]
name = "ml-components-feature-engineering"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "kfp>=2.0",
    "pandas>=2.0",
    "boto3>=1.34",
    "requests>=2.31",
    "pyarrow>=14.0",
]

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["ml_components_feature_engineering*"]

[tool.setuptools.package-data]
"ml_components_feature_engineering" = ["manifest.yaml"]
```

#### `components/feature_engineering/manifest.yaml`

```yaml
name: feature-engineering
version: "1.0.0"
category: feature-engineering
description: |
  Applies StandardScaler normalization to all numeric columns in the validated
  dataset and writes the result as a new Parquet file to MinIO under the same
  bucket, with "data.parquet" replaced by "features.parquet" in the key.
  Only numeric columns are scaled; categorical columns are passed through unchanged.

inputs:
  - name: dataset_path
    type: Dataset
    description: Validated dataset path from data-validation.
    schema: "s3://datasets/<repo>/<commit>/data.parquet"

outputs:
  - name: features_path
    type: Dataset
    description: Normalized feature set written to MinIO.
    schema: "s3://datasets/<repo>/<commit>/features.parquet"

typical_upstream:
  - data-validation
typical_downstream:
  - model-training

resource_profile:
  cpu: "2"
  memory: "8Gi"

tags: [feature-engineering, normalization, sklearn, parquet]
```

#### `components/feature_engineering/tests/test_feature_engineering.py`

```python
import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

import pytest
from unittest.mock import patch, MagicMock, call
from io import BytesIO
import pandas as pd

ENV = {
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY": "test-access-key",
    "MINIO_SECRET_KEY": "test-secret-key",
}


def test_features_path_replaces_data_with_features():
    from ml_components_feature_engineering.component import feature_engineering

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0] * 500, "y": [4.0, 5.0, 6.0] * 500})
    buf = BytesIO()
    df.to_parquet(buf, index=False)

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: buf.getvalue())}

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        result = feature_engineering.python_func(
            dataset_path="s3://datasets/repo/abc/data.parquet"
        )

    assert result.features_path == "s3://datasets/repo/abc/features.parquet"
    mock_s3.put_object.assert_called_once()
```

---

### `model_training`

#### `components/model_training/ml_components_model_training/component.py`

```python
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
```

#### `components/model_training/ml_components_model_training/__init__.py`

```python
from ml_components_model_training.component import model_training
```

#### `components/model_training/pyproject.toml`

```toml
[project]
name = "ml-components-model-training"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "kfp>=2.0",
    "pandas>=2.0",
    "boto3>=1.34",
    "requests>=2.31",
    "pyarrow>=14.0",
    "scikit-learn>=1.0",
    "mlflow>=2.0",
]

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["ml_components_model_training*"]

[tool.setuptools.package-data]
"ml_components_model_training" = ["manifest.yaml"]
```

#### `components/model_training/manifest.yaml`

```yaml
name: model-training
version: "1.0.0"
category: training
description: |
  Trains a LogisticRegression classifier on the normalized feature set from
  feature-engineering. Logs parameters and metrics to MLflow and saves the
  serialized model to the MinIO "models" bucket. The run ID ties this
  artifact back to its MLflow experiment record for full lineage.

inputs:
  - name: features_path
    type: Dataset
    description: Normalized feature Parquet from feature-engineering.
    schema: "s3://datasets/<repo>/<commit>/features.parquet"
  - name: mlflow_run_name
    type: String
    default: "tron-training"
    description: Display name for the MLflow run.
  - name: epochs
    type: Integer
    default: 10
    description: Training epochs (currently passed to MLflow params for logging; LogisticRegression uses max_iter).
  - name: max_iter
    type: Integer
    default: 200
    description: Maximum iterations for LogisticRegression convergence.

outputs:
  - name: model_artifact
    type: Model
    description: MinIO S3 URI of the serialized model pickle.
    schema: "s3://models/tron/<run_id>/model.pkl"
  - name: mlflow_run_id
    type: String
    description: MLflow run ID. Used by model-evaluation to link metrics to this training run.
    schema: "<uuid>"

typical_upstream:
  - feature-engineering
typical_downstream:
  - model-evaluation

resource_profile:
  cpu: "2"
  memory: "8Gi"

tags: [training, sklearn, mlflow, classification]
```

#### `components/model_training/tests/test_model_training.py`

```python
import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import pandas as pd
import numpy as np

ENV = {
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY": "test-access-key",
    "MINIO_SECRET_KEY": "test-secret-key",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
}


def make_features_parquet() -> bytes:
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "label": np.random.randint(0, 2, n),
        }
    )
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def test_returns_model_artifact_and_run_id():
    from ml_components_model_training.component import model_training

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=make_features_parquet)}
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-id-123"

    with patch("boto3.client", return_value=mock_s3), patch.dict(
        "os.environ", ENV
    ), patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"), patch(
        "mlflow.start_run", return_value=mock_run
    ), patch(
        "mlflow.log_params"
    ), patch(
        "mlflow.log_metric"
    ), patch(
        "mlflow.log_param"
    ):
        result = model_training.python_func(
            features_path="s3://datasets/repo/abc/features.parquet",
            epochs=1,
            max_iter=50,
        )

    assert "s3://models/" in result.model_artifact
    assert result.mlflow_run_id == "test-run-id-123"
```

---

### `model_evaluation`

#### `components/model_evaluation/ml_components_model_evaluation/component.py`

```python
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
```

#### `components/model_evaluation/ml_components_model_evaluation/__init__.py`

```python
from ml_components_model_evaluation.component import model_evaluation
```

#### `components/model_evaluation/pyproject.toml`

```toml
[project]
name = "ml-components-model-evaluation"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "kfp>=2.0",
    "pandas>=2.0",
    "boto3>=1.34",
    "requests>=2.31",
    "pyarrow>=14.0",
    "scikit-learn>=1.0",
]

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["ml_components_model_evaluation*"]

[tool.setuptools.package-data]
"ml_components_model_evaluation" = ["manifest.yaml"]
```

#### `components/model_evaluation/manifest.yaml`

```yaml
name: model-evaluation
version: "1.0.0"
category: evaluation
description: |
  Loads the trained model artifact from MinIO and the feature set from
  feature-engineering. Computes held-out accuracy using the same 80/20
  split used in training (random_state=42). Raises if accuracy is below
  min_accuracy — the pipeline is aborted and the model is never registered.
  On success, returns a JSON evaluation report and the accuracy value.

inputs:
  - name: model_artifact
    type: Model
    description: MinIO S3 URI of the pickled model from model-training.
    schema: "s3://models/tron/<run_id>/model.pkl"
  - name: features_path
    type: Dataset
    description: Normalized feature Parquet from feature-engineering.
    schema: "s3://datasets/<repo>/<commit>/features.parquet"
  - name: min_accuracy
    type: Float
    default: 0.80
    description: Minimum acceptable accuracy. Raises and aborts pipeline if not met.

outputs:
  - name: evaluation_report
    type: String
    description: JSON report with status, accuracy, threshold, and model artifact URI.
    schema: '{"status": "passed"|"failed", "accuracy": <float>, "min_accuracy": <float>, "model_artifact": "<uri>"}'
  - name: accuracy
    type: Float
    description: Held-out accuracy score (0.0–1.0).
    schema: "<float in [0.0, 1.0]>"

typical_upstream:
  - model-training
typical_downstream:
  - model-registry

resource_profile:
  cpu: "1"
  memory: "4Gi"

tags: [evaluation, metrics, accuracy, gate, sklearn]
```

#### `components/model_evaluation/tests/test_model_evaluation.py`

```python
import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

import json
import pickle
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ENV = {
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY": "test-access-key",
    "MINIO_SECRET_KEY": "test-secret-key",
}


def make_model_bytes() -> bytes:
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(max_iter=200).fit(X, y)
    return pickle.dumps(model)


def make_features_parquet() -> bytes:
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def test_passing_model_returns_report():
    from ml_components_model_evaluation.component import model_evaluation

    mock_s3 = MagicMock()
    responses = [
        {"Body": MagicMock(read=make_model_bytes)},
        {"Body": MagicMock(read=make_features_parquet)},
    ]
    mock_s3.get_object.side_effect = responses

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        result = model_evaluation.python_func(
            model_artifact="s3://models/tron/run1/model.pkl",
            features_path="s3://datasets/repo/abc/features.parquet",
            min_accuracy=0.0,
        )

    report = json.loads(result.evaluation_report)
    assert report["status"] == "passed"
    assert isinstance(result.accuracy, float)


def test_low_accuracy_raises():
    from ml_components_model_evaluation.component import model_evaluation

    mock_s3 = MagicMock()
    responses = [
        {"Body": MagicMock(read=make_model_bytes)},
        {"Body": MagicMock(read=make_features_parquet)},
    ]
    mock_s3.get_object.side_effect = responses

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        with pytest.raises(ValueError, match="accuracy"):
            model_evaluation.python_func(
                model_artifact="s3://models/tron/run1/model.pkl",
                features_path="s3://datasets/repo/abc/features.parquet",
                min_accuracy=1.0,
            )
```

---

## CI Workflows: ml-platform/ml-components

### `.forgejo/workflows/publish.yaml`

```yaml
name: Publish

on:
  push:
    branches: [main]

jobs:
  detect-changes:
    runs-on: linux
    container:
      image: forgejo-runner-python3.12:latest
    outputs:
      components: ${{ steps.detect.outputs.components }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Detect changed components
        id: detect
        run: |
          BEFORE="${{ github.event.before }}"
          AFTER="${{ github.sha }}"
          EMPTY_TREE="4b825dc642cb6eb9a060e54bf8d69288fbee4904"
          if [ "$BEFORE" = "0000000000000000000000000000000000000000" ]; then
            BEFORE="$EMPTY_TREE"
          fi
          CHANGED=$(git diff --name-only "$BEFORE" "$AFTER")
          echo "Changed files:"
          echo "$CHANGED"

          COMPONENTS=$(echo "$CHANGED" \
            | { grep '^components/' || true; } \
            | cut -d/ -f2 \
            | { grep -v __init__.py || true; } \
            | sort -u \
            | jq -R . | jq -sc .)

          # Ensure we always output a valid JSON array (even if empty)
          if [ "$COMPONENTS" = "" ] || [ "$COMPONENTS" = "null" ]; then
            COMPONENTS="[]"
          fi
          echo "components=$COMPONENTS" >> "$GITHUB_OUTPUT"
          echo "Detected components to publish: $COMPONENTS"

  ensure-base-image:
    needs: detect-changes
    if: needs.detect-changes.outputs.components != '[]' && needs.detect-changes.outputs.components != 'null'
    runs-on: linux
    container:
      image: forgejo-runner-python3.12:latest
    steps:
      - uses: actions/checkout@v4

      - name: Build base image if missing
        run: |
          LOCAL_BASE="ml-components/base:latest"
          if docker image inspect "$LOCAL_BASE" > /dev/null 2>&1; then
            echo "Base image already exists locally, skipping build."
            exit 0
          fi
          echo "Base image missing, building..."
          docker build \
            --platform linux/amd64 \
            -t "$LOCAL_BASE" \
            -f docker/Dockerfile.base \
            .
          echo "Base image tagged locally as $LOCAL_BASE"

  publish-component:
    needs: [detect-changes, ensure-base-image]
    if: needs.detect-changes.outputs.components != '[]' && needs.detect-changes.outputs.components != 'null'
    strategy:
      matrix:
        component: ${{ fromJson(needs.detect-changes.outputs.components) }}
    runs-on: linux
    container:
      image: forgejo-runner-python3.12:latest
    steps:
      - uses: actions/checkout@v4

      - name: Build component image locally
        run: |
          COMPONENT="${{ matrix.component }}"
          uv pip install --python /opt/ml-venv/bin/python -e components/$COMPONENT
          uv pip install --python /opt/ml-venv/bin/python docker
          PYTHONPATH=. kfp component build components/$COMPONENT --no-push-image
          echo "Component Docker image built locally (not pushed)"

      - name: Prepare component.yaml for ZOT push
        run: |
          COMPONENT="${{ matrix.component }}"
          PYTHONPATH=. python3 scripts/compile_component.py $COMPONENT
          echo "Generated components/$COMPONENT/component.yaml"

      - name: Push component spec to ZOT
        run: |
          VERSION=$(python3 -c "import yaml; print(yaml.safe_load(open('components/${{ matrix.component }}/manifest.yaml'))['version'])")
          KEBAB=$(echo "${{ matrix.component }}" | tr '_' '-')
          cd components/${{ matrix.component }}
          oras push --plain-http 172.17.0.1:8001/ml-components/${KEBAB}-spec:${VERSION} \
            component.yaml manifest.yaml
          oras push --plain-http 172.17.0.1:8001/ml-components/${KEBAB}-spec:latest \
            component.yaml manifest.yaml

      - name: Create component git tag
        run: |
          VERSION=$(python3 -c "import yaml; print(yaml.safe_load(open('components/${{ matrix.component }}/manifest.yaml'))['version'])")
          KEBAB=$(echo "${{ matrix.component }}" | tr '_' '-')
          git config user.email "ci@forgejo.local"
          git config user.name "Forgejo CI"
          TAG="${KEBAB}/v${VERSION}"
          git tag "$TAG" || echo "Tag $TAG already exists, skipping."
          git push origin "$TAG" || echo "Tag push skipped."

  publish-python-package:
    needs: [detect-changes, publish-component]
    if: needs.detect-changes.outputs.components != '[]' && needs.detect-changes.outputs.components != 'null'
    strategy:
      matrix:
        component: ${{ fromJson(needs.detect-changes.outputs.components) }}
    runs-on: linux
    container:
      image: forgejo-runner-python3.12:latest
    steps:
      - uses: actions/checkout@v4

      - name: Get component package name and version
        id: pkg
        run: |
          COMPONENT="${{ matrix.component }}"
          KEBAB=$(echo "$COMPONENT" | tr '_' '-')
          VERSION=$(python3 -c "import yaml; print(yaml.safe_load(open('components/${COMPONENT}/manifest.yaml'))['version'])")
          PKG_NAME="ml-components-${KEBAB}"
          echo "pkg_name=${PKG_NAME}" >> "$GITHUB_OUTPUT"
          echo "version=${VERSION}" >> "$GITHUB_OUTPUT"

      - name: Verify version was bumped
        run: |
          CURRENT="${{ steps.pkg.outputs.version }}"
          PYPI_API_BASE=$(echo "${{ vars.PYPI_INDEX_URL }}" | sed 's|/simple/[[:space:]]*$||')
          PUBLISHED=$(curl -sf "${PYPI_API_BASE}/${{ steps.pkg.outputs.pkg_name }}/json" \
            | python3 -c "
            import sys, json
            from packaging.version import Version
            data = json.load(sys.stdin)
            versions = list(data.get('releases', {}).keys())
            print(str(max(versions, key=Version)) if versions else '0.0.0')
            " 2>/dev/null || echo '0.0.0')
          python3 -c "
          from packaging.version import Version
          c, p = Version('${{ steps.pkg.outputs.version }}'), Version('$PUBLISHED')
          if c <= p:
              raise SystemExit(f'FATAL: {${{ steps.pkg.outputs.pkg_name }}} version {c} <= published {p}. Version was not bumped.')
          print(f'Publishing: {p} -> {c}')
          "

      - name: Build wheel
        run: |
          COMPONENT="${{ matrix.component }}"
          python3 -m build components/$COMPONENT --outdir dist

      - name: Publish to Forgejo PyPI
        run: |
          twine upload \
            --repository-url http://172.17.0.1:4000/api/packages/ml-platform/pypi \
            --username "${{ secrets.FJ_USERNAME }}" \
            --password "${{ secrets.FJ_TOKEN }}" \
            dist/*

      - name: Create package git tag
        run: |
          KEBAB=$(echo "${{ matrix.component }}" | tr '_' '-')
          VERSION="${{ steps.pkg.outputs.version }}"
          git config user.email "ci@forgejo.local"
          git config user.name "Forgejo CI"
          TAG="${KEBAB}/v${VERSION}"
          git tag "$TAG" || echo "Tag $TAG already exists, skipping."
          git push origin "$TAG" || echo "Tag push skipped."

  regenerate-catalog:
    needs: [publish-component, publish-python-package]
    if: ${{ always() && needs.publish-component.result == 'success' && needs.publish-python-package.result == 'success' }}
    concurrency:
      group: catalog-publish
      cancel-in-progress: false
    runs-on: linux
    container:
      image: forgejo-runner-python3.12:latest
    steps:
      - uses: actions/checkout@v4

      - name: Build catalog from ZOT
        run: python3 scripts/generate_catalog.py --registry 172.17.0.1:8001

      - name: Push catalog to ZOT
        run: |
          oras push --plain-http 172.17.0.1:8001/ml-components/catalog:latest \
            --disable-path-validation \
            catalog.json:application/vnd.ml.catalog.v1+json
          oras push --plain-http 172.17.0.1:8001/ml-components/catalog:$(date +%F) \
            --disable-path-validation \
            catalog.json:application/vnd.ml.catalog.v1+json
```

---

## File Contents: team-tron/pipelines

### `pyproject.toml` (team-tron/pipelines)

```toml
[project]
name = "team-tron-pipelines"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
    "ml-components-data-ingestion==1.0.0",
    "ml-components-data-validation==1.0.0",
    "ml-components-feature-engineering==1.0.0",
    "ml-components-model-training==1.0.0",
    "ml-components-model-evaluation==1.0.0",
    "kfp>=2.0",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "python-dotenv>=1.0",
]

[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["pipelines*", "kfp_utils*", "config*"]

[tool.uv.sources]
ml-components-data-ingestion = { index = "forgejo-pypi" }
ml-components-data-validation = { index = "forgejo-pypi" }
ml-components-feature-engineering = { index = "forgejo-pypi" }
ml-components-model-training = { index = "forgejo-pypi" }
ml-components-model-evaluation = { index = "forgejo-pypi" }

[[tool.uv.index]]
name = "forgejo-pypi"
url = "http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/"
explicit = true
```

### `pipelines/training_pipeline.py`

```python
from kfp import dsl
from ml_components_data_ingestion import data_ingestion
from ml_components_data_validation import data_validation
from ml_components_feature_engineering import feature_engineering
from ml_components_model_training import model_training
from ml_components_model_evaluation import model_evaluation


@dsl.pipeline(
    name="tron-training-pipeline",
    description="Full supervised learning pipeline: ingest → validate → engineer → train → evaluate.",
)
def training_pipeline(
    lakefs_repo: str = "mlops-data-dev",
    branch: str = "main",
    model_name: str = "tron-classifier",
    min_rows: int = 1000,
    max_null_fraction: float = 0.05,
    epochs: int = 10,
    max_iter: int = 200,
    min_accuracy: float = 0.80,
):
    ingest = data_ingestion(lakefs_repo=lakefs_repo, branch=branch)

    validate = data_validation(
        dataset_path=ingest.outputs["dataset_path"],
        lakefs_commit=ingest.outputs["lakefs_commit"],
        min_rows=min_rows,
        max_null_fraction=max_null_fraction,
    )

    features = feature_engineering(dataset_path=validate.outputs["dataset_path"])

    train = model_training(
        features_path=features.outputs["features_path"],
        mlflow_run_name=model_name,
        epochs=epochs,
        max_iter=max_iter,
    )

    model_evaluation(
        model_artifact=train.outputs["model_artifact"],
        features_path=features.outputs["features_path"],
        min_accuracy=min_accuracy,
    )
```

### `pipelines/retraining_pipeline.py`

```python
from kfp import dsl
from ml_components_data_ingestion.component import data_ingestion
from ml_components_data_validation.component import data_validation
from ml_components_feature_engineering.component import feature_engineering
from ml_components_model_training.component import model_training
from ml_components_model_evaluation.component import model_evaluation


@dsl.pipeline(
    name="tron-retraining-pipeline",
    description=(
        "Hyperparameter-tunable retraining pipeline. Exposes epochs and max_iter "
        "as top-level parameters for experimentation sweeps. Uses a relaxed "
        "min_accuracy threshold (0.70) to accommodate exploratory runs."
    ),
)
def retraining_pipeline(
    lakefs_repo: str = "mlops-data-dev",
    branch: str = "experiment-1",
    model_name: str = "tron-classifier-retrain",
    min_rows: int = 500,
    max_null_fraction: float = 0.10,
    epochs: int = 20,
    max_iter: int = 500,
    min_accuracy: float = 0.70,
):
    ingest = data_ingestion(lakefs_repo=lakefs_repo, branch=branch)

    validate = data_validation(
        dataset_path=ingest.outputs["dataset_path"],
        lakefs_commit=ingest.outputs["lakefs_commit"],
        min_rows=min_rows,
        max_null_fraction=max_null_fraction,
    )

    features = feature_engineering(dataset_path=validate.outputs["dataset_path"])

    train = model_training(
        features_path=features.outputs["features_path"],
        mlflow_run_name=model_name,
        epochs=epochs,
        max_iter=max_iter,
    )

    model_evaluation(
        model_artifact=train.outputs["model_artifact"],
        features_path=features.outputs["features_path"],
        min_accuracy=min_accuracy,
    )
```

### `pipelines/evaluation_pipeline.py`

```python
from kfp import dsl
from ml_components_data_ingestion.component import data_ingestion
from ml_components_data_validation.component import data_validation
from ml_components_feature_engineering.component import feature_engineering
from ml_components_model_training.component import model_training
from ml_components_model_evaluation.component import model_evaluation


@dsl.pipeline(
    name="tron-evaluation-pipeline",
    description=(
        "Quick data and model quality check. Trains a minimal model (epochs=1) "
        "and asserts it passes a low accuracy floor. Use this to validate that a "
        "new data branch is well-formed and learnable before kicking off a full "
        "training run."
    ),
)
def evaluation_pipeline(
    lakefs_repo: str = "mlops-data-dev",
    branch: str = "main",
    model_name: str = "tron-eval-probe",
    min_rows: int = 100,
    max_null_fraction: float = 0.10,
    min_accuracy: float = 0.55,
):
    ingest = data_ingestion(lakefs_repo=lakefs_repo, branch=branch)

    validate = data_validation(
        dataset_path=ingest.outputs["dataset_path"],
        lakefs_commit=ingest.outputs["lakefs_commit"],
        min_rows=min_rows,
        max_null_fraction=max_null_fraction,
    )

    features = feature_engineering(dataset_path=validate.outputs["dataset_path"])

    # Train a minimal model — 1 epoch is intentional, this is a probe not a real model.
    train = model_training(
        features_path=features.outputs["features_path"],
        mlflow_run_name=model_name,
        epochs=1,
        max_iter=50,
    )

    model_evaluation(
        model_artifact=train.outputs["model_artifact"],
        features_path=features.outputs["features_path"],
        min_accuracy=min_accuracy,
    )
```

### `tests/test_pipeline_compile.py`

```python
"""Verify all three pipelines compile to valid KFP YAML without errors."""
import tempfile
from kfp import compiler
from pipelines.training_pipeline import training_pipeline
from pipelines.retraining_pipeline import retraining_pipeline
from pipelines.evaluation_pipeline import evaluation_pipeline


def test_training_pipeline_compiles():
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        compiler.Compiler().compile(training_pipeline, f.name)


def test_retraining_pipeline_compiles():
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        compiler.Compiler().compile(retraining_pipeline, f.name)


def test_evaluation_pipeline_compiles():
    with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        compiler.Compiler().compile(evaluation_pipeline, f.name)
```

### `config/dev.yaml`

```yaml
lakefs_repo: mlops-data-dev
branch: main
model_name: tron-classifier-dev
min_rows: 100
max_null_fraction: 0.10
epochs: 3
max_iter: 100
min_accuracy: 0.60
```

### `config/prod.yaml`

```yaml
lakefs_repo: mlops-data-prod
branch: main
model_name: tron-classifier
min_rows: 10000
max_null_fraction: 0.02
epochs: 20
max_iter: 500
min_accuracy: 0.85
```

### `.env.local.example`

Do **not** put actual secret values in this file or in `.env.local`. Instead, use
`scripts/get_local_env.sh` to populate credentials from the Kubernetes YAML sources:

```bash
eval "$(bash scripts/get_local_env.sh)"
```

If you prefer a file-based workflow, create `.env.local` (gitignored) and fill in
values obtained from the k8s YAML files:

```bash
# Endpoints — non-sensitive, safe to set here
MLFLOW_TRACKING_URI=http://localhost:5000
LAKEFS_ENDPOINT=http://localhost:8000
MINIO_ENDPOINT=http://localhost:9000
LAKEFS_REPO=mlops-data-dev
BRANCH=main

# Secrets — DO NOT commit actual values; get them from:
#   k8s_yamls/lakefs/02-secret.yaml  (LAKEFS_AUTH_ADMIN_ACCESS_KEY_ID / SECRET_ACCESS_KEY)
#   k8s_yamls/minio/02-secret.yaml   (MINIO_ROOT_USER / MINIO_ROOT_PASSWORD)
LAKEFS_ACCESS_KEY=
LAKEFS_SECRET_KEY=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
```

### `scripts/get_local_env.sh`

Fetches credentials from the same Kubernetes YAML source files used by
`install_forgejo.sh` and `create_forgejo_repo.sh`, and prints them as shell
`export` statements. Run it once before local development or seeding:

```bash
# From /workspace/git_repo/mlops/
eval "$(bash repos/team-tron/pipelines/scripts/get_local_env.sh)"
```

```bash
#!/usr/bin/env bash
# Print export statements for all credentials and endpoints needed locally.
# Reads from the same Kubernetes YAML files used by install_forgejo.sh
# and create_forgejo_repo.sh — no secrets ever live in this file.
#
# Usage (from /workspace/git_repo/mlops/):
#   eval "$(bash repos/team-tron/pipelines/scripts/get_local_env.sh)"
set -euo pipefail

MLOPS_ROOT="${MLOPS_ROOT:-$(pwd)}"

_yaml_field() {
    grep "${1}:" "${MLOPS_ROOT}/k8s_yamls/${2}" | awk -F': ' '{print $2}' | tr -d '"'
}

MINIO_ACCESS_KEY=$(_yaml_field 'MINIO_ROOT_USER'                      'minio/02-secret.yaml')
MINIO_SECRET_KEY=$(_yaml_field 'MINIO_ROOT_PASSWORD'                  'minio/02-secret.yaml')
LAKEFS_ACCESS_KEY=$(_yaml_field 'LAKEFS_AUTH_ADMIN_ACCESS_KEY_ID'     'lakefs/02-secret.yaml')
LAKEFS_SECRET_KEY=$(_yaml_field 'LAKEFS_AUTH_ADMIN_SECRET_ACCESS_KEY' 'lakefs/02-secret.yaml')

printf 'export MINIO_ACCESS_KEY=%s\n'        "$MINIO_ACCESS_KEY"
printf 'export MINIO_SECRET_KEY=%s\n'        "$MINIO_SECRET_KEY"
printf 'export LAKEFS_ACCESS_KEY=%s\n'       "$LAKEFS_ACCESS_KEY"
printf 'export LAKEFS_SECRET_KEY=%s\n'       "$LAKEFS_SECRET_KEY"
printf 'export MINIO_ENDPOINT=%s\n'          "${MINIO_ENDPOINT:-http://localhost:9000}"
printf 'export LAKEFS_ENDPOINT=%s\n'         "${LAKEFS_ENDPOINT:-http://localhost:8000}"
printf 'export MLFLOW_TRACKING_URI=%s\n'     "${MLFLOW_TRACKING_URI:-http://localhost:5000}"
printf 'export LAKEFS_REPO=%s\n'             "${LAKEFS_REPO:-mlops-data-dev}"
printf 'export BRANCH=%s\n'                  "${BRANCH:-main}"
```

---

### `scripts/seed_data.py`

Seeds a sklearn classification dataset into MinIO and ensures the LakeFS repo and
branch exist before any pipeline runs. Idempotent — safe to call repeatedly.

**What it does:**

1. Creates MinIO buckets `datasets` and `models` if they do not exist.
2. Creates the LakeFS repo (default: `mlops-data-dev`) if it does not exist.
3. Resolves the current commit ID on the target branch (the initial commit created
   automatically when the repo is made).
4. Generates a synthetic 2 000-row classification dataset via
   `sklearn.datasets.make_classification` (20 features + 1 label column, so the
   last column is the label as `model_training` expects).
5. Writes it as Parquet to MinIO at `datasets/<lakefs_repo>/<commit_id>/data.parquet`
   — exactly the path `data_ingestion` constructs at runtime.

This script must succeed before any pipeline can run end-to-end. The submit CI
workflow runs it automatically before submitting the pipeline. For local runs,
execute it once manually:

```bash
python scripts/seed_data.py
```

```python
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
```

---

### `run_local.py`

```python
#!/usr/bin/env python3
"""Run the training pipeline locally using KFP SubprocessRunner.

Seeds initial data automatically so the pipeline can run without any manual setup.

Before running, populate credentials from Kubernetes YAML sources:
    eval "$(bash scripts/get_local_env.sh)"
    python run_local.py

Or use a gitignored .env.local file (see .env.local.example for the required vars).
"""
import os
from pathlib import Path
from dotenv import load_dotenv  # pip install python-dotenv

# Load .env.local if present (gitignored). Credentials must be in the environment
# either via this file or via eval "$(bash scripts/get_local_env.sh)".
env_file = Path(".env.local")
if env_file.exists():
    load_dotenv(env_file)
else:
    print(
        "Info: .env.local not found. Expecting credentials in the environment.\n"
        "If they are missing, run: eval \"$(bash scripts/get_local_env.sh)\""
    )

# Seed initial data before running — idempotent, safe to call every time.
from scripts.seed_data import main as seed_data
seed_data()

from kfp import local
from pipelines.training_pipeline import training_pipeline

local.init(runner=local.SubprocessRunner(), pipeline_root="/tmp/kfp-tron")

training_pipeline(
    lakefs_repo=os.environ.get("LAKEFS_REPO", "mlops-data-dev"),
    branch=os.environ.get("BRANCH", "main"),
    model_name="tron-classifier-local",
)
```

### `kfp_utils/auth.py`

Handles the full Kubeflow / Dex authentication redirect chain and returns a ready-to-use
`kfp.Client`. Used by `submit.yaml` so the auth logic is testable and not buried in a
bash heredoc.

```python
import html
import os
import re
from urllib.parse import urljoin

import kfp
import requests


def _extract_login_form(html_text: str) -> tuple[str, dict[str, str]]:
    form_match = re.search(
        r"<form[^>]*action=\"([^\"]+)\"[^>]*>", html_text, re.IGNORECASE
    )
    if not form_match:
        raise RuntimeError("Could not find Dex login form in response HTML.")

    action = html.unescape(form_match.group(1))
    hidden_inputs = {
        html.unescape(key): html.unescape(value)
        for key, value in re.findall(
            r"<input[^>]*type=\"hidden\"[^>]*name=\"([^\"]+)\"[^>]*value=\"([^\"]*)\"",
            html_text,
            re.IGNORECASE,
        )
    }
    return action, hidden_inputs


def _extract_form(
    html_text: str, action_pattern: str
) -> tuple[str, dict[str, str]] | None:
    form_match = re.search(
        rf"<form[^>]*action=\"([^\"]*{action_pattern}[^\"]*)\"[^>]*>",
        html_text,
        re.IGNORECASE,
    )
    if not form_match:
        return None

    action = html.unescape(form_match.group(1))
    hidden_inputs = {
        html.unescape(key): html.unescape(value)
        for key, value in re.findall(
            r"<input[^>]*type=\"hidden\"[^>]*name=\"([^\"]+)\"[^>]*value=\"([^\"]*)\"",
            html_text,
            re.IGNORECASE,
        )
    }
    return action, hidden_inputs


def _ensure_pipeline_host(host: str) -> str:
    cleaned = host.rstrip("/")
    if cleaned.endswith("/pipeline"):
        return cleaned
    return f"{cleaned}/pipeline"


def _host_without_pipeline(host: str) -> str:
    cleaned = host.rstrip("/")
    if cleaned.endswith("/pipeline"):
        return cleaned[: -len("/pipeline")]
    return cleaned


def get_authservice_cookie(
    host: str, username: str, password: str, timeout: int = 30
) -> str:
    session = requests.Session()
    ingress_host = _host_without_pipeline(host)

    resp = session.get(
        f"{ingress_host}/pipeline", allow_redirects=True, timeout=timeout
    )

    oauth2_start_form = _extract_form(resp.text, r"/oauth2/start")
    if oauth2_start_form is not None:
        start_action, start_hidden = oauth2_start_form
        start_url = urljoin(resp.url, start_action)
        start_resp = session.post(
            start_url,
            data=start_hidden,
            allow_redirects=True,
            timeout=timeout,
        )
        start_resp.raise_for_status()
        login_page = start_resp
    else:
        resp.raise_for_status()
        login_page = resp

    action, hidden = _extract_login_form(login_page.text)
    login_url = urljoin(login_page.url, action)

    payload = {
        **hidden,
        "login": username,
        "password": password,
    }

    login_resp = session.post(
        login_url, data=payload, allow_redirects=True, timeout=timeout
    )
    login_resp.raise_for_status()

    cookie_name_candidates = [
        "authservice_session",
        "oauth2_proxy",
        "oauth2_proxy_kubeflow",
        "__Host-authservice_session",
    ]

    cookie_name = None
    cookie_value = None
    for candidate in cookie_name_candidates:
        value = session.cookies.get(candidate)
        if value:
            cookie_name = candidate
            cookie_value = value
            break

    if not cookie_name or not cookie_value:
        raise RuntimeError(
            "Dex login completed but no supported auth cookie was found "
            "(authservice_session/oauth2_proxy). Check host URL, credentials, and Dex configuration."
        )
    return f"{cookie_name}={cookie_value}"


def get_kfp_client(
    host: str | None = None,
    namespace: str | None = None,
    cookie: str | None = None,
    token: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> kfp.Client:
    host_value = _ensure_pipeline_host(host or os.getenv("KUBEFLOW_HOST"))
    namespace_value = namespace or os.getenv("KUBEFLOW_NAMESPACE")
    cookie_value = cookie or os.getenv("KUBEFLOW_COOKIE")
    token_value = token or os.getenv("KUBEFLOW_TOKEN")
    username_value = username or os.getenv("KUBEFLOW_USERNAME")
    password_value = password or os.getenv("KUBEFLOW_PASSWORD")

    if not cookie_value and not token_value and username_value and password_value:
        cookie_value = get_authservice_cookie(
            host_value, username_value, password_value
        )
    print(f"Using KUBEFLOW_HOST: {host_value}")
    print(f"Using KUBEFLOW_NAMESPACE: {namespace_value}")
    client_kwargs: dict[str, str] = {
        "host": host_value,
        "namespace": namespace_value,
    }
    if cookie_value:
        client_kwargs["cookies"] = cookie_value
    if token_value:
        client_kwargs["existing_token"] = token_value

    return kfp.Client(**client_kwargs)
```

### CI Workflows: team-tron/pipelines

#### `.forgejo/workflows/ci.yaml`

```yaml
name: CI

on:
  pull_request:
    branches: [main]

jobs:
  compile-and-test:
    runs-on: linux
    container:
      image: forgejo-runner-python3.12:latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        run: pip install uv

      - name: Install dependencies from lock file
        # --extra dev installs pytest (required by the step below).
        # The Forgejo PyPI index is already configured in pyproject.toml as an explicit
        # index (explicit = true), so uv uses it only for ml-components and uses PyPI
        # for everything else. Do NOT pass --index-url here — that would replace PyPI
        # entirely with the Forgejo registry and break resolution of kfp, pandas, etc.
        run: uv sync --frozen --extra dev

      - name: Compile all pipelines
        run: |
          uv run python -c "
          import tempfile
          from kfp import compiler
          from pipelines.training_pipeline import training_pipeline
          from pipelines.retraining_pipeline import retraining_pipeline
          from pipelines.evaluation_pipeline import evaluation_pipeline
          for name, pipeline in [
              ('training', training_pipeline),
              ('retraining', retraining_pipeline),
              ('evaluation', evaluation_pipeline),
          ]:
              with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
                  compiler.Compiler().compile(pipeline, f.name)
              print(f'OK  {name}_pipeline compiled')
          "

      - name: Run tests
        run: uv run pytest tests/ -v --tb=short
```

#### `.forgejo/workflows/submit.yaml`

```yaml
name: Submit Pipeline

on:
  push:
    branches: [main]

jobs:
  submit-training-pipeline:
    runs-on: linux
    container:
      image: forgejo-runner-python3.12:latest
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies from lock file
        run: uv sync --frozen

      - name: Seed initial data
        env:
          LAKEFS_ENDPOINT: http://172.17.0.1:8000
          LAKEFS_ACCESS_KEY: ${{ secrets.LAKEFS_ACCESS_KEY }}
          LAKEFS_SECRET_KEY: ${{ secrets.LAKEFS_SECRET_KEY }}
          MINIO_ENDPOINT: http://172.17.0.1:9000
          MINIO_ACCESS_KEY: ${{ secrets.MINIO_ACCESS_KEY }}
          MINIO_SECRET_KEY: ${{ secrets.MINIO_SECRET_KEY }}
          LAKEFS_REPO: ${{ vars.LAKEFS_REPO }}
          BRANCH: ${{ vars.LAKEFS_BRANCH }}
        run: uv run python scripts/seed_data.py

      - name: Compile training pipeline
        run: |
          uv run python -c "
          from kfp import compiler
          from pipelines.training_pipeline import training_pipeline
          compiler.Compiler().compile(training_pipeline, '/tmp/training_pipeline.yaml')
          print('Pipeline compiled.')
          "

      - name: Authenticate and submit to Kubeflow
        env:
          KUBEFLOW_HOST: ${{ vars.KUBEFLOW_HOST }}
          KUBEFLOW_NAMESPACE: ${{ vars.KUBEFLOW_NAMESPACE }}
          KUBEFLOW_USERNAME: ${{ vars.DEX_USERNAME }}
          KUBEFLOW_PASSWORD: ${{ secrets.DEX_PASSWORD }}
        run: |
          uv run python - <<'PYEOF'
          import os
          from kfp_utils.auth import get_kfp_client

          client = get_kfp_client(
              host=os.environ["KUBEFLOW_HOST"],
              namespace=os.environ["KUBEFLOW_NAMESPACE"],
              username=os.environ["KUBEFLOW_USERNAME"],
              password=os.environ["KUBEFLOW_PASSWORD"],
          )
          run = client.create_run_from_pipeline_package(
              pipeline_file="/tmp/training_pipeline.yaml",
              arguments={},
              run_name="tron-training-ci",
              experiment_name="tron-ci-runs",
          )
          print(f"Pipeline run submitted: {run.run_id}")
          PYEOF
```

---

## Implementation Steps

Execute the following steps in order from `/workspace/git_repo/mlops/`.

### Phase 0: Verify cluster credential injection

`install_all.sh` runs `create_mlops_secret.sh`, which applies the
`k8s_yamls/kubeflow/mlops-poddefault.yaml` PodDefault. This resource injects the
`mlops-credentials` Secret and `pipeline-configmap` ConfigMap into every pod in the
`user-example-com` namespace, including all KFP pipeline pods. Without it every pod
crashes immediately with `KeyError` on the first `os.environ[]` access.

Confirm it is present before submitting any pipeline:

```bash
kubectl -n user-example-com get poddefault mlops-credentials
```

If the output is empty, `install_all.sh` was not run or the PodDefault was deleted.
Re-apply it:

```bash
kubectl apply -f k8s_yamls/kubeflow/mlops-poddefault.yaml
```

---

### Phase 1: Create and push ml-components

**Step 1.1 — Create all source files**

Create the complete directory tree under `repos/ml-platform/ml-components/` using the file contents defined above. Each component is an independent package with its own `pyproject.toml`, `manifest.yaml`, and source directory under `ml_components_<name>/`.

**Step 1.2 — Push ml-components to Forgejo**

```bash
cd /workspace/git_repo/mlops
bash create_forgejo_repo.sh ml-platform ml-components repos/ml-platform/ml-components
```

This creates the `ml-platform` org, the `ml-components` repo, sets all CI variables and secrets (including `PYPI_INDEX_URL`), and pushes the source.

**Step 1.3 — Add Forgejo publish secrets to ml-components**

The publish workflow needs `FJ_USERNAME` and `FJ_TOKEN` secrets that
`create_forgejo_repo.sh` does not set. Read Forgejo credentials from the same k8s
YAML source the install scripts use, then create a scoped API token and set both
secrets:

```bash
# Read Forgejo credentials from k8s YAML (same source as install_forgejo.sh)
FORGEJO_USER=$(grep 'admin-username:' k8s_yamls/forgejo/02-secret.yaml | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_PASSWORD=$(grep 'admin-password:' k8s_yamls/forgejo/02-secret.yaml | awk -F': ' '{print $2}' | tr -d '"')

# Create a scoped API token for the publish workflow
FORGEJO_TOKEN=$(curl -s -X POST "http://172.17.0.1:4000/api/v1/users/${FORGEJO_USER}/tokens" \
  -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" \
  -H "Content-Type: application/json" \
  -d '{"name": "ml-components-ci", "scopes": ["write:package", "read:package"]}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['sha1'])")

BASE="http://172.17.0.1:4000/api/v1/repos/ml-platform/ml-components"

curl -s -X PUT "$BASE/actions/secrets/FJ_USERNAME" \
  -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" \
  -H "Content-Type: application/json" \
  -d "{\"data\": \"${FORGEJO_USER}\"}"

curl -s -X PUT "$BASE/actions/secrets/FJ_TOKEN" \
  -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" \
  -H "Content-Type: application/json" \
  -d "{\"data\": \"${FORGEJO_TOKEN}\"}"
```

**Step 1.4 — Enable branch protection on ml-components/main**

```bash
FORGEJO_USER=$(grep 'admin-username:' k8s_yamls/forgejo/02-secret.yaml | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_PASSWORD=$(grep 'admin-password:' k8s_yamls/forgejo/02-secret.yaml | awk -F': ' '{print $2}' | tr -d '"')

curl -s -X POST "http://172.17.0.1:4000/api/v1/repos/ml-platform/ml-components/branch_protections" \
  -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" \
  -H "Content-Type: application/json" \
  -d '{
    "branch_name": "main",
    "required_approvals": 1,
    "enable_push": false,
    "enable_push_whitelist": false,
    "block_on_rejected_reviews": false
  }'
```

**Step 1.5 — Wait for CI to publish components to Forgejo PyPI**

After push, the Forgejo CI runner will execute `publish.yaml`. This publishes each component as an independent package to the Forgejo PyPI registry. Wait for it to complete before proceeding to Phase 2.

Verify the packages are available:
```bash
curl -s "http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/" | grep ml-components
```

If the CI fails (e.g. because the base Docker image `ml-components/base:latest` does not exist), the `ensure-base-image` job builds it automatically. The component spec push and Python package publish steps run independently.

---

### Phase 2: Create and push team-tron/pipelines

**Step 2.1 — Create all source files**

Create the complete directory tree under `repos/team-tron/pipelines/` using the file contents defined above. Include all `__init__.py` files.

**Step 2.2 — Generate team-tron/pipelines lock file**

The lock file must resolve all 5 component packages from the Forgejo PyPI registry. Confirm Phase 1 Step 1.5 is complete before running this.

```bash
cd repos/team-tron/pipelines
uv lock
cd ../../..
```

**Step 2.3 — Push team-tron/pipelines to Forgejo**

```bash
cd /workspace/git_repo/mlops
bash create_forgejo_repo.sh team-tron pipelines repos/team-tron/pipelines
```

**Step 2.4 — Add LAKEFS_REPO and LAKEFS_BRANCH variables to team-tron/pipelines**

`create_forgejo_repo.sh` (Step 2.3) already sets all credential secrets
(`LAKEFS_ACCESS_KEY`, `LAKEFS_SECRET_KEY`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`,
etc.) by reading from k8s YAML files. The only additional configuration needed is
the two non-sensitive variables that tell the seed script which LakeFS repo and
branch to target:

```bash
# Read Forgejo credentials from k8s YAML (same source as install_forgejo.sh)
FORGEJO_USER=$(grep 'admin-username:' k8s_yamls/forgejo/02-secret.yaml | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_PASSWORD=$(grep 'admin-password:' k8s_yamls/forgejo/02-secret.yaml | awk -F': ' '{print $2}' | tr -d '"')

BASE="http://172.17.0.1:4000/api/v1/repos/team-tron/pipelines"

curl -s -X POST "$BASE/actions/variables/LAKEFS_REPO" \
  -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" \
  -H "Content-Type: application/json" \
  -d '{"value": "mlops-data-dev"}'

curl -s -X POST "$BASE/actions/variables/LAKEFS_BRANCH" \
  -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" \
  -H "Content-Type: application/json" \
  -d '{"value": "main"}'
```

After this step, every push to `main` in `team-tron/pipelines` will automatically
seed the dataset (if not already present) and then compile and submit the pipeline —
no manual data setup is ever required.

---

## Verification Checklist

After both phases complete, verify the following:

```bash
# Read Forgejo credentials from the same k8s YAML source as install_forgejo.sh
FORGEJO_USER=$(grep 'admin-username:' k8s_yamls/forgejo/02-secret.yaml | awk -F': ' '{print $2}' | tr -d '"')
FORGEJO_PASSWORD=$(grep 'admin-password:' k8s_yamls/forgejo/02-secret.yaml | awk -F': ' '{print $2}' | tr -d '"')

# 1. Forgejo orgs and repos exist
curl -s -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" http://172.17.0.1:4000/api/v1/orgs/ml-platform | python3 -c "import sys,json; print('ml-platform org:', json.load(sys.stdin).get('username'))"
curl -s -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" http://172.17.0.1:4000/api/v1/orgs/team-tron | python3 -c "import sys,json; print('team-tron org:', json.load(sys.stdin).get('username'))"

# 2. Both repos exist
curl -s -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" http://172.17.0.1:4000/api/v1/repos/ml-platform/ml-components | python3 -c "import sys,json; d=json.load(sys.stdin); print('ml-components repo:', d.get('name'))"
curl -s -u "${FORGEJO_USER}:${FORGEJO_PASSWORD}" http://172.17.0.1:4000/api/v1/repos/team-tron/pipelines | python3 -c "import sys,json; d=json.load(sys.stdin); print('pipelines repo:', d.get('name'))"

# 3. Component packages are in Forgejo PyPI
curl -s "http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/ml-components-data-ingestion/"
curl -s "http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/ml-components-data-validation/"
curl -s "http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/ml-components-feature-engineering/"
curl -s "http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/ml-components-model-training/"
curl -s "http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/ml-components-model-evaluation/"

# 4. Component specs in ZOT (run after CI publish completes)
curl -s http://localhost:8001/v2/ml-components/data-ingestion-spec/tags/list
curl -s http://localhost:8001/v2/ml-components/data-validation-spec/tags/list
curl -s http://localhost:8001/v2/ml-components/feature-engineering-spec/tags/list
curl -s http://localhost:8001/v2/ml-components/model-training-spec/tags/list
curl -s http://localhost:8001/v2/ml-components/model-evaluation-spec/tags/list

# 5. Catalog artifact exists in ZOT
oras pull --plain-http localhost:8001/ml-components/catalog:latest --output /tmp/catalog-check
cat /tmp/catalog-check/catalog.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Catalog has {len(d[\"components\"])} components')"
```

---

## Key Pitfalls to Avoid

1. **Wrong host in CI workflows**: Use `172.17.0.1:<port>` in all `.forgejo/workflows/*.yaml` files. Never use `localhost` or Kubernetes DNS inside CI workflow steps.

2. **Missing `__init__.py` files**: Each `ml_components_<name>/` directory MUST have an `__init__.py` that re-exports the component function. Without it, imports like `from ml_components_data_ingestion import data_ingestion` will fail.

3. **Lock file not committed**: Both repos must have `uv.lock` committed. CI uses `uv sync --frozen` which fails if the lock file is missing or out of sync with `pyproject.toml`.

4. **Version bump not done**: Any future change to a component's `manifest.yaml` requires a version bump or CI will reject the PR. The initial version for all components is `1.0.0`.

5. **FJ_USERNAME / FJ_TOKEN not set**: The `publish.yaml` workflow's `publish-python-package` job will fail at the `twine upload` step if these secrets are missing. Add them via Phase 1 Step 1.3 before pushing.

6. **`typical_upstream` must be non-empty**: The `manifest-schema.json` requires `typical_upstream` to have at least one entry. For `data_ingestion`, use `["pipeline-start"]` as a sentinel value.

7. **Component output names in pipelines**: The pipeline code accesses outputs via `.outputs["name"]` — the key must exactly match the `NamedTuple` field names in the component signature. The names defined above are authoritative.

8. **Credentials must come from the environment — never from code**: `seed_data.py` and all pipeline components call `os.environ["SECRET_KEY"]` with no fallback default. Missing secrets produce an immediate, descriptive error. For local development, run `eval "$(bash scripts/get_local_env.sh)"` which reads credentials from the same k8s YAML files as `install_forgejo.sh`. For CI, `create_forgejo_repo.sh` sets all necessary secrets automatically from those same YAML files.

9. **No data = instant pipeline failure**: `data_validation` will raise on any dataset with fewer than `min_rows` rows, and `data_ingestion` will raise if the commit-keyed Parquet does not exist in MinIO. The CI `submit.yaml` runs `scripts/seed_data.py` before every pipeline submission (it is idempotent and skips if data already exists). For local runs, `run_local.py` calls `seed_data()` automatically. No manual data setup is ever required.

10. **PodDefault must exist before any pipeline run**: KFP pods inherit service credentials and endpoint URLs via the `mlops-credentials` PodDefault, applied by `create_mlops_secret.sh` during `install_all.sh`. If it is missing, every pipeline pod crashes with `KeyError` on the first `os.environ[]` access. Verify with `kubectl -n user-example-com get poddefault mlops-credentials` and re-apply from `k8s_yamls/kubeflow/mlops-poddefault.yaml` if absent.

11. **Import path inconsistency**: `training_pipeline.py` imports via `__init__.py` (e.g., `from ml_components_data_ingestion import data_ingestion`) while `retraining_pipeline.py` and `evaluation_pipeline.py` import directly from `.component` (e.g., `from ml_components_data_ingestion.component import data_ingestion`). Both styles work because each component's `__init__.py` re-exports the function, but for consistency, prefer the `__init__.py` import style.

12. **Feature engineering excludes label column**: The `feature_engineering` component explicitly excludes the `label` column from StandardScaler normalization. This is intentional — scaling the label would corrupt the target variable. Do not remove this guard.
