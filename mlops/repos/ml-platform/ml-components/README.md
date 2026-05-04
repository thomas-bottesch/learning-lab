# ml-components

Reusable Kubeflow pipeline components for the MLOps platform.

## Components

| Component | Category | Description |
|-----------|----------|-------------|
| `data-ingestion` | ingestion | Resolves LakeFS commit, returns dataset path + commit hash |
| `data-validation` | validation | Gate: row count, null fraction, all-zero checks |
| `feature-engineering` | feature-engineering | StandardScaler normalization, write Parquet to MinIO |
| `model-training` | training | Train LogisticRegression, log to MLflow, save artifact |
| `model-evaluation` | evaluation | Compute accuracy, return pass/fail evaluation report |

## Pipeline order

    data-ingestion → data-validation → feature-engineering → model-training → model-evaluation

## Usage

Each component is an independent package. Import individually:

```python
from ml_components_data_ingestion.component import data_ingestion
from ml_components_data_validation.component import data_validation
from ml_components_feature_engineering.component import feature_engineering
from ml_components_model_training.component import model_training
from ml_components_model_evaluation.component import model_evaluation
```

Or install only the components you need:

```python
# Minimal install — only import what you use
from ml_components_model_training.component import model_training
```

## Development

Each component is an independent package. Install and test individually:

```bash
cd components/data_ingestion
uv sync --extra dev
uv run pytest tests/ -v
```

Or run all component tests from the repo root (requires installing each component's dev dependencies manually):

```bash
for comp in components/*/; do
    (cd "$comp" && uv sync --extra dev && uv run pytest tests/ -v)
done
```

See `AGENTS.md` for authoring guidelines.
