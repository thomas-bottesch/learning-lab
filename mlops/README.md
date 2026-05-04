# MLOps End-To-End Simulation

A fully self-hosted local MLOps platform that simulates an elite ML team pipeline using open-source components. The platform demonstrates **Kubeflow component caching**, **LakeFS dataset versioning**, and **MLflow experiment reproducibility**.

**Kubeflow dashboard:** http://localhost:8080  
**Default user:** `user@example.com` / `12341234`

---

## Quick Start

### Step 1 — Start the Kubernetes cluster

```bash
python3 host_config/k8s_cluster.py --start
```

Installs k3s on the host with Docker as the container runtime.

### Step 2 — Install all infrastructure services

```bash
bash install_all.sh
```

Installs services in dependency order:

| # | Service | Purpose | Port |
|---|---------|---------|------|
| 1 | [LakeFS](INFRASTRUCTURE_SPEC.md#lakefs) | Data versioning (Git for data) | 8000 |
| 2 | [MinIO](INFRASTRUCTURE_SPEC.md#minio) | S3-compatible object storage | 9000/9001 |
| 3 | [Zot](INFRASTRUCTURE_SPEC.md#zot) | OCI container registry | 8001 |
| 4 | [Forgejo](INFRASTRUCTURE_SPEC.md#forgejo) | Git platform + CI/CD | 4000/30422 |
| 5 | [MLflow](INFRASTRUCTURE_SPEC.md#mlflow) | ML experiment tracking | 5000 |
| 6 | [Kubeflow](INFRASTRUCTURE_SPEC.md#kubeflow) | ML pipeline orchestration | 8080 |

Then creates credential injection for pipelines:

```bash
bash create_mlops_configmap.sh   # Service endpoints → k8s ConfigMap
bash create_mlops_secret.sh      # Credentials → k8s Secret
```

### Step 3 — Create and populate the components repository

```bash
# Create repo and push source contents — triggers CI publish workflow
bash create_forgejo_repo.sh ml-platform ml-components repos/ml-platform/ml-components/
```

The **publish workflow** (`publish.yaml`) triggered by the push:

1. Detects which component directories changed
2. Builds each component's Docker image locally (via Forgejo runner's Docker socket)
3. Pushes component specs (YAML manifests) to the Zot OCI registry
4. Builds and publishes Python packages to Forgejo PyPI
5. Generates and pushes a component catalog to Zot

After push, add Forgejo publish secrets (`FJ_USERNAME` / `FJ_TOKEN`) required for PyPI uploads (see `ML_REPO_STRATEGY_SPEC.md` Phase 1 Step 1.3).

Verify packages are published:

```bash
curl -s "http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/" | grep ml-components
```

### Step 4 — Create and populate the pipelines repository

```bash
# Create repo and push source contents — triggers CI workflows
bash create_forgejo_repo.sh team-tron pipelines repos/team-tron/pipelines
```

The `team-tron/pipelines` repo contains 3 KFP pipelines that consume all 5 components from the `ml-platform/ml-components` repo:

| Pipeline | Purpose |
|----------|---------|
| [`training_pipeline.py`](repos/team-tron/pipelines/pipelines/training_pipeline.py) | Full supervised training run |
| [`retraining_pipeline.py`](repos/team-tron/pipelines/pipelines/retraining_pipeline.py) | Hyperparameter-tunable retrain |
| [`evaluation_pipeline.py`](repos/team-tron/pipelines/pipelines/evaluation_pipeline.py) | Quick data + model quality check |

After push, add LakeFS repo/branch variables (see `ML_REPO_STRATEGY_SPEC.md` Phase 2 Step 2.4).

---

## What This Simulation Demonstrates

The `team-tron/pipelines` repo uses Forgejo CI to automatically run pipelines on every push to `main`. The **submit workflow** (`submit.yaml`) does:

1. **Seed data** — Creates a synthetic dataset in MinIO and registers it in LakeFS
2. **Compile pipeline** — Compiles the KFP DAG to YAML
3. **Authenticate & submit** — Logs into Kubeflow via Dex OAuth and submits the run

This demonstrates three core MLOps capabilities:

### Kubeflow Component Caching

Each of the 5 components (`data-ingestion`, `data-validation`, `feature-engineering`, `model-training`, `model-evaluation`) is a standalone KFP component published to Forgejo PyPI. When a component's input parameters and source code are unchanged, Kubeflow skips execution and returns cached output — dramatically speeding up iterative development.

### Dataset Versioning with LakeFS

The `data-ingestion` component resolves a LakeFS commit hash before reading data, ensuring every pipeline run uses an immutable snapshot. LakeFS tracks dataset changes like Git tracks code, enabling full data lineage.

### Reproducibility with MLflow

The `model-training` component logs all parameters, metrics, and artifacts to MLflow. Every model is tied to a specific run ID, dataset commit, and code version — making experiments fully reproducible.

---

## Architecture

### The 5 Components (ml-platform/ml-components)

| Component | Role |
|-----------|------|
| `data_ingestion` | Resolves LakeFS commit, returns dataset path + commit hash |
| `data_validation` | Gate: row count, null fraction, all-zero checks |
| `feature_engineering` | StandardScaler normalization, writes Parquet to MinIO |
| `model_training` | Trains LogisticRegression, logs to MLflow, saves artifact |
| `model_evaluation` | Computes accuracy, returns pass/fail report |

Pipeline DAG: `ingestion → validation → engineering → training → evaluation`

### Networking

| Context | Host Address |
|---------|-------------|
| From devcontainer / host | `localhost:<port>` |
| From Forgejo runner (Docker on host) | `172.17.0.1:<port>` |
| From Kubeflow pipeline Pods | `<service>.<namespace>.svc.cluster.local:<port>` |

---

## References

- [Infrastructure Details](INFRASTRUCTURE_SPEC.md) — Service configs, credentials, backup/restore
- [ML Repo Strategy](ML_REPO_STRATEGY_SPEC.md) — Component monorepo + pipeline repo design, full file contents
