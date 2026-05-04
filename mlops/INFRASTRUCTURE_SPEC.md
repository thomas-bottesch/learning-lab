# MLOps Platform Specification

This document describes the architecture, infrastructure, and operational details of the MLOps platform. It is intended to be used as a reference by agents and engineers working on the project.

---

## Overview

This project provides a fully self-hosted, local MLOps platform running on a k3s Kubernetes cluster. It simulates the infrastructure of an elite ML team pipeline using entirely open-source components.

### Components

| Service | Purpose | Port |
|---------|---------|------|
| LakeFS | Data versioning (Git for data) | 8000 |
| MinIO | S3-compatible object storage | 9000 (API), 9001 (Console) |
| Zot | OCI container registry | 8001 |
| Forgejo | Git platform + CI/CD | 4000 (HTTP), 30422 (SSH) |
| MLflow | ML experiment tracking | 5000 |
| Kubeflow | ML pipeline orchestration | 8080 |

---

## Infrastructure Architecture

### Three Environments

There are three distinct environments that interact with each other:

1. **Host machine** — Runs the Docker daemon and the k3s Kubernetes cluster. The Docker daemon is configured with the fixed bridge IP `172.17.0.1/16`. This IP is the host's address reachable from within Docker containers and from the k3s cluster nodes.

2. **VSCode devcontainer** — All development work and commands (kubectl, helm, bash scripts) are executed here. The devcontainer uses `network=host`, which means `localhost` inside the devcontainer is identical to `localhost` on the host. The devcontainer shares the host's Docker socket and is configured with the host's kubeconfig.

3. **k3s cluster** — A local Kubernetes cluster running on the host, using Docker as its container runtime. Services are exposed via LoadBalancer type Kubernetes Services, which makes them accessible on `localhost` from both the host and the devcontainer.

### Critical Networking Constraint

Forgejo CI/CD runners create Docker containers directly on the Docker daemon (not as Kubernetes Pods). This means:

- Kubernetes DNS (e.g., `service.namespace.svc.cluster.local`) is **not available** inside Forgejo workflow jobs.
- `localhost` inside a Forgejo runner container **does not** refer to the devcontainer or host localhost.
- To communicate with any Kubernetes service from a Forgejo workflow, use **`172.17.0.1:<port>`** as the host address.

This is why `create_forgejo_repo.sh` sets `KUBEFLOW_HOST="http://172.17.0.1:8080"` as an Actions variable rather than using a Kubernetes DNS name or localhost.

---

## Directory Structure

```
mlops/
├── host_config/                # Host-level setup (k3s, Docker daemon)
│   ├── k8s_cluster.py          # CLI to start/stop the k3s cluster
│   ├── install_k3s.sh          # k3s installer script
│   ├── daemon.json             # Docker daemon config (overlay2, MTU 1400, fixed CIDR)
│   ├── docker_images/
│   │   └── Dockerfile.ci       # Custom Docker image for Forgejo CI runners
│   └── save_lakefs_db.sh       # Backs up lakeFS database
├── k8s_yamls/                  # Kubernetes manifests, one subdirectory per service
│   ├── forgejo/
│   ├── kubeflow/
│   ├── lakefs/
│   ├── minio/
│   ├── mlflow/
│   └── zot/
├── container_config/           # k9s configuration
├── forgejo-prepared-db/        # Optional: pre-seeded Forgejo database for restore
├── lakefs-prepared-db/         # Optional: pre-seeded lakeFS database for restore
├── install_all.sh              # Master install script (runs all installs in order)
├── install_lakefs.sh
├── install_minio.sh
├── install_zot.sh
├── install_forgejo.sh
├── install_mlflow.sh
├── install_kubeflow.sh
├── uninstall_lakefs.sh
├── uninstall_forgejo.sh
├── uninstall_mlflow.sh
├── uninstall_zot.sh
├── create_forgejo_repo.sh      # Creates an org/repo in Forgejo with full CI/CD secrets
├── update_forgejo_repo.sh      # Pushes updated source to an existing Forgejo repo
├── create_mlops_configmap.sh   # Creates k8s ConfigMap with service endpoints
├── create_mlops_secret.sh      # Creates k8s Secret from extracted service credentials
├── save_forgejo_db.sh          # Backs up Forgejo database
├── start_notebook.sh           # Deploys a Jupyter notebook in Kubeflow
└── requirements.txt
```

---

## Startup Sequence

### 1. Start the k3s cluster (on the host)

```bash
python3 mlops/host_config/k8s_cluster.py --start
```

This installs k3s `v1.35.0+k3s3` on the host using Docker as the container runtime. It configures MTU settings to prevent network fragmentation issues.

### 2. Install all services (inside the devcontainer)

```bash
bash install_all.sh
```

Installation order is critical due to service dependencies:

1. `install_lakefs.sh` — Deploys LakeFS; can optionally restore from `lakefs-prepared-db/`
2. `install_minio.sh` — Deploys MinIO; creates buckets: `lakefs-data`, `mlflow-artifacts`, `zot-registry`, `dvc-data`, `datasets`, `models`
3. `install_zot.sh` — Deploys Zot; initializes `ml-components` OCI repository via ORAS
4. `install_forgejo.sh` — Deploys Forgejo; builds custom CI runner image from `Dockerfile.ci`; can optionally restore from `forgejo-prepared-db/`
5. `install_mlflow.sh` — Deploys MLflow
6. `install_kubeflow.sh` — Clones and applies the official Kubeflow manifests (`v1.11-branch`) using kustomize; configures user profile and LoadBalancer for the Istio gateway
7. `create_mlops_configmap.sh` — Creates `mlops-endpoints` ConfigMap in the `kubeflow-user-example-com` namespace (namespace created by step 6)
8. `create_mlops_secret.sh` — Creates `mlops-credentials` Secret by extracting credentials from LakeFS and MinIO secrets

---

## Service Details

### LakeFS

- **Image:** `treeverse/lakefs:1.77.0`
- **Namespace:** `lakefs`
- **Port:** 8000 (LoadBalancer)
- **Database:** SQLite at `/data/lakefs/metadata`
- **Blockstore backend:** MinIO (`s3://lakefs-data`)
- **Credentials:** Access key `AKIAJM7NJ6KZDV4UETMQ`, secret key in `k8s_yamls/lakefs/02-secret.yaml`
- **Backup/restore:** `host_config/save_lakefs_db.sh` and `lakefs-prepared-db/`

### MinIO

- **Image:** `minio/minio:latest`
- **Namespace:** `minio`
- **Ports:** 9000 (API), 9001 (Console), both LoadBalancer
- **Credentials:** `minioadmin` / `minioadmin123` (in `k8s_yamls/minio/02-secret.yaml`)
- **Buckets created on install:** `lakefs-data`, `mlflow-artifacts`, `zot-registry`, `dvc-data`, `datasets`, `models`

### Zot

- **Image:** `ghcr.io/project-zot/zot-linux-amd64:v2.1.11`
- **Namespace:** `zot`
- **Port:** 8001 (LoadBalancer → internal 5000)
- **Storage backend:** MinIO S3 bucket `zot-registry` via `http://minio.minio.svc.cluster.local:9000`
- **Pre-initialized repository:** `ml-components`
- **UI and search extensions** are enabled

### Forgejo

- **Image:** `data.forgejo.org/forgejo/forgejo:14.0.1-rootless`
- **Namespace:** `forgejo`
- **Ports:** 4000 HTTP (LoadBalancer → internal 3000), 30422 SSH (LoadBalancer)
- **Database:** SQLite at `/data/forgejo/forgejo.db`
- **Admin credentials:** `forgejo_admin` / `forgejo_password` (in `k8s_yamls/forgejo/02-secret.yaml`)
- **CI/CD Runner:** Forgejo Runner v12.7.0, Docker-based (not Kubernetes), capacity 1
  - Runner containers are built from `host_config/docker_images/Dockerfile.ci` (Python 3.12 base with ML dependencies)
  - Runner mounts the host Docker socket to start sibling containers
- **Backup/restore:** `save_forgejo_db.sh` and `forgejo-prepared-db/`

### MLflow

- **Image:** `ghcr.io/mlflow/mlflow:v2.16.1`
- **Namespace:** `mlflow`
- **Port:** 5000 (LoadBalancer)
- **Backend store:** SQLite at `/mlflow/mlflow.db`
- **Artifact store:** MinIO S3 bucket `mlflow-artifacts`
- **Resources:** 2–4 GiB RAM, 1–2 CPU cores

### Kubeflow

- **Source:** Official manifests from `https://github.com/kubeflow/manifests.git` (`v1.11-branch`)
- **Gateway port:** 8080 (LoadBalancer via Istio ingress in `istio-system`)
- **Default user:** `user@example.com` / `12341234`
- **Kubeflow namespace for user:** `kubeflow-user-example-com`
- **ConfigMap** `mlops-endpoints` and **Secret** `mlops-credentials` are injected into the user namespace to provide all service endpoints and credentials to pipelines
- **KServe RBAC:** `k8s_yamls/kubeflow/kserve-pipeline-rbac.yaml` — applied by `install_kubeflow.sh` after the main manifests. Grants the `default-editor` ServiceAccount (the SA all pipeline run Pods execute under) permission to create, read, update, and delete `InferenceService` resources in `kubeflow-user-example-com`. Required by the `model_serving` pipeline component, which calls `load_incluster_config()` and applies an `InferenceService` CRD at runtime. Without it the component fails with `403 Forbidden`. A `ClusterRole` is used so the same role definition can be rebound into additional team namespaces by adding a new `RoleBinding` — no `ClusterRole` duplication needed.

---

## Credential Reference

| Service | Username/Key | Password/Secret |
|---------|-------------|-----------------|
| LakeFS | `AKIAJM7NJ6KZDV4UETMQ` | See `k8s_yamls/lakefs/02-secret.yaml` |
| MinIO | `minioadmin` | `minioadmin123` |
| Forgejo | `forgejo_admin` | `forgejo_password` |
| MLflow | — (no auth) | — |
| Kubeflow | `user@example.com` | `12341234` |

---

## Setting Up a Repository in Forgejo

The `create_forgejo_repo.sh` script creates a Forgejo organization and repository, configures all CI/CD Actions secrets and variables, and pushes source code.

```bash
bash create_forgejo_repo.sh <ORG_NAME> <REPO_NAME> <SOURCE_DIR>
```

**What it does:**

1. Extracts Forgejo admin credentials from `k8s_yamls/forgejo/02-secret.yaml`
2. Waits for Forgejo to be available
3. Creates the organization (if it does not exist)
4. Creates the repository under that organization (if it does not exist)
5. Sets **Actions Variables** (non-sensitive):
   - `KUBEFLOW_HOST` = `http://172.17.0.1:8080` (uses host IP, not DNS, because runner is Docker-based)
   - `KUBEFLOW_USERNAME` / `DEX_USERNAME` = `user@example.com`
   - `MLFLOW_TRACKING_URI` = `http://mlflow.mlflow.svc.cluster.local:5000` (DNS works here because this variable is consumed by Kubeflow pipelines running inside the cluster, not by the runner directly)
   - `MINIO_ENDPOINT`, `LAKEFS_ENDPOINT`, bucket names, namespace — all extracted from ConfigMaps/manifests
   - `PYPI_INDEX_URL` = `http://172.17.0.1:4000/api/packages/ml-platform/pypi/simple/`
6. Sets **Actions Secrets** (sensitive): Kubeflow password, MinIO keys, LakeFS keys, AWS-compatible keys
7. Copies source files to `/tmp/<REPO_NAME>`, initializes git, and pushes to Forgejo

**Note:** When adding new workflows that call services from within a Forgejo runner, always use `172.17.0.1:<port>` instead of `localhost` or Kubernetes DNS names.

---

## Kubernetes ConfigMap and Secret for Pipelines

### ConfigMap: `mlops-endpoints` (in `kubeflow-user-example-com`)

Created by `create_mlops_configmap.sh`. Contains service endpoint URLs:
- `LAKEFS_ENDPOINT`
- `MINIO_ENDPOINT`
- `MLFLOW_TRACKING_URI`
- `LAKEFS_BUCKET_NAME`, `DVC_BUCKET_NAME`

### Secret: `mlops-credentials` (in `kubeflow-user-example-com`)

Created by `create_mlops_secret.sh`. Contains credentials extracted from service secrets:
- `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- `LAKEFS_ACCESS_KEY`, `LAKEFS_SECRET_KEY`

These are injected into Kubeflow pipeline steps running as Pods, allowing them to authenticate with LakeFS and MinIO using standard environment variable conventions.

---

## Database Backup and Restore

Both Forgejo and LakeFS support snapshot-based backup and restore to preserve state (user accounts, OAuth apps, registered repositories, LakeFS repositories and commits) across reinstalls.

| Service | Save script | Restore directory |
|---------|-------------|-------------------|
| Forgejo | `save_forgejo_db.sh` | `forgejo-prepared-db/` |
| LakeFS | `host_config/save_lakefs_db.sh` | `lakefs-prepared-db/` |

The install scripts detect whether a prepared database directory is present and restore from it automatically.

---

## Host Docker Daemon Configuration

File: `host_config/daemon.json`

Key settings:
- **Storage driver:** `overlay2`
- **MTU:** `1400` (required to avoid fragmentation over overlay networking)
- **Fixed CIDR:** `172.17.0.1/16` — this ensures a stable, predictable IP for the Docker bridge that is reachable from within the k3s cluster and from Forgejo runner containers
- **NVIDIA runtime:** configured for GPU support
- **Log rotation:** JSON log driver, max 100MB per file

---

## Adding New Services or Workflows

When adding a new service or Forgejo workflow, keep in mind:

- **Service accessible from Kubeflow pipelines:** use `service.namespace.svc.cluster.local` DNS names (Pods run inside the cluster).
- **Service accessible from Forgejo runner:** use `172.17.0.1:<port>` (runner containers run outside the cluster on the host Docker daemon).
- **New credentials:** add them as Kubernetes Secrets in the relevant namespace, and if needed for pipelines, extend `create_mlops_secret.sh` to include them in `mlops-credentials`.
- **New endpoints:** extend `create_mlops_configmap.sh` and `create_forgejo_repo.sh` accordingly.
- **Manifests:** follow the existing pattern in `k8s_yamls/<service>/` with numbered YAML files.

---