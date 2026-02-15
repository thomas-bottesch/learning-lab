# Expert Pipeline Patterns for Kubeflow + LakeFS + MinIO

This document gives practical pipeline patterns tailored to your local single-node k3s stack.

## Pattern 1: Branch-Based Data + Train + Conditional Promote

Implemented in [lakefs_minio_training_pipeline.py](lakefs_minio_training_pipeline.py).

Flow:

1. Create a lakeFS feature branch from `main`.
2. Generate or ingest a dataset and write it into that branch.
3. Commit data changes in lakeFS.
4. Train model in Kubeflow from the branch data via lakeFS S3 gateway.
5. If metric threshold passes, merge feature branch back into `main`.

Why this is expert-level:

- Reproducibility through immutable lakeFS commits.
- Promotion gate based on model quality.
- Full lineage: branch -> object -> model metrics.

## Pattern 2: Multi-Stage Promotion (Dev -> Staging -> Prod)

Use separate long-lived lakeFS branches:

- `dev`
- `staging`
- `prod`

Suggested Kubeflow pipeline stages:

1. `prepare_data_dev`
2. `train_dev`
3. `validate_schema_and_drift`
4. `promote_dev_to_staging`
5. `train_staging`
6. `evaluate_staging`
7. `manual_or_policy_gate`
8. `promote_staging_to_prod`

For single-node k3s, keep this lightweight and use thresholds + optional manual approval in UI.

## Pattern 3: Scheduled Retraining with Backfill Safety

Run daily/weekly with a scheduled Kubeflow run:

1. Create branch `retrain-<date>` from `main`.
2. Load latest data snapshot (or rolling window) into the branch.
3. Train + compare against current production baseline metric.
4. Merge only if candidate outperforms baseline by margin.
5. Persist model artifact path and lakeFS commit in run metadata.

## Pattern 4: Champion/Challenger with Shadow Evaluation

1. Keep champion model metadata in MinIO (`models/champion/latest.json`).
2. Train challenger from latest branch snapshot.
3. Evaluate both on same holdout set.
4. Promote challenger only if statistically/operationally better.

## Minimal Hardening for Your Current Repo

1. Move access keys from pipeline d[text](../../save_lakefs_db.sh)efaults into Kubernetes Secrets and inject as env vars.
2. Add data quality step (null %, schema checks, class balance).
3. Add model registration metadata artifact (`model_card.json`).
4. Add run labels: `lakefs_branch`, `lakefs_commit`, `dataset_object`.

## Compile and submit

From `projects/kubeflow`:

```bash
python lakefs_minio_training_pipeline.py
```

Then upload generated YAML in Kubeflow Pipelines UI, or submit with your existing client utilities.

## Notes for your cluster

- Your services are reachable by cluster DNS:
  - lakeFS: `http://lakefs.lakefs.svc.cluster.local:8000`
  - MinIO: `http://minio.minio.svc.cluster.local:9000`
- For production, replace local SQLite lakeFS backend with PostgreSQL.
- For production, rotate credentials and avoid hardcoded defaults.
