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
