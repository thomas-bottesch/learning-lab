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
