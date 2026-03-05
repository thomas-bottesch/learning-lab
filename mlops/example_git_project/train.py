"""
Expert MLOps Training Script

This script demonstrates a production-ready training workflow with:
1. Data versioning with LakeFS
2. Experiment tracking with MLflow
3. Model storage in MinIO
4. Reproducible artifacts
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_version_data(client) -> pd.DataFrame:
    """Load data and log data versioning information."""
    logger.info("Loading iris dataset")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # Log data versioning configuration
    logger.info(f"Data versioning available via:")
    logger.info(f"  - LakeFS Endpoint: {client.config.get('lakefs_endpoint')}")
    logger.info(f"  - Repository: {client.config.get('lakefs_repository')}")
    logger.info(f"  - Data Bucket: {client.config.get('data_bucket', 'datasets')}")

    # Note: Actual data versioning would be handled by:
    # 1. The Kubeflow pipeline creating a LakeFS branch
    # 2. Training component reading from that branch
    # 3. This is configured in the pipeline, not here

    return df


def train_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Train and evaluate model."""
    logger.info("Training model")

    # Prepare features and target
    X = df.drop("target", axis=1).values
    y = df["target"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"Model accuracy: {accuracy:.4f}")

    return {
        "model": model,
        "accuracy": accuracy,
        "classification_report": report,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def log_to_mlflow(config: Dict[str, str], results: Dict[str, Any]) -> str:
    """Log experiment to MLflow using MLOpsClient."""
    from utils.mlops_utils import MLOpsClient

    client = MLOpsClient()

    with client.start_run(
        experiment_name=config["experiment_name"],
        run_name=f"iris-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tags={
            "project": "iris-classification",
            "model_type": "LogisticRegression",
        },
    ) as run:
        # Log parameters
        client.log_param("model_type", "LogisticRegression")
        client.log_param("max_iter", 200)
        client.log_param("random_state", 42)

        # Log metrics
        client.log_metric("accuracy", results["accuracy"])

        # Log model
        client.log_model(
            results["model"],
            artifact_path="model",
            registered_model_name="iris-classifier",
        )

        # Log classification report as JSON
        report_path = "/tmp/classification_report.json"
        with open(report_path, "w") as f:
            json.dump(results["classification_report"], f, indent=2)
        client.log_artifact(report_path)

        logger.info(f"MLflow run ID: {run.info.run_id}")
        return run.info.run_id


def save_artifacts(config: Dict[str, str], results: Dict[str, Any]):
    """Save artifacts to MinIO using MLOpsClient."""
    from utils.mlops_utils import MLOpsClient

    client = MLOpsClient()
    model_bucket = config["model_bucket"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_filename = f"iris_model_{timestamp}.joblib"
    model_path = f"/tmp/{model_filename}"
    joblib.dump(results["model"], model_path)

    # Upload to MinIO
    s3_url = client.upload_to_minio(
        file_path=model_path, bucket_name=model_bucket, object_name=model_filename
    )
    logger.info(f"Model saved to MinIO: {s3_url}")

    # Also save metrics
    metrics_filename = f"metrics_{timestamp}.json"
    metrics_path = f"/tmp/{metrics_filename}"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "accuracy": results["accuracy"],
                "classification_report": results["classification_report"],
                "timestamp": timestamp,
                "model_filename": model_filename,
                "mlflow_run_id": results.get("mlflow_run_id", "unknown"),
            },
            f,
            indent=2,
        )

    client.upload_to_minio(
        file_path=metrics_path, bucket_name=model_bucket, object_name=metrics_filename
    )
    logger.info(f"Metrics saved to MinIO: {model_bucket}/{metrics_filename}")


def main():
    """Main training workflow using MLOpsClient."""
    logger.info("Starting expert MLOps training workflow")

    try:
        # Create MLOps client once
        from utils.mlops_utils import MLOpsClient

        client = MLOpsClient()
        logger.info(f"MLOps client initialized")

        # Step 1: Load and version data
        df = load_and_version_data(client)

        # Step 2: Train model
        results = train_model(df)

        # Step 3: Log to MLflow
        run_id = log_to_mlflow(client, results)
        results["mlflow_run_id"] = run_id  # Store for later use

        # Step 4: Save artifacts
        save_artifacts(client, results)

        logger.info("Training workflow completed successfully")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"MLflow Run ID: {run_id}")

        # Return success
        return 0

    except Exception as e:
        logger.error(f"Training workflow failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
