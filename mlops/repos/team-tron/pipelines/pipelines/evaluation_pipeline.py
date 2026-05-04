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
