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
