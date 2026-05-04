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
