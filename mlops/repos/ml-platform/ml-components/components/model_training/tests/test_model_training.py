import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import pandas as pd
import numpy as np

ENV = {
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY": "test-access-key",
    "MINIO_SECRET_KEY": "test-secret-key",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
}


def make_features_parquet() -> bytes:
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "label": np.random.randint(0, 2, n),
        }
    )
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def test_returns_model_artifact_and_run_id():
    from ml_components_model_training.component import model_training

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=make_features_parquet)}
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-id-123"

    with patch("boto3.client", return_value=mock_s3), patch.dict(
        "os.environ", ENV
    ), patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"), patch(
        "mlflow.start_run", return_value=mock_run
    ), patch(
        "mlflow.log_params"
    ), patch(
        "mlflow.log_metric"
    ), patch(
        "mlflow.log_param"
    ):
        result = model_training.python_func(
            features_path="s3://datasets/repo/abc/features.parquet",
            epochs=1,
            max_iter=50,
        )

    assert "s3://models/" in result.model_artifact
    assert result.mlflow_run_id == "test-run-id-123"
