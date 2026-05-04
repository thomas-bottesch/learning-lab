import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

import json
import pickle
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ENV = {
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY": "test-access-key",
    "MINIO_SECRET_KEY": "test-secret-key",
}


def make_model_bytes() -> bytes:
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(max_iter=200).fit(X, y)
    return pickle.dumps(model)


def make_features_parquet() -> bytes:
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "label": y})
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def test_passing_model_returns_report():
    from ml_components_model_evaluation.component import model_evaluation

    mock_s3 = MagicMock()
    responses = [
        {"Body": MagicMock(read=make_model_bytes)},
        {"Body": MagicMock(read=make_features_parquet)},
    ]
    mock_s3.get_object.side_effect = responses

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        result = model_evaluation.python_func(
            model_artifact="s3://models/tron/run1/model.pkl",
            features_path="s3://datasets/repo/abc/features.parquet",
            min_accuracy=0.0,
        )

    report = json.loads(result.evaluation_report)
    assert report["status"] == "passed"
    assert isinstance(result.accuracy, float)


def test_low_accuracy_raises():
    from ml_components_model_evaluation.component import model_evaluation

    mock_s3 = MagicMock()
    responses = [
        {"Body": MagicMock(read=make_model_bytes)},
        {"Body": MagicMock(read=make_features_parquet)},
    ]
    mock_s3.get_object.side_effect = responses

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        with pytest.raises(ValueError, match="accuracy"):
            model_evaluation.python_func(
                model_artifact="s3://models/tron/run1/model.pkl",
                features_path="s3://datasets/repo/abc/features.parquet",
                min_accuracy=1.0,
            )
