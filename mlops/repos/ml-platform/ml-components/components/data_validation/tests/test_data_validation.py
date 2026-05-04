import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import pandas as pd


def make_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


ENV = {
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY": "test-access-key",
    "MINIO_SECRET_KEY": "test-secret-key",
}


def test_valid_dataset_passes():
    from ml_components_data_validation.component import data_validation

    df = pd.DataFrame({"a": range(1500), "b": [1.0] * 1500})
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: make_parquet_bytes(df))
    }

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        result = data_validation.python_func(
            dataset_path="s3://datasets/repo/abc/data.parquet",
            lakefs_commit="abc123",
        )

    report = json.loads(result.validation_report)
    assert report["status"] == "passed"
    assert result.dataset_path == "s3://datasets/repo/abc/data.parquet"


def test_too_few_rows_raises():
    from ml_components_data_validation.component import data_validation

    df = pd.DataFrame({"a": range(5), "b": [1.0] * 5})
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: make_parquet_bytes(df))
    }

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        with pytest.raises(ValueError, match="row count"):
            data_validation.python_func(
                dataset_path="s3://datasets/repo/abc/data.parquet",
                lakefs_commit="abc123",
                min_rows=1000,
            )
