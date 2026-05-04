import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

import pytest
from unittest.mock import patch, MagicMock, call
from io import BytesIO
import pandas as pd

ENV = {
    "MINIO_ENDPOINT": "http://minio:9000",
    "MINIO_ACCESS_KEY": "test-access-key",
    "MINIO_SECRET_KEY": "test-secret-key",
}


def test_features_path_replaces_data_with_features():
    from ml_components_feature_engineering.component import feature_engineering

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0] * 500, "y": [4.0, 5.0, 6.0] * 500})
    buf = BytesIO()
    df.to_parquet(buf, index=False)

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: buf.getvalue())}

    with patch("boto3.client", return_value=mock_s3), patch.dict("os.environ", ENV):
        result = feature_engineering.python_func(
            dataset_path="s3://datasets/repo/abc/data.parquet"
        )

    assert result.features_path == "s3://datasets/repo/abc/features.parquet"
    mock_s3.put_object.assert_called_once()
