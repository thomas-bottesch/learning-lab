import sys
from pathlib import Path

# Make the component package importable when running tests directly
_component_dir = Path(__file__).parent.parent
if str(_component_dir) not in sys.path:
    sys.path.insert(0, str(_component_dir))

from unittest.mock import patch, MagicMock


def test_returns_correct_path_and_commit():
    from ml_components_data_ingestion.component import data_ingestion

    mock_response = MagicMock()
    mock_response.json.return_value = {"commit_id": "abc123def456"}
    mock_response.raise_for_status = MagicMock()

    env = {
        "LAKEFS_ENDPOINT": "http://lakefs:8000",
        "LAKEFS_ACCESS_KEY": "key",
        "LAKEFS_SECRET_KEY": "secret",
    }
    with patch("requests.get", return_value=mock_response), patch.dict(
        "os.environ", env
    ):
        result = data_ingestion.python_func(lakefs_repo="test-repo", branch="main")

    assert result.lakefs_commit == "abc123def456"
    assert "abc123def456" in result.dataset_path
    assert result.dataset_path.startswith("s3://datasets/")


def test_raises_on_lakefs_error():
    from ml_components_data_ingestion.component import data_ingestion

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404")

    env = {
        "LAKEFS_ENDPOINT": "http://lakefs:8000",
        "LAKEFS_ACCESS_KEY": "key",
        "LAKEFS_SECRET_KEY": "secret",
    }
    with patch("requests.get", return_value=mock_response), patch.dict(
        "os.environ", env
    ):
        try:
            data_ingestion.python_func(lakefs_repo="missing-repo", branch="main")
            assert False, "Expected exception"
        except Exception:
            pass
