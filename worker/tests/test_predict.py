"""Tests for predict path (M12): dispatch, tabular, CLI."""

import json
import sys
from pathlib import Path
import subprocess
import pytest

WORKER_ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"


def test_predict_dispatch_routing(monkeypatch, tmp_path):
    """Verify dispatch_predict routes to correct modality predict function."""
    from doclab_worker.predict import dispatch_predict

    # Mock each modality's predict function
    tabular_called = []
    image_called = []
    text_called = []

    def mock_tabular_predict(checkpoints_dir, input_value):
        tabular_called.append(True)
        return {"prediction": "mock", "confidence": 0.9}

    def mock_image_predict(checkpoints_dir, input_value):
        image_called.append(True)
        return {"prediction": "mock", "confidence": 0.9}

    def mock_text_predict(checkpoints_dir, input_value):
        text_called.append(True)
        return {"prediction": "mock", "confidence": 0.9}

    monkeypatch.setattr("doclab_worker.tabular.predict", mock_tabular_predict)
    monkeypatch.setattr("doclab_worker.image.predict", mock_image_predict)
    monkeypatch.setattr("doclab_worker.text.predict", mock_text_predict)

    # Test tabular routing
    manifest = {"schema_version": 1, "modality": "tabular"}
    input_data = {"type": "tabular_json", "value": "{}"}
    dispatch_predict(manifest, input_data, tmp_path, tmp_path)
    assert len(tabular_called) == 1

    # Test image routing
    manifest = {"schema_version": 1, "modality": "image"}
    input_data = {"type": "image_path", "value": "/path/to/image.jpg"}
    dispatch_predict(manifest, input_data, tmp_path, tmp_path)
    assert len(image_called) == 1

    # Test text routing
    manifest = {"schema_version": 1, "modality": "text"}
    input_data = {"type": "text", "value": "some text"}
    dispatch_predict(manifest, input_data, tmp_path, tmp_path)
    assert len(text_called) == 1


def test_tabular_predict_with_mock_checkpoint(monkeypatch, tmp_path):
    """Test tabular predict with mocked checkpoint files."""
    from doclab_worker.tabular import predict
    import pandas as pd

    # Create fake checkpoint structure
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    # Write manifest
    manifest = {
        "schema_version": 1,
        "modality": "tabular",
        "label_column": "target",
        "model_type": "xgboost",
    }
    (checkpoints_dir / "manifest.json").write_text(json.dumps(manifest))

    # Write preprocess metadata
    preprocess = {"feature_columns": ["age", "glucose"]}
    (checkpoints_dir / "preprocess.json").write_text(json.dumps(preprocess))

    # Create a fake model.joblib file (just touch it)
    (checkpoints_dir / "model.joblib").write_bytes(b"fake model")

    # Mock joblib.load to return a fake model
    class FakeModel:
        def predict_proba(self, X):
            return [[0.3, 0.7]]  # Binary classification probabilities

    def mock_joblib_load(path):
        return FakeModel()

    monkeypatch.setattr("joblib.load", mock_joblib_load)

    # Run prediction
    input_json = json.dumps({"age": 50, "glucose": 120})
    result = predict(checkpoints_dir, input_json)

    # Verify result structure
    assert "prediction" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], (int, float))


def test_predict_cli_missing_checkpoint(tmp_path):
    """Test --predict CLI with missing checkpoint returns error."""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    request = {
        "schema_version": 1,
        "experiment_id": "test_exp",
        "experiment_dir": str(exp_dir),
        "input": {"type": "tabular_json", "value": "{}"},
    }
    request_path = tmp_path / "predict_request.json"
    request_path.write_text(json.dumps(request))

    result = subprocess.run(
        [sys.executable, "-m", "doclab_worker", "--predict", str(request_path)],
        cwd=str(WORKER_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    error_path = tmp_path / "error.json"
    assert error_path.exists()
    error = json.loads(error_path.read_text())
    # Error gets wrapped but message should indicate missing checkpoint
    assert "checkpoints" in error["message"].lower() or error["code"] == "checkpoint_missing"


def test_predict_cli_invalid_input_type(tmp_path):
    """Test --predict CLI with wrong input type for modality."""
    # Create experiment dir with checkpoint
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir()
    manifest = {"schema_version": 1, "modality": "tabular"}
    (checkpoints_dir / "manifest.json").write_text(json.dumps(manifest))

    # Wrong input type for tabular
    request = {
        "schema_version": 1,
        "experiment_id": "test_exp",
        "experiment_dir": str(exp_dir),
        "input": {"type": "image_path", "value": "/path/to/image.jpg"},  # Wrong type
    }
    request_path = tmp_path / "predict_request.json"
    request_path.write_text(json.dumps(request))

    result = subprocess.run(
        [sys.executable, "-m", "doclab_worker", "--predict", str(request_path)],
        cwd=str(WORKER_ROOT),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    error_path = tmp_path / "error.json"
    assert error_path.exists()
    error = json.loads(error_path.read_text())
    # Error is caught and wrapped as "unknown" but message should indicate the issue
    assert "tabular requires input.type='tabular_json'" in error["message"]


def test_image_predict_dispatch(monkeypatch, tmp_path):
    """Test image predict is called for image modality."""
    from doclab_worker.predict import dispatch_predict

    image_called = []

    def mock_image_predict(checkpoints_dir, input_value):
        image_called.append(input_value)
        return {
            "prediction": "NORMAL",
            "confidence": 0.85,
            "detail": "Mock image prediction",
            "warning": "Research only",
        }

    monkeypatch.setattr("doclab_worker.image.predict", mock_image_predict)

    manifest = {"schema_version": 1, "modality": "image"}
    input_data = {"type": "image_path", "value": "/test/image.jpg"}
    result = dispatch_predict(manifest, input_data, tmp_path, tmp_path)

    assert len(image_called) == 1
    assert image_called[0] == "/test/image.jpg"
    assert result["prediction"] == "NORMAL"


def test_text_predict_dispatch(monkeypatch, tmp_path):
    """Test text predict is called for text modality."""
    from doclab_worker.predict import dispatch_predict

    text_called = []

    def mock_text_predict(checkpoints_dir, input_value):
        text_called.append(input_value)
        return {
            "prediction": "Summary text",
            "confidence": None,
            "detail": "Mock text prediction",
            "warning": "Research only",
        }

    monkeypatch.setattr("doclab_worker.text.predict", mock_text_predict)

    manifest = {"schema_version": 1, "modality": "text"}
    input_data = {"type": "text", "value": "Long medical text to summarize"}
    result = dispatch_predict(manifest, input_data, tmp_path, tmp_path)

    assert len(text_called) == 1
    assert text_called[0] == "Long medical text to summarize"
    assert result["prediction"] == "Summary text"
