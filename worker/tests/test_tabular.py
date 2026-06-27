"""M2 tabular worker tests: determinism, metrics contract, baseline, errors."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TINY_CSV = FIXTURES / "tiny.csv"
WORKER_ROOT = Path(__file__).parent.parent

METRICS_KEYS = {
    "schema_version", "primary_metric", "metric_value", "baseline_metric",
    "device", "seed", "split", "n_train", "n_val", "n_test",
    "model_type", "framework", "checkpoint_dir",
}


def write_plan(tmp_path: Path, **overrides) -> Path:
    plan = {
        "schema_version": 1,
        "dataset_id": "tiny_fixture",
        "local_csv": str(TINY_CSV),
        "label_column": "readmitted",
        "model_type": "xgboost",
        "framework": "xgboost",
        "device": "cpu",
        "seed": 42,
        "split": [0.8, 0.1, 0.1],
        "primary_metric": "accuracy",
    }
    plan.update(overrides)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def run_worker(plan_path: Path):
    return subprocess.run(
        [sys.executable, "-m", "doclab_worker", "--job", str(plan_path)],
        cwd=str(WORKER_ROOT),
        capture_output=True,
        text=True,
    )


def test_metrics_shape_and_baseline(tmp_path):
    plan_path = write_plan(tmp_path)
    result = run_worker(plan_path)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((tmp_path / "metrics.json").read_text())

    assert set(metrics.keys()) == METRICS_KEYS
    assert metrics["schema_version"] == 1
    assert metrics["device"] == "cpu"
    assert metrics["seed"] == 42
    assert metrics["split"] == [0.8, 0.1, 0.1]
    assert 0.0 <= metrics["metric_value"] <= 1.0
    assert 0.0 <= metrics["baseline_metric"] <= 1.0
    # learnable fixture: model should beat the majority-class baseline
    assert metrics["metric_value"] >= metrics["baseline_metric"]
    assert isinstance(metrics["n_train"], int) and metrics["n_train"] > 0


def test_determinism(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    p1 = write_plan(dir_a)
    p2 = write_plan(dir_b)

    assert run_worker(p1).returncode == 0
    assert run_worker(p2).returncode == 0
    m1 = json.loads((dir_a / "metrics.json").read_text())
    m2 = json.loads((dir_b / "metrics.json").read_text())
    assert m1["metric_value"] == m2["metric_value"]
    assert m1["baseline_metric"] == m2["baseline_metric"]


def test_bad_plan_writes_error(tmp_path):
    plan_path = write_plan(tmp_path, schema_version=99)
    result = run_worker(plan_path)
    assert result.returncode != 0
    assert not (tmp_path / "metrics.json").exists()
    err = json.loads((tmp_path / "error.json").read_text())
    assert err["code"] == "bad_plan"
    assert err["code"] in {"dataset_missing", "bad_plan", "train_failed", "oom", "unknown"}
    assert err["stage"] in {"load", "preprocess", "train", "eval", "write"}
    assert err["device_fallback"] is False


def test_missing_label_is_bad_plan(tmp_path):
    plan_path = write_plan(tmp_path, label_column="does_not_exist")
    result = run_worker(plan_path)
    assert result.returncode != 0
    err = json.loads((tmp_path / "error.json").read_text())
    assert err["code"] == "bad_plan"
    assert err["stage"] == "preprocess"

