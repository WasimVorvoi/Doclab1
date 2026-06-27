"""Worker entrypoint dispatch tests.

These stay in-process and monkeypatch the heavy modality workers. They protect
the Rust<->Python plan contract: a serialized plan must route to the engine its
modality declares, and unsupported modalities must honor the error contract.
"""

import json
from pathlib import Path

from doclab_worker import __main__ as worker_main
from doclab_worker import image, text


def _write_plan(tmp_path: Path, **overrides) -> Path:
    plan = {
        "schema_version": 1,
        "dataset_id": "fake_dataset",
        "hf_id": "fake/dataset",
        "revision": "abc1234",
        "label_column": "label",
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


def test_image_plan_dispatches_to_image_worker(monkeypatch, tmp_path):
    calls = {}

    def fake_run_job(plan, data_root, job_dir):
        calls["plan"] = plan
        calls["data_root"] = data_root
        calls["job_dir"] = job_dir
        return {"primary_metric": "accuracy", "model_type": "cnn"}

    monkeypatch.setattr(image, "run_job", fake_run_job)
    plan_path = _write_plan(
        tmp_path,
        modality="image",
        model_type="cnn",
        framework="pytorch",
        device="mps",
    )

    metrics = worker_main.run_job(plan_path)

    assert metrics["model_type"] == "cnn"
    assert calls["plan"]["modality"] == "image"
    assert calls["data_root"].name == ".doclab"


def test_text_plan_dispatches_to_text_worker(monkeypatch, tmp_path):
    calls = {}

    def fake_run_job(plan, data_root, job_dir):
        calls["plan"] = plan
        calls["data_root"] = data_root
        calls["job_dir"] = job_dir
        return {"primary_metric": "rouge_l", "model_type": "lora_t5_small"}

    monkeypatch.setattr(text, "run_job", fake_run_job)
    plan_path = _write_plan(
        tmp_path,
        modality="text",
        text_column="Text",
        label_column="Summary",
        model_type="lora_t5_small",
        framework="transformers",
        device="mps",
        primary_metric="rouge_l",
    )

    metrics = worker_main.run_job(plan_path)

    assert metrics["primary_metric"] == "rouge_l"
    assert calls["plan"]["text_column"] == "Text"
    assert calls["data_root"].name == ".doclab"


def test_unsupported_modality_writes_error_contract(tmp_path):
    plan_path = _write_plan(tmp_path, modality="audio")

    code = worker_main.main(["--job", str(plan_path)])

    assert code == 1
    assert not (tmp_path / "metrics.json").exists()
    error = json.loads((tmp_path / "error.json").read_text())
    assert error["code"] == "bad_plan"
    assert error["stage"] == "load"
    assert "unsupported modality" in error["message"]
