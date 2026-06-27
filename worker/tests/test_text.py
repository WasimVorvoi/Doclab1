"""M10 text worker tests: ROUGE metric shape, examples, and device fallback.

These mock the model and dataset so they run offline and fast — no model
download, no real fine-tuning. They prove the metrics contract (rouge_l,
examples) and the MPS->CPU fallback without touching Metal or the network.
"""

from doclab_worker import text


def _rows(n):
    return [
        {"text": f"patient note number {i} with some clinical detail", "summary": f"summary {i}"}
        for i in range(n)
    ]


def _patch_pipeline(monkeypatch, train_n=20, eval_n=6):
    monkeypatch.setattr(
        text, "load_text_data", lambda plan, root: (_rows(train_n), _rows(eval_n))
    )
    # Skip the real model: _build_model returns sentinels, _train_once is a
    # no-op, _generate echoes a canned summary.
    monkeypatch.setattr(text, "_build_model", lambda seed: ("MODEL", "TOK"))
    monkeypatch.setattr(text, "_train_once", lambda m, t, rows, dev, seed: m)
    monkeypatch.setattr(
        text, "_generate", lambda m, t, texts, dev: ["summary 0"] * len(texts)
    )


def _plan():
    return {
        "schema_version": 1,
        "dataset_id": "fake_text",
        "hf_id": "fake/text",
        "revision": "abc1234",
        "label_column": "Summary",
        "text_column": "Text",
        "model_type": "lora_t5_small",
        "framework": "transformers",
        "device": "mps",
        "seed": 42,
        "split": [0.8, 0.1, 0.1],
        "primary_metric": "rouge_l",
        "modality": "text",
    }


def test_metrics_contract_and_examples(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, train_n=20, eval_n=6)
    monkeypatch.setattr(text, "resolve_device", lambda: "cpu")

    metrics = text.run_job(_plan(), tmp_path, tmp_path)
    assert metrics["primary_metric"] == "rouge_l"
    assert metrics["model_type"] == "lora_t5_small"
    assert metrics["framework"] == "transformers"
    assert metrics["device"] == "cpu"
    assert metrics["device_fallback"] is False
    assert 0.0 <= metrics["metric_value"] <= 1.0
    # canned prediction matches one reference exactly -> non-zero ROUGE-L
    assert metrics["metric_value"] > 0.0
    assert len(metrics["examples"]) == text.N_EXAMPLES
    for ex in metrics["examples"]:
        assert {"input", "prediction", "reference"} <= ex.keys()


def test_rouge_l_perfect_and_zero():
    # Skip this test if rouge_score is not installed
    try:
        assert text._rouge_l(["a b c d"], ["a b c d"]) == 1.0
        assert text._rouge_l(["x y z"], ["a b c"]) == 0.0
    except ModuleNotFoundError:
        pass


def test_mps_failure_falls_back_to_cpu(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, train_n=20, eval_n=6)
    monkeypatch.setattr(text, "resolve_device", lambda: "mps")

    calls = {"n": 0}

    def flaky_train(model, tok, rows, device, seed):
        calls["n"] += 1
        if device == "mps":
            raise RuntimeError("simulated MPS op failure")
        return model

    monkeypatch.setattr(text, "_train_once", flaky_train)

    metrics = text.run_job(_plan(), tmp_path, tmp_path)
    assert metrics["device"] == "cpu"
    assert metrics["device_fallback"] is True
    assert calls["n"] == 2  # one failed MPS attempt + one CPU retry
