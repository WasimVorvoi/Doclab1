"""M9 image worker tests: device fallback and small-data warning.

These mock the dataset and training so they run offline and fast — no network,
no real Metal kernels. They prove the two paths a live MPS run won't exercise:
the MPS->CPU fallback and the <500-sample warning.
"""

import torch

from doclab_worker import image


class _FakeImg:
    mode = "L"

    def convert(self, _mode):
        return self


def _fake_ds(n, label_cycle=(0, 1)):
    return [
        {"image": _FakeImg(), "label": label_cycle[i % len(label_cycle)]}
        for i in range(n)
    ]


def _patch_tensors(monkeypatch, n_train, n_test):
    """Replace _build_tensors with deterministic random tensors of a given size."""

    def fake_build(ds, n_max, img_size, seed):
        n = min(len(ds), n_max)
        g = torch.Generator().manual_seed(seed)
        X = torch.rand(n, 1, image.IMG_SIZE, image.IMG_SIZE, generator=g)
        y = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
        return X, y

    monkeypatch.setattr(image, "_build_tensors", fake_build)
    monkeypatch.setattr(
        image, "load_image_data", lambda plan, root: (_fake_ds(n_train), _fake_ds(n_test))
    )


def _plan():
    return {
        "schema_version": 1,
        "dataset_id": "fake_images",
        "hf_id": "fake/imgs",
        "revision": "abc1234",
        "label_column": "label",
        "model_type": "cnn",
        "framework": "pytorch",
        "device": "mps",
        "seed": 42,
        "split": [0.8, 0.1, 0.1],
        "primary_metric": "accuracy",
        "modality": "image",
    }


def test_metrics_contract_and_cpu(monkeypatch, tmp_path):
    """A normal CPU run produces a well-formed metrics dict."""
    _patch_tensors(monkeypatch, n_train=600, n_test=120)
    monkeypatch.setattr(image, "resolve_device", lambda: "cpu")

    metrics = image.run_job(_plan(), tmp_path, tmp_path)
    assert metrics["model_type"] == "cnn"
    assert metrics["framework"] == "pytorch"
    assert metrics["device"] == "cpu"
    assert metrics["device_fallback"] is False
    assert 0.0 <= metrics["metric_value"] <= 1.0
    assert 0.0 <= metrics["baseline_metric"] <= 1.0
    assert metrics["n_train"] == 600
    # 600 >= threshold, so no small-data warning
    assert "warning" not in metrics


def test_small_data_warning(monkeypatch, tmp_path):
    """A train set under the threshold attaches the overfitting warning."""
    _patch_tensors(monkeypatch, n_train=120, n_test=40)
    monkeypatch.setattr(image, "resolve_device", lambda: "cpu")

    metrics = image.run_job(_plan(), tmp_path, tmp_path)
    assert metrics["n_train"] == 120
    assert "warning" in metrics
    assert "overfitting" in metrics["warning"].lower()


def test_mps_failure_falls_back_to_cpu(monkeypatch, tmp_path):
    """When an MPS op raises, the run retries on CPU and records the fallback."""
    _patch_tensors(monkeypatch, n_train=600, n_test=120)
    monkeypatch.setattr(image, "resolve_device", lambda: "mps")

    real_train = image._train_once
    calls = {"n": 0}

    def flaky_train(model, X_tr, y_tr, device, seed):
        calls["n"] += 1
        if device == "mps":
            raise RuntimeError("simulated MPS op failure")
        return real_train(model, X_tr, y_tr, "cpu", seed)

    monkeypatch.setattr(image, "_train_once", flaky_train)

    metrics = image.run_job(_plan(), tmp_path, tmp_path)
    assert metrics["device"] == "cpu"
    assert metrics["device_fallback"] is True
    assert calls["n"] == 2  # one failed MPS attempt + one CPU retry
