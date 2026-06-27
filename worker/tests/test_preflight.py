"""Preflight + prefetch selection tests.

Offline and monkeypatch-based: they exercise the CRITICAL/WARN classification
and exit-code semantics without touching a real ~/.doclab or the network.
"""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


preflight = _load("preflight")
prefetch = _load("prefetch")


def _seed_bundle(tmp_path: Path) -> Path:
    d = tmp_path / "seed"
    d.mkdir()
    (d / "plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "diabetes_readmission",
                "modality": "tabular",
                "primary_metric": "accuracy",
                "model_type": "xgboost",
            }
        )
    )
    (d / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metric_value": 0.64,
                "baseline_metric": 0.54,
            }
        )
    )
    (d / "model_card.md").write_text("# card\nnot for clinical care")
    return d


def _all_green(monkeypatch, tmp_path, *, device="mps", db=True):
    """Patch every check into a passing state; callers override as needed."""
    monkeypatch.setattr(preflight, "check_worker", lambda: (preflight.PASS, "ok"))
    monkeypatch.setattr(preflight, "check_deps", lambda: (preflight.PASS, "ok"))
    monkeypatch.setattr(preflight, "check_device", lambda: (preflight.PASS, f"device={device}"))
    monkeypatch.setattr(preflight, "check_db", lambda: (preflight.PASS, "db") if db else (preflight.WARN, "no db"))
    monkeypatch.setattr(preflight, "check_seed_bundle", lambda: (preflight.PASS, "seed ok"))
    monkeypatch.setattr(preflight, "check_datasets", lambda: [(preflight.PASS, "diabetes_readmission cached")])


def test_all_checks_pass(monkeypatch, tmp_path, capsys):
    _all_green(monkeypatch, tmp_path)
    code = preflight.main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: READY" in out
    assert "0 FAIL" in out


def test_missing_critical_dataset_fails(monkeypatch, tmp_path, capsys):
    _all_green(monkeypatch, tmp_path)
    monkeypatch.setattr(
        preflight,
        "check_datasets",
        lambda: [(preflight.FAIL, "golden dataset diabetes_readmission NOT cached")],
    )
    code = preflight.main([])
    out = capsys.readouterr().out
    assert code == 1
    assert "NOT READY" in out
    assert "[FAIL]" in out


def test_only_golden_dataset_is_critical(monkeypatch, tmp_path):
    market = {
        "datasets": [
            {"id": "diabetes_readmission", "data_type": "tabular"},
            {"id": "heart_disease_uci", "data_type": "tabular"},
            {"id": "chest_xray_pneumonia", "data_type": "image"},
        ]
    }
    cached_root = tmp_path / "doclab" / "datasets"
    (cached_root / "diabetes_readmission").mkdir(parents=True)
    (cached_root / "diabetes_readmission" / "marker").write_text("cached")
    marketplace = tmp_path / "datasets.yaml"
    marketplace.write_text("datasets: []")
    monkeypatch.setattr(preflight.yaml, "safe_load", lambda _: market)
    monkeypatch.setattr(preflight, "MARKETPLACE", marketplace)
    monkeypatch.setattr(preflight, "data_root", lambda: tmp_path / "doclab")

    results = preflight.check_datasets()

    assert (preflight.PASS, "dataset diabetes_readmission cached") in results
    assert any(
        status == preflight.WARN and "heart_disease_uci" in msg
        for status, msg in results
    )
    assert all(status != preflight.FAIL for status, _ in results)


def test_warnings_do_not_fail(monkeypatch, tmp_path, capsys):
    _all_green(monkeypatch, tmp_path, db=False)
    monkeypatch.setattr(
        preflight, "check_device", lambda: (preflight.WARN, "device=cpu on Apple Silicon")
    )
    monkeypatch.setattr(
        preflight,
        "check_datasets",
        lambda: [
            (preflight.PASS, "diabetes_readmission cached"),
            (preflight.WARN, "chest_xray_pneumonia (image) not cached — stretch demo only"),
        ],
    )
    code = preflight.main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "Status: READY" in out
    assert "WARN" in out


def test_broken_seed_bundle_fails(monkeypatch, tmp_path):
    bundle = _seed_bundle(tmp_path)
    (bundle / "plan.json").write_text("{not json")
    monkeypatch.setattr(preflight, "SEED_DIR", bundle)
    status, msg = preflight.check_seed_bundle()
    assert status == preflight.FAIL
    assert "plan.json" in msg


def test_worker_import_failure_fails(monkeypatch):
    monkeypatch.setattr(preflight, "_can_import", lambda name: name != "doclab_worker")
    status, _ = preflight.check_worker()
    assert status == preflight.FAIL


def test_seed_bundle_intact_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "SEED_DIR", _seed_bundle(tmp_path))
    status, _ = preflight.check_seed_bundle()
    assert status == preflight.PASS


def test_real_seed_bundle_is_demo_ready():
    status, msg = preflight.check_seed_bundle()
    assert status == preflight.PASS, msg


def test_seed_bundle_must_beat_baseline(monkeypatch, tmp_path):
    bundle = _seed_bundle(tmp_path)
    (bundle / "metrics.json").write_text(
        json.dumps({"schema_version": 1, "metric_value": 0.50, "baseline_metric": 0.54})
    )
    monkeypatch.setattr(preflight, "SEED_DIR", bundle)
    status, msg = preflight.check_seed_bundle()
    assert status == preflight.FAIL
    assert "baseline_metric" in msg


def test_seed_bundle_requires_clinical_care_disclaimer(monkeypatch, tmp_path):
    bundle = _seed_bundle(tmp_path)
    (bundle / "model_card.md").write_text("# card\nnot for clinical decisions")
    monkeypatch.setattr(preflight, "SEED_DIR", bundle)
    status, msg = preflight.check_seed_bundle()
    assert status == preflight.FAIL
    assert "clinical-care" in msg


def test_prefetch_all_selects_every_modality(monkeypatch, capsys):
    datasets = [
        {"id": "tab1", "data_type": "tabular", "hf_id": "x/tab1", "revision": "abcdef0"},
        {"id": "img1", "data_type": "image", "hf_id": "x/img1", "revision": "abcdef0"},
        {"id": "txt1", "data_type": "text", "hf_id": "x/txt1", "revision": "abcdef0"},
    ]
    monkeypatch.setattr(prefetch.yaml, "safe_load", lambda _: {"datasets": datasets})
    monkeypatch.setattr(prefetch.Path, "read_text", lambda self: "")
    fetched = []
    monkeypatch.setattr(prefetch, "load_dataset", lambda hf_id, **kw: fetched.append(hf_id))
    monkeypatch.setattr(prefetch, "data_root", lambda: Path("/tmp/doclab_test_root"))

    assert prefetch.main(["--all"]) == 0
    assert set(fetched) == {"x/tab1", "x/img1", "x/txt1"}


def test_prefetch_default_is_tabular_only(monkeypatch):
    datasets = [
        {"id": "tab1", "data_type": "tabular", "hf_id": "x/tab1", "revision": "abcdef0"},
        {"id": "img1", "data_type": "image", "hf_id": "x/img1", "revision": "abcdef0"},
    ]
    monkeypatch.setattr(prefetch.yaml, "safe_load", lambda _: {"datasets": datasets})
    monkeypatch.setattr(prefetch.Path, "read_text", lambda self: "")
    fetched = []
    monkeypatch.setattr(prefetch, "load_dataset", lambda hf_id, **kw: fetched.append(hf_id))
    monkeypatch.setattr(prefetch, "data_root", lambda: Path("/tmp/doclab_test_root"))

    assert prefetch.main([]) == 0
    assert fetched == ["x/tab1"]
