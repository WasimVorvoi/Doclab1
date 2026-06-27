"""Demo-readiness preflight check for DocLab.

Run on the presentation laptop during setup to verify the app will demo
cleanly. Automates the manual 15-min checklist in Current/DEMO.md.

    cd worker && .venv/bin/python scripts/preflight.py

Exits 0 only if all CRITICAL checks pass; exits 1 otherwise. WARN items are
informational (e.g. CPU instead of MPS, uncached stretch datasets) and never
fail the run. No network calls.
"""

import importlib
import platform
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKETPLACE = REPO_ROOT / "marketplace" / "datasets.yaml"
SEED_DIR = REPO_ROOT / "demo" / "seed_experiment"

# Make the worker package importable no matter the CWD the presenter runs from.
_WORKER_DIR = str(REPO_ROOT / "worker")
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

GOLDEN_DATASET_ID = "diabetes_readmission"
CRITICAL_DEPS = ["xgboost", "sklearn", "pandas", "datasets"]
STRETCH_DEPS = ["torch", "torchvision", "transformers", "peft"]


def data_root() -> Path:
    return Path.home() / ".doclab"


def _dir_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _can_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def check_worker() -> tuple[str, str]:
    if not _can_import("doclab_worker"):
        return FAIL, "cannot import doclab_worker (run from worker/ with its venv)"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "doclab_worker", "--help"],
            cwd=str(REPO_ROOT / "worker"),
            capture_output=True,
            text=True,
        )
    except Exception as e:  # noqa: BLE001
        return FAIL, f"worker CLI failed to launch: {e}"
    if proc.returncode != 0:
        return FAIL, f"`doclab_worker --help` exited {proc.returncode}"
    return PASS, "worker importable and CLI runnable"


def check_deps() -> tuple[str, str]:
    crit_missing = [d for d in CRITICAL_DEPS if not _can_import(d)]
    stretch_missing = [d for d in STRETCH_DEPS if not _can_import(d)]
    if crit_missing:
        return FAIL, f"missing critical deps: {', '.join(crit_missing)}"
    if stretch_missing:
        return WARN, f"missing stretch (image/text) deps: {', '.join(stretch_missing)}"
    return PASS, "all tabular + stretch deps importable"


def check_device() -> tuple[str, str]:
    try:
        from doclab_worker.device import resolve_device

        device = resolve_device()
    except Exception as e:  # noqa: BLE001
        return WARN, f"could not resolve device: {e}"
    is_apple_silicon = platform.machine() == "arm64" and platform.system() == "Darwin"
    if device == "cpu" and is_apple_silicon:
        return WARN, "device=cpu on Apple Silicon (MPS unavailable; CPU fallback is fine)"
    return PASS, f"device={device}"


def check_datasets() -> list[tuple[str, str]]:
    try:
        market = yaml.safe_load(MARKETPLACE.read_text())
    except Exception as e:  # noqa: BLE001
        return [(FAIL, f"cannot read marketplace/datasets.yaml: {e}")]
    results: list[tuple[str, str]] = []
    cache = data_root() / "datasets"
    for d in market.get("datasets", []):
        ds_id = d.get("id", "?")
        data_type = d.get("data_type")
        cached = _dir_has_files(cache / ds_id)
        if cached:
            results.append((PASS, f"dataset {ds_id} cached"))
        elif ds_id == GOLDEN_DATASET_ID:
            results.append((FAIL, f"golden dataset {ds_id} NOT cached — run prefetch.py"))
        else:
            results.append((WARN, f"dataset {ds_id} ({data_type}) not cached — optional demo only"))
    return results


def check_db() -> tuple[str, str]:
    if (data_root() / "doclab.db").exists():
        return PASS, "~/.doclab/doclab.db present"
    return WARN, "~/.doclab/doclab.db missing — launch the app once to init DB + seed row"


def check_seed_bundle() -> tuple[str, str]:
    import json

    plan = SEED_DIR / "plan.json"
    metrics = SEED_DIR / "metrics.json"
    card = SEED_DIR / "model_card.md"
    for f in (plan, metrics, card):
        if not f.exists():
            return FAIL, f"seed bundle missing {f.name} (Fallback B depends on it)"
    try:
        plan_blob = json.loads(plan.read_text())
    except Exception as e:  # noqa: BLE001
        return FAIL, f"seed bundle plan.json is not valid JSON: {e}"
    try:
        metrics_blob = json.loads(metrics.read_text())
    except Exception as e:  # noqa: BLE001
        return FAIL, f"seed bundle metrics.json is not valid JSON: {e}"

    if plan_blob.get("schema_version") != 1:
        return FAIL, "seed bundle plan.json has unsupported schema_version"
    if plan_blob.get("dataset_id") != GOLDEN_DATASET_ID:
        return FAIL, f"seed bundle plan.json must use {GOLDEN_DATASET_ID}"
    if plan_blob.get("modality") != "tabular":
        return FAIL, "seed bundle plan.json must declare tabular modality"
    if plan_blob.get("primary_metric") != "accuracy":
        return FAIL, "seed bundle plan.json must use accuracy as primary_metric"
    if plan_blob.get("model_type") not in {"xgboost", "logistic_regression"}:
        return FAIL, "seed bundle plan.json must use a supported tabular model_type"
    if metrics_blob.get("schema_version") != 1:
        return FAIL, "seed bundle metrics.json has unsupported schema_version"
    for key in ("metric_value", "baseline_metric"):
        value = metrics_blob.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            return FAIL, f"seed bundle metrics.json has invalid {key}"
    if metrics_blob["metric_value"] <= metrics_blob["baseline_metric"]:
        return FAIL, "seed bundle metric_value must beat baseline_metric"

    card_text = card.read_text()
    if not card_text.strip():
        return FAIL, "seed bundle model_card.md is empty"
    if "not for clinical care" not in card_text.lower():
        return FAIL, "seed bundle model_card.md missing non-clinical-care disclaimer"
    return PASS, "seed experiment bundle intact"


def main(argv: list[str] | None = None) -> int:
    results: list[tuple[str, str]] = [
        check_worker(),
        check_deps(),
        check_device(),
        check_db(),
        check_seed_bundle(),
    ]
    results.extend(check_datasets())

    print("DocLab Demo Preflight Check")
    print("=" * 40)
    for status, msg in results:
        print(f"[{status}] {msg}")

    n_pass = sum(1 for s, _ in results if s == PASS)
    n_warn = sum(1 for s, _ in results if s == WARN)
    n_fail = sum(1 for s, _ in results if s == FAIL)
    print("=" * 40)
    print(f"Summary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")

    if n_fail:
        print("Status: NOT READY — resolve the FAIL items above before demoing.")
    else:
        print("Status: READY (all critical checks passed)")

    if any("not cached" in m for _, m in results):
        print("  tip: cache stretch datasets with `python scripts/prefetch.py --all`")
    if any("launch the app once" in m for _, m in results):
        print("  tip: run the app once so it creates ~/.doclab/doclab.db + the seed run")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
