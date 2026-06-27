"""Prediction dispatch for saved checkpoints (M12).

Loads a checkpoint manifest, routes to the modality-specific predict function,
and returns a prediction.json matching the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from .errors import WorkerError

SCHEMA_VERSION = 1


def load_manifest(checkpoints_dir: Path) -> dict:
    """Load and validate the checkpoint manifest."""
    manifest_path = checkpoints_dir / "manifest.json"
    if not manifest_path.exists():
        raise WorkerError(
            "checkpoint_missing",
            "load",
            f"manifest.json not found in {checkpoints_dir}",
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise WorkerError("checkpoint_missing", "load", f"manifest is invalid JSON: {e}")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise WorkerError(
            "checkpoint_missing",
            "load",
            f"unsupported manifest schema_version: {manifest.get('schema_version')!r}",
        )
    return manifest


def dispatch_predict(
    manifest: dict, input_data: dict, checkpoints_dir: Path, data_root: Path
) -> dict:
    """Route to modality-specific predict based on manifest."""
    modality = manifest.get("modality", "").lower()
    input_type = input_data.get("type", "")

    if modality == "tabular":
        if input_type != "tabular_json":
            raise WorkerError(
                "bad_input",
                "predict",
                f"tabular requires input.type='tabular_json', got '{input_type}'"
            )
        from . import tabular
        return tabular.predict(checkpoints_dir, input_data.get("value", ""))

    if modality == "image":
        from . import image
        return image.predict(checkpoints_dir, input_data.get("value", ""))

    if modality == "text":
        from . import text
        return text.predict(checkpoints_dir, input_data.get("value", ""))

    raise WorkerError(
        "bad_input",
        "predict",
        f"unsupported modality in manifest: {modality!r}",
    )
