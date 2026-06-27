"""Worker error contract.

On any failure the worker writes an `error.json` beside the plan, echoes the
same blob to stderr, and exits non-zero. `code` is a closed set so the Rust
side (M3) can branch on it deterministically.
"""

import json
import sys
from pathlib import Path

# Closed set of machine-readable error codes (see MILESTONES M2).
ERROR_CODES = {"dataset_missing", "bad_plan", "train_failed", "oom", "unknown"}

# Pipeline stages, for locating where a failure happened.
STAGES = {"load", "preprocess", "train", "eval", "write"}


class WorkerError(Exception):
    """A failure mapped to the error contract."""

    def __init__(self, code: str, stage: str, message: str):
        if code not in ERROR_CODES:
            code = "unknown"
        if stage not in STAGES:
            stage = "load"
        self.code = code
        self.stage = stage
        self.message = message
        super().__init__(message)

    def to_dict(self, device_fallback: bool = False) -> dict:
        return {
            "schema_version": 1,
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "device_fallback": device_fallback,
        }


def write_error_json(
    error_path: Path, err: WorkerError, device_fallback: bool = False
) -> None:
    """Write the error blob beside the plan and echo it to stderr."""
    blob = err.to_dict(device_fallback=device_fallback)
    text = json.dumps(blob, indent=2)
    try:
        error_path.write_text(text)
    except OSError:
        pass  # stderr echo below is the fallback signal
    print(text, file=sys.stderr)
