# DocLab Worker

Python training worker. Invoked by the Rust shell as:

```bash
python -m doclab_worker --job <path/to/plan.json>
```

JSON in (`plan.json`) → JSON out (`metrics.json`), or `error.json` + non-zero exit on failure.
All three modality paths are implemented: tabular (XGBoost, M2), image (PyTorch CNN, M9), and
text (Transformers + LoRA, M10). The worker dispatches on `plan.modality`.

## Setup

```bash
cd worker
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**macOS prerequisite for XGBoost:** `brew install libomp` (XGBoost needs the OpenMP runtime, or
its import fails with `libxgboost.dylib could not be loaded`). Without it the worker still runs
via the LogisticRegression fallback, but the intended path is XGBoost.

`--help` runs on the standard library alone (no venv needed for the M0 healthcheck).

**Note on LLM planning:** Goal parsing and dataset selection happen in the Rust backend (Tauri), not in this Python worker. The worker only consumes `plan.json` and trains models locally. LLM calls (if configured) are handled by Rust to keep API keys secure.

## Running a job

```bash
python -m doclab_worker --job <path/to/plan.json>
```

On success, writes `metrics.json` beside the plan and exits 0. On failure, writes `error.json`
(closed-set `code` + `stage`), echoes it to stderr, and exits non-zero.

## Tests

```bash
.venv/bin/python -m pytest tests
```

## Demo prep

```bash
.venv/bin/python scripts/prefetch.py       # cache the golden tabular datasets
.venv/bin/python scripts/prefetch.py --all # optional: cache image/text too
.venv/bin/python scripts/preflight.py      # verify demo-readiness without network
```

## Runtime data

DocLab stores datasets and experiments under the data root **`~/.doclab/`**
(`doclab.db`, `datasets/<id>/`, `experiments/<id>/`). The Rust shell creates
this on first run (M3); the worker reads/writes paths handed to it via the plan.
