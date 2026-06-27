"""Pre-cache curated datasets into ~/.doclab/datasets/ for offline demos.

Run once during demo setup so the live golden path needs no network:

    cd worker && .venv/bin/python scripts/prefetch.py

By default it prefetches every tabular dataset in marketplace/datasets.yaml.
Pass dataset ids to limit it, or --all to cache every modality (image/text too):

    .venv/bin/python scripts/prefetch.py diabetes_readmission
    .venv/bin/python scripts/prefetch.py --all

Mirrors the worker's cache path (data_root/datasets/<id>) so a later training
run finds the dataset already on disk.
"""

import sys
from pathlib import Path

import yaml
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKETPLACE = REPO_ROOT / "marketplace" / "datasets.yaml"


def data_root() -> Path:
    root = Path.home() / ".doclab"
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    return root


def main(argv: list[str]) -> int:
    market = yaml.safe_load(MARKETPLACE.read_text())
    datasets = market.get("datasets", [])

    cache_all = "--all" in argv
    wanted = {a for a in argv if not a.startswith("--")}

    if wanted:
        selected = [d for d in datasets if d.get("id") in wanted]
    elif cache_all:
        selected = list(datasets)
    else:
        selected = [d for d in datasets if d.get("data_type") == "tabular"]

    if not selected:
        print("nothing to prefetch (no matching datasets)")
        return 0

    root = data_root()
    failures = 0
    for d in selected:
        ds_id = d["id"]
        cache_dir = root / "datasets" / ds_id
        print(f"prefetching {ds_id} ({d['hf_id']}@{d['revision'][:7]}) ...")
        try:
            load_dataset(d["hf_id"], revision=d["revision"], cache_dir=str(cache_dir))
            print(f"  cached -> {cache_dir}")
        except Exception as e:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  FAILED: {e}")

    if failures:
        print(f"done with {failures} failure(s)")
        return 1
    print("all datasets cached")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
