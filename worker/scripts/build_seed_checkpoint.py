"""Generate minimal checkpoint for demo seed experiment.

Run once to populate demo/seed_experiment/checkpoints/ with a tiny
trained model that can be used for Try without running a full train.
"""

import json
import sys
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
SEED_DIR = REPO_ROOT / "demo" / "seed_experiment"
CHECKPOINTS_DIR = SEED_DIR / "checkpoints"


def main():
    CHECKPOINTS_DIR.mkdir(exist_ok=True)

    # Create tiny synthetic training data
    data = {
        "age": [45, 60, 55, 70, 50],
        "glucose": [100, 150, 120, 180, 110],
        "readmitted": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    X = df[["age", "glucose"]]
    y = df["readmitted"]

    # Train minimal model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Save checkpoint files
    joblib.dump(model, CHECKPOINTS_DIR / "model.joblib")

    manifest = {
        "schema_version": 1,
        "modality": "tabular",
        "model_type": "logistic_regression",
        "label_column": "readmitted",
        "created_at": "2026-05-31T00:00:00Z",
    }
    (CHECKPOINTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    preprocess = {"feature_columns": ["age", "glucose"], "label_column": "readmitted"}
    (CHECKPOINTS_DIR / "preprocess.json").write_text(json.dumps(preprocess, indent=2))

    print(f"✓ Seed checkpoint created at {CHECKPOINTS_DIR}")
    print("  Files: manifest.json, preprocess.json, model.joblib")


if __name__ == "__main__":
    main()
