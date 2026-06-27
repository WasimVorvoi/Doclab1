"""Tabular training engine (M2).

Small importable steps: load_plan -> load_data -> preprocess -> split ->
train -> evaluate -> build_metrics. The CLI in __main__.py chains these and
maps any WorkerError to the error.json contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import WorkerError

SCHEMA_VERSION = 1


def load_plan(path: Path) -> dict:
    if not path.exists():
        raise WorkerError("bad_plan", "load", f"plan not found: {path}")
    try:
        plan = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise WorkerError("bad_plan", "load", f"plan is not valid JSON: {e}")
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise WorkerError(
            "bad_plan",
            "load",
            f"unsupported plan schema_version: {plan.get('schema_version')!r}",
        )
    return plan


def load_data(plan: dict, data_root: Path) -> pd.DataFrame:
    """Load the dataset into one DataFrame.

    Supports `local_csv` (offline, for tests) and Hugging Face datasets pinned
    to a revision (cached under the data root).
    """
    if plan.get("local_csv"):
        csv = Path(plan["local_csv"])
        if not csv.exists():
            raise WorkerError("dataset_missing", "load", f"csv not found: {csv}")
        return pd.read_csv(csv)

    hf_id = plan.get("hf_id")
    revision = plan.get("revision")
    if not hf_id or not revision:
        raise WorkerError("bad_plan", "load", "plan missing hf_id/revision")
    try:
        from datasets import load_dataset

        cache_dir = data_root / "datasets" / plan.get("dataset_id", "unknown")
        ds = load_dataset(hf_id, revision=revision, cache_dir=str(cache_dir))
        frames = [split.to_pandas() for split in ds.values()]
        return pd.concat(frames, ignore_index=True)
    except WorkerError:
        raise
    except Exception as e:
        raise WorkerError("dataset_missing", "load", f"failed to load {hf_id}: {e}")


def preprocess(df: pd.DataFrame, label_column: str):
    """Split off the label, impute numerics, one-hot encode object columns."""
    if not label_column or label_column not in df.columns:
        raise WorkerError(
            "bad_plan", "preprocess", f"label_column {label_column!r} not in data"
        )
    y_raw = df[label_column]
    X = df.drop(columns=[label_column])

    obj_cols = X.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        X = pd.get_dummies(X, columns=list(obj_cols), dummy_na=False)

    num_cols = X.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # Coerce label to integer class ids (handles bool/str/float labels).
    y = pd.factorize(y_raw, sort=True)[0].astype(int)
    return X.to_numpy(dtype=float), np.asarray(y)


def split(X, y, ratios, seed):
    """80/10/10 (configurable) via two stratified splits, fixed seed."""
    from sklearn.model_selection import train_test_split

    test_frac = ratios[2]
    val_frac = ratios[1]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=(val_frac + test_frac), random_state=seed, stratify=y
    )
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=seed, stratify=y_tmp
    )
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def train(X_tr, y_tr, seed):
    """XGBoost classifier; fall back to LogisticRegression on any error.

    A fallback is NOT a job failure — only a total inability to fit raises.
    Returns (model, model_type, framework).
    """
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=0,
            eval_metric="logloss",
            tree_method="hist",
        )
        model.fit(X_tr, y_tr)
        return model, "xgboost", "xgboost"
    except Exception:
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(X_tr, y_tr)
            return model, "logistic_regression", "sklearn"
        except Exception as e:
            raise WorkerError("train_failed", "train", f"could not fit model: {e}")


def evaluate(model, X_te, y_te):
    """Test accuracy + majority-class baseline accuracy."""
    from sklearn.metrics import accuracy_score

    try:
        preds = model.predict(X_te)
        accuracy = float(accuracy_score(y_te, preds))
        _, counts = np.unique(y_te, return_counts=True)
        baseline = float(counts.max() / counts.sum())
        return accuracy, baseline
    except Exception as e:
        raise WorkerError("train_failed", "eval", f"evaluation failed: {e}")


def build_metrics(plan, accuracy, baseline, splits, model_type, framework) -> dict:
    (X_tr, _), (X_val, _), (X_te, _) = splits
    return {
        "schema_version": SCHEMA_VERSION,
        "primary_metric": plan.get("primary_metric", "accuracy"),
        "metric_value": round(accuracy, 6),
        "baseline_metric": round(baseline, 6),
        "device": "cpu",
        "seed": plan["seed"],
        "split": plan["split"],
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_te)),
        "model_type": model_type,
        "framework": framework,
    }


def save_checkpoint(model, model_type, framework, label_column, feature_columns, job_dir: Path):
    """Save model checkpoint and manifest for later inference."""
    import joblib

    checkpoints_dir = job_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, checkpoints_dir / "model.joblib")

    # Save preprocessing metadata
    preprocess_meta = {
        "label_column": label_column,
        "feature_columns": feature_columns,
    }
    (checkpoints_dir / "preprocess.json").write_text(json.dumps(preprocess_meta, indent=2))

    # Save manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "modality": "tabular",
        "model_type": model_type,
        "framework": framework,
        "label_column": label_column,
    }
    (checkpoints_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return "checkpoints"


def predict(checkpoints_dir: Path, input_json: str) -> dict:
    """Load checkpoint and run inference on tabular input."""
    import joblib
    import pandas as pd

    # Load manifest
    manifest_path = checkpoints_dir / "manifest.json"
    if not manifest_path.exists():
        raise WorkerError("checkpoint_missing", "load", "manifest.json not found")

    manifest = json.loads(manifest_path.read_text())
    label_column = manifest.get("label_column", "")

    # Load model
    model_path = checkpoints_dir / "model.joblib"
    if not model_path.exists():
        raise WorkerError("checkpoint_missing", "load", "model.joblib not found")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise WorkerError("checkpoint_missing", "load", f"cannot load model: {e}")

    # Load preprocessing metadata
    preprocess_path = checkpoints_dir / "preprocess.json"
    if not preprocess_path.exists():
        raise WorkerError("checkpoint_missing", "load", "preprocess.json not found")

    preprocess_meta = json.loads(preprocess_path.read_text())
    feature_columns = preprocess_meta.get("feature_columns", [])

    # Parse input
    try:
        input_data = json.loads(input_json) if isinstance(input_json, str) else input_json
    except json.JSONDecodeError as e:
        raise WorkerError("bad_input", "predict", f"input is not valid JSON: {e}")

    if not isinstance(input_data, dict):
        raise WorkerError("bad_input", "predict", "input must be a JSON object")

    # Convert to DataFrame and align with training features
    try:
        df = pd.DataFrame([input_data])

        # Apply same preprocessing as training (one-hot encoding)
        obj_cols = df.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            df = pd.get_dummies(df, columns=list(obj_cols), dummy_na=False)

        # Fill missing numeric values with 0 (median not available at predict time)
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].fillna(0)

        # Align columns with training (add missing, remove extra)
        for col_idx in feature_columns:
            col_name = f"feature_{col_idx}"
            if col_name not in df.columns:
                df[col_name] = 0

        # Convert to numpy array
        X = df.to_numpy(dtype=float)

        # Ensure correct number of features
        if X.shape[1] != len(feature_columns):
            # Pad or truncate to match training shape
            if X.shape[1] < len(feature_columns):
                padding = np.zeros((X.shape[0], len(feature_columns) - X.shape[1]))
                X = np.hstack([X, padding])
            else:
                X = X[:, :len(feature_columns)]

    except Exception as e:
        raise WorkerError("bad_input", "predict", f"cannot process input: {e}")

    # Run inference
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
        else:
            pred_idx = int(model.predict(X)[0])
            confidence = None

        return {
            "schema_version": SCHEMA_VERSION,
            "modality": "tabular",
            "prediction": f"Class {pred_idx}",
            "confidence": round(confidence, 4) if confidence is not None else None,
            "detail": "Prototype prediction only.",
            "warning": "Not for clinical use.",
        }
    except Exception as e:
        raise WorkerError("predict_failed", "predict", f"inference failed: {e}")
