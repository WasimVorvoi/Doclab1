"""CLI entrypoint for the DocLab training worker.

Usage:
    python -m doclab_worker --help
    python -m doclab_worker --job <path/to/plan.json>

Reads a plan.json, dispatches to the modality engine it declares
(tabular / image / text), and writes `metrics.json` beside the plan on
success — or `error.json` + non-zero exit on failure.
"""

import argparse
import json
import sys
from pathlib import Path

from . import tabular
from .errors import WorkerError, write_error_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doclab_worker",
        description="DocLab training worker: reads a plan.json, writes metrics.json.",
    )
    parser.add_argument(
        "--job",
        metavar="PLAN_JSON",
        help="Path to a plan.json describing the training job.",
    )
    parser.add_argument(
        "--predict",
        metavar="PREDICT_JSON",
        help="Path to a predict_request.json for inference on a saved checkpoint.",
    )
    return parser


def data_root() -> Path:
    root = Path.home() / ".doclab"
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_job(plan_path: Path) -> dict:
    """Dispatch to the right training engine based on the plan's modality.

    Tabular is the default for older Phase 1 plans. Image and text plans route
    to their modality-specific engines when the plan carries `modality` or
    `data_type`.
    """
    plan = tabular.load_plan(plan_path)
    job_dir = plan_path.parent
    modality = (plan.get("modality") or plan.get("data_type") or "tabular").lower()

    if modality == "image":
        from . import image

        return image.run_job(plan, data_root(), job_dir)

    if modality == "text":
        from . import text

        return text.run_job(plan, data_root(), job_dir)

    if modality != "tabular":
        raise WorkerError(
            "bad_plan", "load", f"unsupported modality/data_type: {modality!r}"
        )

    df = tabular.load_data(plan, data_root())
    X, y = tabular.preprocess(df, plan.get("label_column", ""))
    splits = tabular.split(X, y, plan["split"], plan["seed"])
    (X_tr, y_tr), _, (X_te, y_te) = splits
    model, model_type, framework = tabular.train(X_tr, y_tr, plan["seed"])
    accuracy, baseline = tabular.evaluate(model, X_te, y_te)

    # Save checkpoint
    feature_columns = list(range(X.shape[1]))  # Column indices after preprocessing
    checkpoint_dir = tabular.save_checkpoint(
        model, model_type, framework, plan.get("label_column", ""), feature_columns, job_dir
    )

    metrics = tabular.build_metrics(
        plan, accuracy, baseline, splits, model_type, framework
    )
    metrics["checkpoint_dir"] = checkpoint_dir
    return metrics


def run_predict(request_path: Path) -> dict:
    """Load a checkpoint and run inference on the provided input."""
    from . import predict

    if not request_path.exists():
        raise WorkerError("bad_input", "load", f"request not found: {request_path}")

    try:
        request = json.loads(request_path.read_text())
    except json.JSONDecodeError as e:
        raise WorkerError("bad_input", "load", f"request is not valid JSON: {e}")

    if request.get("schema_version") != 1:
        raise WorkerError(
            "bad_input",
            "load",
            f"unsupported request schema_version: {request.get('schema_version')!r}",
        )

    experiment_dir = Path(request.get("experiment_dir", ""))
    if not experiment_dir.exists():
        raise WorkerError(
            "checkpoint_missing",
            "load",
            f"experiment_dir not found: {experiment_dir}",
        )

    checkpoints_dir = experiment_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise WorkerError(
            "checkpoint_missing",
            "load",
            f"checkpoints dir not found: {checkpoints_dir}",
        )

    manifest = predict.load_manifest(checkpoints_dir)
    input_data = request.get("input", {})

    return predict.dispatch_predict(manifest, input_data, checkpoints_dir, data_root())



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.predict:
        request_path = Path(args.predict)
        request_dir = request_path.parent
        prediction_path = request_dir / "prediction.json"
        error_path = request_dir / "error.json"

        try:
            prediction = run_predict(request_path)
        except WorkerError as err:
            write_error_json(error_path, err)
            return 1
        except Exception as e:
            write_error_json(error_path, WorkerError("unknown", "predict", str(e)))
            return 1

        try:
            prediction_path.write_text(json.dumps(prediction, indent=2))
        except OSError as e:
            write_error_json(
                error_path, WorkerError("unknown", "write", f"cannot write prediction: {e}")
            )
            return 1

        print(f"wrote {prediction_path}")
        return 0

    if args.job is None:
        parser.print_help()
        return 0

    plan_path = Path(args.job)
    job_dir = plan_path.parent
    metrics_path = job_dir / "metrics.json"
    error_path = job_dir / "error.json"

    try:
        metrics = run_job(plan_path)
    except WorkerError as err:
        write_error_json(error_path, err)
        return 1
    except Exception as e:  # never crash without honoring the error contract
        write_error_json(error_path, WorkerError("unknown", "train", str(e)))
        return 1

    try:
        metrics_path.write_text(json.dumps(metrics, indent=2))
    except OSError as e:
        write_error_json(
            error_path, WorkerError("unknown", "write", f"cannot write metrics: {e}")
        )
        return 1

    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
