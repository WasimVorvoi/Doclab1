"""Image classification path (M9): small CNN on a 2-class medical image set.

Mirrors the tabular contract — reads a plan, returns a metrics dict — but
trains a tiny ConvNet with PyTorch. Subsamples and caps epochs so a demo run
finishes in a few minutes. Uses MPS on Apple Silicon with a one-shot CPU
fallback if a Metal op fails mid-train.
"""

from __future__ import annotations

from pathlib import Path

from .device import resolve_device
from .errors import WorkerError

SCHEMA_VERSION = 1

# Hackathon time caps — a demo run must finish in a few minutes.
MAX_TRAIN = 800
MAX_EVAL = 200
IMG_SIZE = 64
EPOCHS = 3
BATCH_SIZE = 32
SMALL_DATA_THRESHOLD = 500


def load_image_data(plan: dict, data_root: Path):
    """Load a 2-class HF image dataset, pinned to revision and cached on disk.

    Returns (train_ds, test_ds) as Hugging Face datasets with 'image' and
    'label' columns. Subsampling happens later so we know the true train size.
    """
    hf_id = plan.get("hf_id")
    revision = plan.get("revision")
    if not hf_id or not revision:
        raise WorkerError("bad_plan", "load", "plan missing hf_id/revision")
    try:
        from datasets import load_dataset

        cache_dir = data_root / "datasets" / plan.get("dataset_id", "unknown")
        ds = load_dataset(hf_id, revision=revision, cache_dir=str(cache_dir))
    except Exception as e:
        raise WorkerError("dataset_missing", "load", f"failed to load {hf_id}: {e}")

    split_names = list(ds.keys())
    train_split = "train" if "train" in ds else split_names[0]
    test_split = (
        "test"
        if "test" in ds
        else ("validation" if "validation" in ds else train_split)
    )
    return ds[train_split], ds[test_split]


def _build_tensors(ds, n_max, img_size, seed):
    """Subsample, resize to grayscale img_size x img_size, return (X, y) tensors."""
    import torch
    from torchvision import transforms

    n = min(len(ds), n_max)
    rng = list(range(len(ds)))
    import random

    random.Random(seed).shuffle(rng)
    idx = rng[:n]

    tfm = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    xs, ys = [], []
    for i in idx:
        row = ds[i]
        img = row["image"]
        if img.mode != "L":
            img = img.convert("L")
        xs.append(tfm(img))
        ys.append(int(row["label"]))
    X = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


class TinyConvNet:
    """Factory for a small 2-class CNN — kept tiny for hackathon time limits."""

    @staticmethod
    def build(img_size: int, n_classes: int):
        import torch.nn as nn

        flat = 16 * (img_size // 4) * (img_size // 4)
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(flat, n_classes),
        )


def _train_once(model, X_tr, y_tr, device, seed):
    """Train the CNN on a given device. Raises on any failure (caller handles
    the MPS->CPU fallback)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = model.to(device)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    n = X_tr.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_SIZE):
            batch = perm[start : start + BATCH_SIZE]
            opt.zero_grad()
            out = model(X_tr[batch])
            loss = loss_fn(out, y_tr[batch])
            loss.backward()
            opt.step()
    return model


def run_job(plan: dict, data_root: Path, job_dir: Path) -> dict:
    """Image classification pipeline. Returns a metrics dict matching the
    tabular contract, plus device_fallback and an optional warning."""
    import json
    import torch

    seed = plan.get("seed", 42)
    train_ds, test_ds = load_image_data(plan, data_root)

    X_tr, y_tr = _build_tensors(train_ds, MAX_TRAIN, IMG_SIZE, seed)
    X_te, y_te = _build_tensors(test_ds, MAX_EVAL, IMG_SIZE, seed)
    n_classes = int(max(int(y_tr.max()), int(y_te.max())) + 1)

    requested = resolve_device()
    device_used = requested
    device_fallback = False
    model = TinyConvNet.build(IMG_SIZE, n_classes)
    try:
        model = _train_once(model, X_tr, y_tr, requested, seed)
    except Exception as e:
        if requested == "cpu":
            raise WorkerError("train_failed", "train", f"could not train CNN: {e}")
        # MPS op failed — retry once on CPU and record the fallback.
        device_used = "cpu"
        device_fallback = True
        model = TinyConvNet.build(IMG_SIZE, n_classes)
        try:
            model = _train_once(model, X_tr, y_tr, "cpu", seed)
        except Exception as e2:
            raise WorkerError("train_failed", "train", f"CNN failed on CPU too: {e2}")

    model.eval()
    with torch.no_grad():
        preds = model(X_te.to(device_used)).argmax(dim=1).cpu()
    accuracy = float((preds == y_te).float().mean())
    counts = torch.bincount(y_te)
    baseline = float(counts.max().item() / counts.sum().item())

    # Save checkpoint
    checkpoints_dir = job_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save model state dict with metadata
    checkpoint = {
        "state_dict": model.state_dict(),
        "img_size": IMG_SIZE,
        "n_classes": n_classes,
    }
    torch.save(checkpoint, checkpoints_dir / "model.pt")

    # Extract class names if available from dataset
    classes = []
    if hasattr(train_ds.features["label"], "names"):
        classes = train_ds.features["label"].names

    # Save manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "modality": "image",
        "model_type": "cnn",
        "framework": "pytorch",
        "img_size": IMG_SIZE,
        "n_classes": n_classes,
        "classes": classes,
    }
    (checkpoints_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "primary_metric": plan.get("primary_metric", "accuracy"),
        "metric_value": round(accuracy, 6),
        "baseline_metric": round(baseline, 6),
        "device": device_used,
        "device_fallback": device_fallback,
        "seed": seed,
        "split": plan.get("split", [0.8, 0.1, 0.1]),
        "n_train": int(X_tr.shape[0]),
        "n_val": 0,
        "n_test": int(X_te.shape[0]),
        "model_type": "cnn",
        "framework": "pytorch",
        "checkpoint_dir": "checkpoints",
    }
    if X_tr.shape[0] < SMALL_DATA_THRESHOLD:
        metrics["warning"] = (
            "Small dataset; high accuracy may reflect overfitting, not clinical signal."
        )
    return metrics


def predict(checkpoints_dir: Path, image_path: str) -> dict:
    """Load checkpoint and run inference on a single image."""
    import json
    import torch
    from PIL import Image
    from torchvision import transforms

    # Load manifest
    manifest_path = checkpoints_dir / "manifest.json"
    if not manifest_path.exists():
        raise WorkerError("checkpoint_missing", "load", "manifest.json not found")

    manifest = json.loads(manifest_path.read_text())
    img_size = manifest.get("img_size", IMG_SIZE)
    n_classes = manifest.get("n_classes", 2)
    classes = manifest.get("classes", [])

    # Load model checkpoint
    checkpoint_path = checkpoints_dir / "model.pt"
    if not checkpoint_path.exists():
        raise WorkerError("checkpoint_missing", "load", "model.pt not found")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = TinyConvNet.build(img_size, n_classes)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    except Exception as e:
        raise WorkerError("checkpoint_missing", "load", f"cannot load model: {e}")

    # Load and preprocess image
    img_path = Path(image_path)
    if not img_path.exists():
        raise WorkerError("bad_input", "predict", f"image not found: {image_path}")

    try:
        img = Image.open(img_path)
        if img.mode != "L":
            img = img.convert("L")

        tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        img_tensor = tfm(img).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        raise WorkerError("bad_input", "predict", f"cannot process image: {e}")

    # Run inference
    try:
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        # Map to class name if available
        if classes and pred_idx < len(classes):
            prediction = classes[pred_idx]
        else:
            prediction = f"Class {pred_idx}"

        return {
            "schema_version": SCHEMA_VERSION,
            "modality": "image",
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "detail": "Prototype label only.",
            "warning": "Not for clinical use.",
        }
    except Exception as e:
        raise WorkerError("predict_failed", "predict", f"inference failed: {e}")


