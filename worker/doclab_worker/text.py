"""Text summarization path (M10): LoRA fine-tune a small seq2seq model.

Mirrors the worker contract — reads a plan, returns a metrics dict — but the
metric is ROUGE-L (a similarity score, not accuracy) and the metrics carry a
few fixed input->prediction->reference examples for the model card. Subsamples
and caps epochs/steps so a demo run finishes in a few minutes. Uses MPS on
Apple Silicon with a one-shot CPU fallback.
"""

from __future__ import annotations

from pathlib import Path

from .device import resolve_device
from .errors import WorkerError

SCHEMA_VERSION = 1

MODEL_NAME = "google-t5/t5-small"
MAX_TRAIN = 200
MAX_EVAL = 40
MAX_INPUT_TOKENS = 256
MAX_TARGET_TOKENS = 64
EPOCHS = 1
BATCH_SIZE = 8
N_EXAMPLES = 3
PREFIX = "summarize: "


def load_text_data(plan: dict, data_root: Path):
    """Load a summarization dataset, pinned to revision and cached on disk.

    Returns (train_rows, eval_rows) as lists of {"text", "summary"} dicts. The
    column names come from the plan: `text_column` (default "Text") and
    `label_column` (the summary, default "Summary").
    """
    hf_id = plan.get("hf_id")
    revision = plan.get("revision")
    if not hf_id or not revision:
        raise WorkerError("bad_plan", "load", "plan missing hf_id/revision")
    text_col = plan.get("text_column", "Text")
    summary_col = plan.get("label_column", "Summary")
    try:
        from datasets import load_dataset

        cache_dir = data_root / "datasets" / plan.get("dataset_id", "unknown")
        ds = load_dataset(hf_id, revision=revision, cache_dir=str(cache_dir))
    except Exception as e:
        raise WorkerError("dataset_missing", "load", f"failed to load {hf_id}: {e}")

    names = list(ds.keys())
    split = ds["train"] if "train" in ds else ds[names[0]]
    if text_col not in split.column_names or summary_col not in split.column_names:
        raise WorkerError(
            "bad_plan",
            "load",
            f"columns {text_col!r}/{summary_col!r} not in {split.column_names}",
        )

    rows = [
        {"text": str(r[text_col]), "summary": str(r[summary_col])}
        for r in split.select(range(min(len(split), MAX_TRAIN + MAX_EVAL)))
    ]
    train_rows = rows[:MAX_TRAIN]
    eval_rows = rows[MAX_TRAIN : MAX_TRAIN + MAX_EVAL]
    if not eval_rows:  # tiny dataset — reuse the tail of train for eval
        eval_rows = train_rows[-min(len(train_rows), MAX_EVAL) :]
    return train_rows, eval_rows


def _build_model(seed):
    """Load t5-small and wrap it with a LoRA adapter."""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )
    return get_peft_model(base, lora), tok


def _encode(tok, rows):
    import torch

    inputs = tok(
        [PREFIX + r["text"] for r in rows],
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    labels = tok(
        [r["summary"] for r in rows],
        max_length=MAX_TARGET_TOKENS,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    labels[labels == tok.pad_token_id] = -100
    return inputs["input_ids"], inputs["attention_mask"], labels


def _train_once(model, tok, rows, device, seed):
    """One short LoRA fine-tune pass. Raises on failure (caller handles fallback)."""
    import torch

    torch.manual_seed(seed)
    model = model.to(device)
    input_ids, attn, labels = _encode(tok, rows)
    input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    n = input_ids.shape[0]
    for _ in range(EPOCHS):
        for start in range(0, n, BATCH_SIZE):
            sl = slice(start, start + BATCH_SIZE)
            opt.zero_grad()
            out = model(
                input_ids=input_ids[sl],
                attention_mask=attn[sl],
                labels=labels[sl],
            )
            out.loss.backward()
            opt.step()
    return model


def _generate(model, tok, texts, device):
    import torch

    model.eval()
    enc = tok(
        [PREFIX + t for t in texts],
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        ids = model.generate(
            **enc, max_new_tokens=MAX_TARGET_TOKENS, num_beams=2
        )
    return tok.batch_decode(ids, skip_special_tokens=True)


def _rouge_l(predictions, references) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return float(sum(scores) / len(scores)) if scores else 0.0


def run_job(plan: dict, data_root: Path, job_dir: Path) -> dict:
    """LoRA summarization pipeline. Returns a metrics dict with ROUGE-L as the
    primary metric plus a few fixed input->prediction->reference examples."""
    import json

    seed = plan.get("seed", 42)
    train_rows, eval_rows = load_text_data(plan, data_root)

    requested = resolve_device()
    device_used = requested
    device_fallback = False
    model, tok = _build_model(seed)
    try:
        model = _train_once(model, tok, train_rows, requested, seed)
    except Exception as e:
        if requested == "cpu":
            raise WorkerError("train_failed", "train", f"LoRA fine-tune failed: {e}")
        device_used = "cpu"
        device_fallback = True
        model, tok = _build_model(seed)
        try:
            model = _train_once(model, tok, train_rows, "cpu", seed)
        except Exception as e2:
            raise WorkerError("train_failed", "train", f"LoRA failed on CPU too: {e2}")

    eval_texts = [r["text"] for r in eval_rows]
    eval_refs = [r["summary"] for r in eval_rows]
    try:
        preds = _generate(model, tok, eval_texts, device_used)
    except Exception:
        if not device_fallback:
            device_used = "cpu"
            device_fallback = True
            model = model.to("cpu")
            preds = _generate(model, tok, eval_texts, "cpu")
        else:
            raise WorkerError("train_failed", "eval", "generation failed")

    rouge_l = _rouge_l(preds, eval_refs)
    examples = [
        {
            "input": eval_texts[i][:400],
            "prediction": preds[i],
            "reference": eval_refs[i][:400],
        }
        for i in range(min(N_EXAMPLES, len(eval_texts)))
    ]

    # Save checkpoint
    checkpoints_dir = job_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter and tokenizer
    adapter_dir = checkpoints_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    # Save manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "modality": "text",
        "model_type": "lora_t5_small",
        "framework": "transformers",
        "text_column": plan.get("text_column", "Text"),
        "label_column": plan.get("label_column", "Summary"),
        "model_name": MODEL_NAME,
    }
    (checkpoints_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "schema_version": SCHEMA_VERSION,
        "primary_metric": "rouge_l",
        "metric_value": round(rouge_l, 6),
        "baseline_metric": 0.0,
        "device": device_used,
        "device_fallback": device_fallback,
        "seed": seed,
        "split": plan.get("split", [0.8, 0.1, 0.1]),
        "n_train": len(train_rows),
        "n_val": 0,
        "n_test": len(eval_rows),
        "model_type": "lora_t5_small",
        "framework": "transformers",
        "examples": examples,
        "checkpoint_dir": "checkpoints",
    }


def predict(checkpoints_dir: Path, input_text: str) -> dict:
    """Load checkpoint and generate summary for input text."""
    import json
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Load manifest
    manifest_path = checkpoints_dir / "manifest.json"
    if not manifest_path.exists():
        raise WorkerError("checkpoint_missing", "load", "manifest.json not found")

    manifest = json.loads(manifest_path.read_text())
    model_name = manifest.get("model_name", MODEL_NAME)

    # Load adapter
    adapter_dir = checkpoints_dir / "adapter"
    if not adapter_dir.exists():
        raise WorkerError("checkpoint_missing", "load", "adapter directory not found")

    try:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        model.eval()
    except Exception as e:
        raise WorkerError("checkpoint_missing", "load", f"cannot load model: {e}")

    # Validate input
    if not input_text or not input_text.strip():
        raise WorkerError("bad_input", "predict", "input text is empty")

    # Generate summary
    try:
        device = resolve_device()
        model = model.to(device)

        inputs = tokenizer(
            PREFIX + input_text,
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_TOKENS,
                num_beams=2,
            )

        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "schema_version": SCHEMA_VERSION,
            "modality": "text",
            "prediction": summary,
            "confidence": None,
            "detail": "Generated summary (not validated against reference).",
            "warning": "Not for clinical use.",
        }
    except Exception as e:
        raise WorkerError("predict_failed", "predict", f"generation failed: {e}")

