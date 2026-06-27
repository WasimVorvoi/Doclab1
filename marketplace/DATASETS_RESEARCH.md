# Hugging Face dataset research (curated for DocLab)

_Last searched: 2026-05-31. Only datasets in `datasets.yaml` are loadable at runtime._

## Selection criteria

- Public, research-friendly, no private patient uploads
- Clear label/target column for the worker contract
- Loads with current `datasets` library (no deprecated loading scripts)
- Pinned `revision` commit SHA in YAML

## Included in `datasets.yaml`

| `id` | HF repo | Why |
|------|---------|-----|
| `diabetes_readmission` | `imodels/diabetes-readmission` | Phase 1 golden path; binary readmission; already verified with M2 worker |
| `heart_disease_uci` | `buio/heart-disease` | Second tabular option; 303 rows; `target` 0/1; fast CPU train |
| `chest_xray_pneumonia` | `hf-vision/chest-xray-pneumonia` | Binary NORMAL/PNEUMONIA; standard demo for M9 image path |
| `medical_text_summarization_synthetic` | `mustaphounii04/Synthetic-Medical-Text-Summarization` | `Text` + `Summary`; synthetic; safe for M10 |

## Considered but not indexed

| HF repo | Reason skipped |
|---------|----------------|
| `keremberke/chest-xray-classification` | Legacy dataset script; fails on modern `datasets` |
| `trpakov/chest-xray-classification` | Same — loading script no longer supported |
| `albertvillanova/medmnist-v2` | MedMNIST v2 script unsupported without conversion |
| `UniqueData/chest-x-rays` | Only 97 images but **17-class** `type` — poor fit for binary normal/abnormal demo |
| `supersam7/hospital_readmission_rates_2020` | Aggregate hospital stats, not patient-level tabular ML |
| `auphong2707/hospital-readmission-risk-data` | Low adoption; unverified schema vs. imodels port |

## Agent keyword hints

- Readmission → `diabetes_readmission`
- Heart / cardiology / chest pain → `heart_disease_uci`
- X-ray / pneumonia / normal abnormal image → `chest_xray_pneumonia`
- Summarize / medical text → `medical_text_summarization_synthetic`
