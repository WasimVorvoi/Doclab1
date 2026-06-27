# DocLab

**Doctor Laboratory** — a local-first healthcare ML prototyping lab.

Describe a clinical AI idea in plain language. DocLab finds an approved public dataset, shows you a training plan, runs the prototype locally, and saves metrics plus a **model card**—without writing ML code or picking frameworks yourself.

> **Not for clinical use.** DocLab is for research and prototyping only. It does not diagnose, recommend treatment, or connect to hospital EHRs. No private patient uploads.

---

## How it works

```
You describe a goal  →  Agent + curated datasets  →  You approve the plan
  →  Train locally  →  Metrics + experiment history  →  Model card
```

| You might say | DocLab handles |
|---------------|----------------|
| Predict hospital readmission risk | Tabular data → XGBoost → accuracy + baseline check |
| Classify medical images as normal or abnormal | Image dataset → CNN → accuracy + small-data note |
| Summarize medical education text | Text dataset → small LM + LoRA → ROUGE-L + examples |

Full product and build plan: **[spec.md](./Current/spec.md)**  
Live demo script: **[DEMO.md](./Current/DEMO.md)**

---

## Status

Hackathon MVP, feature-complete. All three modality paths work end-to-end: **tabular** (XGBoost),
**image** (PyTorch CNN), and **text** (Transformers + LoRA). Milestones M0–M10 are done; M11
(demo rehearsal) is the only remaining work and is presenter-side, not code.

See [MILESTONES.md](./Current/MILESTONES.md) for per-milestone status and [spec.md](./Current/spec.md)
for scope.

---

## Stack

| Layer | Tech |
|-------|------|
| Desktop | [Tauri](https://tauri.app/) |
| UI | React + Tailwind |
| Orchestration | Rust, SQLite, Tokio |
| ML worker | Python (XGBoost, PyTorch, Transformers + LoRA) |
| Data | Curated Hugging Face datasets (indexed locally) |

---

## Project layout

```
DocLAB/
├── README.md
├── Current/
│   ├── spec.md             # Vision, architecture, hackathon scope (source of truth)
│   ├── MILESTONES.md       # Per-milestone status + exit criteria
│   ├── M0_PLAN.md … M11_PLAN.md  # Per-milestone implementation notes
│   ├── TESTING.md          # How to write tests in this repo
│   ├── DEMO.md             # Judge demo script
│   ├── UI.md               # UI design reference
│   └── handoff.md          # Current handoff / next-step summary
├── Archived_Plans/         # Superseded earlier spec/demo/handoff
├── marketplace/
│   └── datasets.yaml       # Curated dataset index (source of truth)
├── src/                    # Tauri + React app
├── src-tauri/              # Rust shell (orchestration, SQLite, Tauri commands)
├── worker/                 # Python training worker (tabular / image / text)
├── demo/seed_experiment/   # Committed fallback artifact bundle
└── ~/.doclab/              # Runtime data (created on first run)
    ├── doclab.db
    ├── datasets/
    └── experiments/
```

---

## Quick start

### Prerequisites

- macOS, Windows, or Linux
- [Rust](https://rustup.rs/) (for Tauri)
- [Node.js](https://nodejs.org/) 18+
- Python 3.11+ with venv
- **Apple Silicon:** PyTorch **MPS** for image/text training (recommended on MacBook demo)
- **Intel Mac / no GPU:** CPU for all modalities
- NVIDIA CUDA not used (Mac has no CUDA)
- **macOS + XGBoost:** `brew install libomp` (else the tabular path falls back to LogisticRegression)

### Run

```bash
# Clone the repo
git clone <repo-url>
cd DocLAB

# Python worker (set up first so the Rust shell can find the venv)
cd worker
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..

# Frontend + Tauri (full app)
npm install
npm run tauri dev
```

Optional, for an offline demo: `python worker/scripts/prefetch.py` pre-caches the curated datasets.

Data and experiments are stored under `~/.doclab/` (`doclab.db`, `datasets/<id>/`,
`experiments/<id>/`), created on first run. See [spec.md](./Current/spec.md) for details.

### Environment Variables

DocLab supports optional LLM-assisted planning for ambiguous goals:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DOCLAB_AGENT_MODE` | `rules` (keyword-based) \| `hybrid` (rules + LLM) \| `llm` (LLM-first) | `rules` |
| `DOCLAB_LLM_PROVIDER` | `openai` \| `anthropic` | `openai` |
| `OPENAI_API_KEY` | Your OpenAI API key (if using OpenAI) | - |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (if using Anthropic) | - |
| `DOCLAB_LLM_MODEL` | Model name | `gpt-4o-mini` or `claude-3-5-haiku-20241022` |
| `HF_TOKEN` | Optional Hugging Face token for dataset downloads | - |

**Default behavior (no API key):** DocLab uses rule-based planning only. The app works identically without any LLM configuration.

**Hybrid mode:** LLM assists only when goals are ambiguous (long or conflicting keywords). Falls back to rules if LLM fails.

**Security:** API keys are read by Rust only, never exposed to the frontend bundle. LLM calls happen locally via the Tauri backend. Training always runs locally—no data leaves your machine.

---

## Try a saved prototype

After training completes, DocLab saves model checkpoints locally. You can test your trained prototypes with new inputs directly from the Results screen:

- **Tabular models**: Paste a JSON object with feature names and values matching the training schema
- **Image models**: Select a local test image and get a prediction with confidence score
- **Text models**: Paste text and generate a summary

**Important:** Use only de-identified, synthetic, or public test inputs. Predictions are for research and prototyping only—not for clinical care.

Checkpoints are stored in `~/.doclab/experiments/<id>/checkpoints/` and include:
- Model weights (`model.joblib`, `model.pt`, or LoRA adapter)
- Preprocessing metadata
- Manifest with model configuration

To run predictions from the command line:
```bash
cd worker
source .venv/bin/activate
python -m doclab_worker --predict path/to/predict_request.json
```

---

## Documentation

| Doc | Description |
|-----|-------------|
| [Current/spec.md](./Current/spec.md) | Full hackathon spec: vision, agent, marketplace, training, eval, phases |
| [Current/DEMO.md](./Current/DEMO.md) | Step-by-step demo for judges with fallbacks |

---

## Safety & scope

DocLab intentionally avoids:

- Private patient data uploads
- Hospital EHR integration
- Clinical decision support claims
- Community / sharing marketplace (for now)

Use only **public, synthetic, or de-identified** datasets from the curated marketplace.

---

## License

TBD — add before public release.
