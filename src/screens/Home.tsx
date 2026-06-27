import { useRef, useState, type CSSProperties } from "react";
import { AppShell } from "../components/AppShell";
import { Icon } from "../components/Icon";
import { useRouter } from "../router";
import type { Dataset } from "../types/tauri";

interface Example {
  icon: string;
  title: string;
  blurb: string;
  goal: string;
  accent: string;
}

const EXAMPLES: Example[] = [
  {
    icon: "analytics",
    title: "Predict readmission risk",
    blurb: "Using EHR-style tabular data and historical patient flows.",
    goal: "Predict hospital readmission risk from patient-style tabular data",
    accent: "#0f766e", // teal
  },
  {
    icon: "medical_information",
    title: "Classify medical images",
    blurb: "Identify anomalies in chest X-rays from a curated public set.",
    goal: "Classify chest X-ray images as normal or abnormal",
    accent: "#4f46e5", // indigo
  },
  {
    icon: "summarize",
    title: "Summarize medical education text",
    blurb: "Extract key insights from open clinical-education passages.",
    goal: "Summarize medical education text into concise notes",
    accent: "#b45309", // amber
  },
  {
    icon: "table_chart",
    title: "Classify clinical tabular records",
    blurb: "Predict an outcome label from structured encounter data.",
    goal: "Classify diabetic patient encounters by readmission outcome",
    accent: "#be185d", // rose
  },
];

export function Home() {
  const { navigate, params } = useRouter();
  const [goal, setGoal] = useState((params.prefillGoal as string) ?? "");
  const [attached, setAttached] = useState<Dataset | null>(
    (params.attachedDataset as Dataset) ?? null,
  );
  const [upload, setUpload] = useState<string | null>(
    (params.upload as string) ?? null,
  );
  const fileInputRef = useRef<HTMLInputElement>(null);

  function start(text: string) {
    const trimmed = text.trim();
    if (!trimmed) return;
    // Fold the attached dataset into the goal so the agent prefers it.
    const fullGoal = attached
      ? `${trimmed} (using the ${attached.name} dataset)`
      : trimmed;
    navigate("plan", { goal: fullGoal, datasetId: attached?.id });
  }

  return (
    <AppShell showSearch>
      <div className="relative flex min-h-full flex-col items-center justify-center overflow-hidden px-8 py-12">
        {/* Decorative hero backdrop */}
        <div aria-hidden className="pointer-events-none absolute inset-0">
          <div className="bg-grid absolute inset-0" />
          <div className="animate-aurora absolute -top-20 right-[14%] h-72 w-72 rounded-full bg-accent/12 blur-3xl" />
          <div
            className="animate-aurora absolute left-[6%] top-1/3 h-64 w-64 rounded-full bg-accent/[0.06] blur-3xl"
            style={{ animationDelay: "-8s" }}
          />
        </div>

        <div className="relative z-10 w-full max-w-[720px] animate-fade-in-up">
          {/* Eyebrow */}
          <div className="mb-5 flex justify-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/[0.06] px-3 py-1 font-label-sm text-label-sm font-medium text-accent">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-accent" />
              </span>
              Local-first · runs offline · no PHI ever leaves this machine
            </span>
          </div>

          <h2 className="mb-8 text-center font-headline-lg text-[34px] leading-[1.12] font-semibold tracking-tight text-primary">
            What healthcare AI model do you
            <br className="hidden sm:block" /> want to{" "}
            <span className="text-accent">prototype</span>?
          </h2>

          {/* Goal input */}
          <div className="relative rounded-xl border border-border bg-surface-container-lowest p-1 shadow-lg shadow-black/5 transition-all focus-within:border-accent focus-within:shadow-accent/10 focus-within:ring-2 focus-within:ring-accent/30">
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) start(goal);
              }}
              spellCheck={false}
              placeholder="I want to predict hospital readmission risk from patient-style tabular data..."
              className="h-[160px] w-full resize-none border-none bg-transparent p-4 font-body-md text-body-md text-text-primary placeholder:text-text-disabled focus:outline-none focus:ring-0"
            />
            {/* Attached items (uploaded file + curated dataset) */}
            {(attached || upload) && (
              <div className="flex flex-wrap gap-2 px-3 pb-2">
                {upload && (
                  <span className="inline-flex max-w-full items-center gap-2 rounded-lg border border-border-strong bg-surface-muted py-1.5 pl-2.5 pr-1.5 font-label-sm text-label-sm text-text-primary animate-fade-in">
                    <Icon name="description" size={14} className="shrink-0 text-text-secondary" />
                    <span className="truncate">{upload}</span>
                    <button
                      type="button"
                      aria-label="Remove upload"
                      onClick={() => setUpload(null)}
                      className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-text-muted transition-colors hover:bg-surface-container hover:text-text-primary"
                    >
                      <Icon name="close" size={14} />
                    </button>
                  </span>
                )}
                {attached && (
                  <span className="inline-flex max-w-full items-center gap-2 rounded-lg border border-border-strong bg-surface-muted py-1.5 pl-2.5 pr-1.5 font-label-sm text-label-sm text-text-primary animate-fade-in">
                    <Icon name="database" size={14} className="shrink-0 text-text-secondary" />
                    <span className="truncate">{attached.name}</span>
                    <button
                      type="button"
                      aria-label="Remove dataset"
                      onClick={() => setAttached(null)}
                      className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-text-muted transition-colors hover:bg-surface-container hover:text-text-primary"
                    >
                      <Icon name="close" size={14} />
                    </button>
                  </span>
                )}
              </div>
            )}

            <div className="flex items-center justify-between px-3 pb-2">
              <div className="flex items-center gap-2">
                {/* Upload a local file */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.tsv,.json,.parquet,.zip"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) setUpload(f.name);
                    e.target.value = "";
                  }}
                />
                <button
                  type="button"
                  aria-label="Upload a file"
                  title="Upload a local data file"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-text-secondary transition-all hover:border-border-strong hover:bg-surface-muted hover:text-text-primary active:scale-95"
                >
                  <Icon name="add" size={18} />
                </button>
                <button
                  type="button"
                  onClick={() => navigate("datasets", { goal, upload })}
                  className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 font-label-sm text-label-sm transition-all active:scale-[0.98] ${
                    attached
                      ? "border-border-strong bg-surface-muted text-text-primary"
                      : "border-border text-text-secondary hover:border-border-strong hover:bg-surface-muted"
                  }`}
                >
                  <Icon name={attached ? "swap_horiz" : "attach_file"} size={16} />
                  {attached ? "Change dataset" : "Attach dataset"}
                </button>
              </div>
              <button
                onClick={() => start(goal)}
                disabled={!goal.trim()}
                className="flex items-center gap-2 rounded bg-accent px-5 py-2 font-headline-md text-headline-md text-accent-on shadow-sm shadow-accent/20 transition-all hover:bg-accent-hover hover:shadow-md hover:shadow-accent/30 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
              >
                Start Prototype
                <Icon name="arrow_forward" size={18} />
              </button>
            </div>
          </div>

          <p className="mt-3 flex items-center justify-center gap-1.5 text-center font-label-sm text-label-sm text-text-muted">
            <Icon name="info" size={14} />
            Research &amp; prototyping only — not for clinical care. Curated public data only; no PHI or patient uploads.
          </p>

          {/* Examples */}
          <div className="mt-10">
            <p className="mb-4 text-center font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
              Example scenarios
            </p>
            <div className="stagger grid grid-cols-1 gap-3 sm:grid-cols-2">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={ex.title}
                  style={{ "--i": i + 1, "--color-accent": ex.accent } as CSSProperties}
                  onClick={() => {
                    setGoal(ex.goal);
                    start(ex.goal);
                  }}
                  className="card-accent group relative overflow-hidden rounded-lg border border-border bg-surface p-4 text-left transition-all hover:border-accent/40 hover:bg-accent/5 hover:-translate-y-0.5 hover:shadow-md hover:shadow-black/5"
                >
                  <div className="flex items-start gap-3">
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent/10 text-accent transition-transform duration-300 group-hover:scale-110">
                      <Icon name={ex.icon} size={20} />
                    </span>
                    <div>
                      <h3 className="mb-1 font-headline-md text-headline-md text-text-primary">
                        {ex.title}
                      </h3>
                      <p className="font-label-sm text-label-sm text-text-muted">
                        {ex.blurb}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
