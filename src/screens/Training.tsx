import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { AppShell } from "../components/AppShell";
import { Icon } from "../components/Icon";
import { useRouter } from "../router";
import { friendlyError } from "../lib/errors";
import type { ExperimentDetail } from "../types/tauri";

interface Step {
  title: string;
  detail: string;
}

const STEPS: Step[] = [
  { title: "Dataset selected", detail: "Curated dataset loaded from registry." },
  { title: "Dataset inspected", detail: "Schema validated, missing values flagged." },
  { title: "Data prepared", detail: "Numeric features scaled; categorical variables encoded." },
  { title: "Train / eval / test split", detail: "80% / 10% / 10% stratified split, fixed seed." },
  { title: "Model training", detail: "Fitting gradient-boosted decision trees." },
  { title: "Evaluation", detail: "Scoring on held-out test set vs. majority baseline." },
  { title: "Best checkpoint saved", detail: "Highest-scoring model persisted to the experiment." },
  { title: "Model card generated", detail: "Doctor-facing summary written with limitations & risks." },
];

export function Training() {
  const { params, navigate } = useRouter();
  const experimentId = params.experimentId as string;
  const goal = (params.goal as string) ?? "Predict hospital readmission risk";
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [logOpen, setLogOpen] = useState(true);

  useEffect(() => {
    if (!experimentId) {
      navigate("home");
      return;
    }

    // Poll every 2 seconds until complete or failed
    const poll = setInterval(async () => {
      try {
        const exp = await invoke<ExperimentDetail>("get_experiment", { id: experimentId });
        setDetail(exp);
        if (exp.status === "complete" || exp.status === "failed") {
          clearInterval(poll);
          if (exp.status === "complete") {
            // Auto-navigate to Results after 1 second
            setTimeout(() => navigate("results", { experimentId }), 1000);
          }
        }
      } catch (e) {
        console.error("Failed to poll experiment:", e);
      }
    }, 2000);

    // Initial fetch
    invoke<ExperimentDetail>("get_experiment", { id: experimentId })
      .then(setDetail)
      .catch(console.error);

    return () => clearInterval(poll);
  }, [experimentId, navigate]);

  if (!detail) {
    return (
      <AppShell title="Training">
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <Icon name="hourglass_empty" size={48} className="mx-auto mb-4 animate-pulse text-accent" />
            <p className="font-headline-md text-headline-md text-text-primary">
              Loading experiment...
            </p>
          </div>
        </div>
      </AppShell>
    );
  }

  const trainingDone = detail.status === "complete";
  const trainingFailed = detail.status === "failed";
  const activeIndex = trainingDone ? STEPS.length : trainingFailed ? 4 : 4;

  function stateOf(i: number): "done" | "active" | "todo" {
    if (trainingFailed && i === 4) return "active";
    if (i < activeIndex) return "done";
    if (i === activeIndex) return "active";
    return "todo";
  }

  const logs = detail.workerStdout || "";

  return (
    <AppShell title="Training">
      <div className="mx-auto max-w-[1080px] p-8">
        <div className="mb-8 flex items-end justify-between">
          <div>
            <h2 className="mb-1 font-headline-lg text-headline-lg text-primary">
              {trainingDone ? "Training complete" : trainingFailed ? "Training failed" : "Training in progress"}
            </h2>
            <p className="font-body-md text-text-muted">
              {trainingFailed ? `Error: ${detail.errorMessage}` : `Autonomous prototype run for: ${goal}.`}
            </p>
          </div>
          {trainingDone && (
            <button
              onClick={() => navigate("results", { experimentId })}
              className="flex items-center gap-2 rounded bg-accent px-5 py-2 font-headline-md text-headline-md text-accent-on shadow-sm shadow-accent/20 transition-all hover:bg-accent-hover hover:shadow-md hover:shadow-accent/30 active:scale-[0.98] animate-fade-in"
            >
              View Results
              <Icon name="arrow_forward" size={18} />
            </button>
          )}
        </div>

        {/* Stepper */}
        <div className="mb-6 rounded-lg border border-border bg-surface p-8">
          <div className="relative flex flex-col gap-6">
            <div className="absolute left-4 top-4 bottom-8 z-0 w-px bg-border-strong" />
            {STEPS.map((step, i) => {
              const state = stateOf(i);
              return (
                <div key={step.title} className="relative z-10 flex gap-6">
                  <div
                    className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
                      state === "done"
                        ? "border border-success-text bg-success-bg"
                        : state === "active"
                          ? "border-2 border-accent bg-surface ring-4 ring-accent/15"
                          : "border border-outline-variant bg-surface"
                    }`}
                  >
                    {state === "done" && (
                      <Icon name="check" size={16} className="pop text-success-text" />
                    )}
                    {state === "active" && (
                      <span className="h-3 w-3 animate-pulse rounded-full bg-accent" />
                    )}
                  </div>
                  <div className={`pt-1 ${state === "todo" ? "opacity-50" : ""}`}>
                    <h3 className="font-headline-md text-headline-md text-primary">
                      {step.title}
                    </h3>
                    {state !== "todo" && (
                      <p className="mt-1 font-label-sm text-text-muted">
                        {step.detail}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Logs */}
        <div className="overflow-hidden rounded-lg border border-border bg-surface">
          <button
            onClick={() => setLogOpen((o) => !o)}
            className="flex w-full items-center justify-between bg-surface-muted p-4 transition-colors hover:bg-surface-container"
          >
            <div className="flex items-center gap-2">
              <Icon name="terminal" size={20} className="text-text-secondary" />
              <span className="font-headline-md text-headline-md text-primary">
                Technical logs
              </span>
            </div>
            <Icon
              name="expand_less"
              className={`text-text-secondary transition-transform duration-200 ${
                logOpen ? "" : "rotate-180"
              }`}
            />
          </button>
          {logOpen && (
            <pre className="h-64 overflow-y-auto whitespace-pre-wrap bg-log-bg p-4 font-code-sm text-code-sm text-log-text">
              {logs || "Waiting for worker output..."}
              {!trainingDone && !trainingFailed && <span className="animate-pulse">{"\n_"}</span>}
            </pre>
          )}
        </div>

        {trainingFailed && (
          <div className="mt-6 rounded-lg border border-error bg-error-bg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Icon name="error" className="text-error" />
              <h3 className="font-headline-md text-headline-md text-error">
                Training Failed
              </h3>
            </div>
            <p className="font-body-md text-text-primary mb-4">
              {friendlyError(detail.errorCode)}
            </p>
            <details className="mb-2">
              <summary className="cursor-pointer font-label-sm text-label-sm text-text-muted">
                Technical details
              </summary>
              <p className="mt-2 font-body-md text-text-secondary">
                Error code: {detail.errorCode || "unknown"}
              </p>
              <p className="mt-1 font-body-md text-text-secondary">
                {detail.errorMessage || "Worker failed without error details"}
              </p>
            </details>
            <div className="mt-4 flex gap-3">
              <button
                onClick={() => navigate("home")}
                className="rounded border border-border bg-surface px-4 py-2 font-body-md text-text-primary transition-colors hover:bg-surface-muted"
              >
                Start Over
              </button>
              <button
                onClick={() => navigate("experiments")}
                className="rounded border border-border bg-surface px-4 py-2 font-body-md text-text-primary transition-colors hover:bg-surface-muted"
              >
                View History
              </button>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}
