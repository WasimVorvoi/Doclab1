import { useEffect, useState } from "react";
import { AppShell } from "../components/AppShell";
import { Icon } from "../components/Icon";
import { useRouter } from "../router";
import { runAgent, type AgentResult } from "../agent";
import type { WorkerPlan } from "../types/tauri";

const STEPS = [
  { label: "Goal", icon: "check_circle", state: "done" },
  { label: "Agent plan", icon: "model_training", state: "active" },
  { label: "Training", icon: "data_exploration", state: "todo" },
  { label: "Evaluation", icon: "analytics", state: "todo" },
] as const;

function approachLabel(plan: WorkerPlan): string {
  if (plan.modality === "image") return "Medical image classifier";
  if (plan.modality === "text") return "Medical text summarizer";
  return "Structured-data predictor";
}

function metricLabel(metric: string): string {
  if (metric === "rouge_l") return "ROUGE-L similarity";
  return metric.replace(/_/g, " ");
}

function metricHelp(plan: WorkerPlan): string {
  if (plan.primary_metric === "rouge_l") {
    return "Compared with reference summaries and fixed qualitative examples.";
  }
  return "Compared against a majority-class baseline.";
}

export function Plan() {
  const { params, navigate } = useRouter();
  const goal = (params.goal as string) ?? "Predict hospital readmission risk";
  const [agentResult, setAgentResult] = useState<AgentResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [starting, setStarting] = useState(false);
  const [copiedDataset, setCopiedDataset] = useState(false);

  useEffect(() => {
    runAgent(goal)
      .then(setAgentResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [goal]);

  function startTraining() {
    if (!agentResult) return;
    setStarting(true);
    navigate("training", {
      goal,
      pendingRun: {
        plan: agentResult.planPreview.plan,
        goalText: goal,
        agentArtifacts: {
          intent: JSON.stringify(agentResult.intent),
          selection: JSON.stringify(agentResult.selection),
          profile: JSON.stringify(agentResult.profile),
        },
      },
    });
  }

  async function copyDatasetRef() {
    if (!agentResult) return;
    const dataset = agentResult.planPreview.dataset;
    await navigator.clipboard.writeText(`${dataset.hfId}@${dataset.revision}`);
    setCopiedDataset(true);
    window.setTimeout(() => setCopiedDataset(false), 1500);
  }

  if (loading) {
    return (
      <AppShell title="Planning...">
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <Icon name="model_training" size={48} className="mx-auto mb-4 animate-pulse text-accent" />
            <p className="font-headline-md text-headline-md text-text-primary">
              Agent analyzing goal...
            </p>
          </div>
        </div>
      </AppShell>
    );
  }

  if (error) {
    return (
      <AppShell title="Plan">
        <div className="mx-auto max-w-[720px] p-8">
          <div className="rounded-lg border border-error bg-error-bg p-6">
            <div className="mb-4 flex items-center gap-2">
              <Icon name="error" className="text-error" />
              <h3 className="font-headline-md text-headline-md text-error">
                Agent Error
              </h3>
            </div>
            <p className="mb-4 font-body-md text-text-primary">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="rounded border border-border bg-surface px-4 py-2 font-body-md text-text-primary transition-colors hover:bg-surface-muted"
            >
              Try Again
            </button>
          </div>
        </div>
      </AppShell>
    );
  }

  if (!agentResult) return null;

  const { intent, selection, planPreview } = agentResult;
  const { dataset, plan, summary } = planPreview;

  // Build agent log from real steps
  const logLines = [
    { tag: "OK", text: `Parsed intent: "${intent.task_type}" task, ${intent.modality} modality` },
    { tag: "OK", text: `Queried curated dataset registry: ${selection.dataset_name}` },
    { tag: "OK", text: `Selected training approach: ${approachLabel(plan)}` },
    { tag: ">", text: "Plan finalized — awaiting your confirmation..." },
  ];

  return (
    <AppShell title="Prototype Plan">
      <div className="mx-auto max-w-[1080px] space-y-6 p-8">
        {/* Stepper */}
        <div className="flex flex-wrap items-center gap-2 font-label-sm text-label-sm text-text-muted">
          {STEPS.map((s, i) => (
            <div key={s.label} className="flex items-center gap-2">
              <span
                className={`flex items-center gap-1 ${
                  s.state === "done"
                    ? "text-success-text"
                    : s.state === "active"
                      ? "font-semibold text-accent"
                      : ""
                }`}
              >
                <Icon
                  name={s.icon}
                  size={16}
                  className={s.state === "active" ? "animate-pulse" : ""}
                />
                {s.label}
              </span>
              {i < STEPS.length - 1 && (
                <span className="h-px w-8 bg-border-strong" />
              )}
            </div>
          ))}
        </div>

        {/* Plan panel */}
        <div className="overflow-hidden rounded-lg border border-border bg-surface">
          <div className="flex items-center justify-between border-b border-border bg-surface-muted px-6 py-4">
            <div className="flex items-center gap-3">
              <Icon name="memory" className="text-accent" />
              <h3 className="font-headline-md text-headline-md text-primary">
                Planning agent — compiled prototype
              </h3>
            </div>
            <span className="rounded bg-surface-container-high px-2 py-1 font-code-sm text-code-sm text-text-secondary">
              Status: Ready
            </span>
          </div>

          <div className="stagger grid grid-cols-1 gap-4 p-6 md:grid-cols-2">
            {/* Objective */}
            <div className="col-span-1 rounded border border-border bg-background p-4 md:col-span-2">
              <h4 className="mb-2 font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                Primary objective
              </h4>
              <p className="font-headline-md text-headline-md text-primary">
                {goal}
              </p>
            </div>

            {/* Task + modality */}
            <div className="space-y-4 rounded border border-border bg-background p-4">
              <div>
                <h4 className="mb-1 font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  Task type
                </h4>
                <div className="flex items-center gap-2">
                  <Icon name="category" size={18} className="text-secondary" />
                  <span className="font-body-md text-text-primary capitalize">
                    {intent.task_type}
                  </span>
                </div>
              </div>
              <div className="border-t border-border pt-3">
                <h4 className="mb-1 font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  Data modality
                </h4>
                <div className="flex items-center gap-2">
                  <Icon name="table_rows" size={18} className="text-secondary" />
                  <span className="font-body-md text-text-primary capitalize">
                    {intent.modality} healthcare data
                  </span>
                </div>
              </div>
            </div>

            {/* Dataset */}
            <div className="flex flex-col justify-between rounded border border-border bg-background p-4">
              <div>
                <h4 className="mb-2 font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  Selected dataset
                </h4>
                <p className="mb-3 font-body-md text-text-primary">
                  {dataset.name}
                </p>
              </div>
              <div className="flex items-center justify-between gap-3 rounded border border-surface-tint bg-log-bg p-3">
                <code className="font-code-sm text-code-sm text-log-text">
                  {dataset.hfId}@{dataset.revision.substring(0, 7)}
                </code>
                <button
                  type="button"
                  onClick={copyDatasetRef}
                  title="Copy dataset reference"
                  className="rounded p-1 text-text-muted transition-colors hover:bg-surface-muted hover:text-log-text"
                >
                  <Icon name={copiedDataset ? "check" : "content_copy"} size={16} />
                </button>
              </div>
            </div>

            {/* Model + metric + rationale */}
            <div className="col-span-1 grid grid-cols-1 gap-6 rounded border border-border bg-background p-4 md:col-span-2 md:grid-cols-3">
              <div className="border-border pr-4 md:border-r">
                <h4 className="mb-2 font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  Training approach
                </h4>
                <p className="mb-2 font-headline-md text-headline-md text-primary">
                  {approachLabel(plan)}
                </p>
                <span className="inline-block rounded border border-border-strong bg-surface-muted px-2 py-0.5 font-code-sm text-code-sm text-text-secondary">
                  Runs locally · {plan.device.toUpperCase()}
                </span>
              </div>
              <div className="border-border pr-4 md:border-r">
                <h4 className="mb-2 font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  Target metric
                </h4>
                <p className="font-headline-md text-headline-md text-primary capitalize">
                  {metricLabel(plan.primary_metric)}
                </p>
                <p className="mt-1 font-label-sm text-text-muted">
                  {metricHelp(plan)}
                </p>
              </div>
              <div>
                <h4 className="mb-2 font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  Why this approach
                </h4>
                <p className="font-body-md text-sm leading-relaxed text-text-primary">
                  {summary}
                </p>
              </div>
            </div>
          </div>

          {/* Approved-data confirmation */}
          <div className="border-t border-border bg-warning-bg/40 px-6 py-4">
            <label className="flex cursor-pointer items-start gap-3">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(e) => setConfirmed(e.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-border-strong text-accent accent-accent focus:ring-accent"
              />
              <span className="font-body-md text-sm text-text-secondary">
                I confirm this prototype uses{" "}
                <span className="font-semibold text-text-primary">
                  approved public / de-identified data only
                </span>{" "}
                — no PHI, EHR exports, or private patient uploads — and is for
                research &amp; prototyping, not clinical care.
              </span>
            </label>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 border-t border-border bg-background px-6 py-4">
            <button
              onClick={() => navigate("home")}
              className="rounded border border-transparent px-4 py-2 font-body-md text-text-primary transition-colors hover:border-border hover:bg-surface-muted"
            >
              Back
            </button>
            <button
              disabled={!confirmed || starting}
              onClick={startTraining}
              className="flex items-center gap-2 rounded bg-accent px-6 py-2 font-headline-md text-headline-md text-accent-on shadow-sm shadow-accent/20 transition-all hover:bg-accent-hover hover:shadow-md hover:shadow-accent/30 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
            >
              {starting ? (
                <>
                  <Icon name="hourglass_empty" size={18} className="animate-pulse" />
                  Starting...
                </>
              ) : (
                <>
                  <Icon name="play_arrow" size={18} />
                  Start Prototype
                </>
              )}
            </button>
          </div>
        </div>

        {/* Agent execution log */}
        <div className="rounded-lg border border-[#2a2a2a] bg-log-bg p-4 font-code-sm text-code-sm text-log-text opacity-90 transition-opacity hover:opacity-100">
          <div className="mb-2 flex items-center gap-2 text-[#888]">
            <Icon name="terminal" size={14} />
            <span>Agent execution log</span>
          </div>
          <div className="space-y-1">
            {logLines.map((line, i) => (
              <p key={i}>
                <span
                  className={
                    line.tag === "OK" ? "text-success-text" : "text-[#888]"
                  }
                >
                  {line.tag === "OK" ? "[OK]" : ">"}
                </span>{" "}
                {line.text}
              </p>
            ))}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
