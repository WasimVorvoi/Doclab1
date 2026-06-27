import { useEffect, useState, type CSSProperties } from "react";
import { invoke } from "@tauri-apps/api/core";
import { AppShell } from "../components/AppShell";
import { Icon } from "../components/Icon";
import { Badge } from "../components/Badge";
import { useRouter } from "../router";
import type { ExperimentSummary } from "../types/tauri";

const STATUS_TONE: Record<string, "success" | "running" | "error" | "neutral"> = {
  complete: "success",
  running: "running",
  failed: "error",
};

function formatMetric(exp: ExperimentSummary): string {
  if (exp.metricValue === null) return "—";
  if (exp.primaryMetric === "rouge_l") return exp.metricValue.toFixed(2);
  return `${(exp.metricValue * 100).toFixed(1)}%`;
}

export function Experiments() {
  const { navigate } = useRouter();
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    invoke<ExperimentSummary[]>("list_experiments")
      .then(setExperiments)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <AppShell title="Experiments">
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <Icon name="hourglass_empty" size={48} className="mx-auto mb-4 animate-pulse text-accent" />
            <p className="font-headline-md text-headline-md text-text-primary">
              Loading experiments...
            </p>
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell title="Experiments">
      <div className="mx-auto max-w-[1080px] p-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="h-7 w-1 rounded-full bg-accent" />
              <h1 className="font-headline-lg text-headline-lg text-text-primary">
                Experiments
              </h1>
            </div>
            <p className="mt-1 font-body-md text-text-muted">
              Every prototype run, newest first.
            </p>
          </div>
          <button className="flex items-center gap-2 rounded border border-border bg-surface-container-lowest px-3 py-2 font-label-sm text-label-sm text-text-primary transition-colors hover:bg-surface-muted">
            <Icon name="filter_list" size={16} /> Filter
          </button>
        </div>

        {experiments.length === 0 ? (
          <div className="rounded-lg border border-border bg-surface p-12 text-center">
            <Icon name="science" size={48} className="mx-auto mb-4 text-text-muted" />
            <h3 className="mb-2 font-headline-md text-headline-md text-text-primary">
              No experiments yet
            </h3>
            <p className="mb-4 font-body-md text-text-muted">
              Start your first prototype from the Home screen.
            </p>
            <button
              onClick={() => navigate("home")}
              className="rounded bg-accent px-4 py-2 font-headline-md text-headline-md text-accent-on shadow-sm shadow-accent/20 transition-all hover:bg-accent-hover hover:shadow-md hover:shadow-accent/30 active:scale-[0.98]"
            >
              Go to Home
            </button>
          </div>
        ) : (
          <div className="overflow-hidden rounded-lg border border-border bg-surface-container-lowest">
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left">
                <thead>
                  <tr className="border-b border-border bg-background font-label-sm text-label-sm text-text-muted">
                    <th className="px-4 py-3 font-normal">Experiment</th>
                    <th className="px-4 py-3 font-normal">Goal</th>
                    <th className="px-4 py-3 font-normal">Dataset</th>
                    <th className="px-4 py-3 font-normal">Metric</th>
                    <th className="px-4 py-3 font-normal">Status</th>
                    <th className="px-4 py-3 text-right font-normal">Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border font-body-md text-body-md text-text-primary">
                  {experiments.map((exp, i) => (
                    <tr
                      key={exp.id}
                      data-reveal="fade"
                      style={{ "--reveal-delay": `${Math.min(i, 9) * 45}ms` } as CSSProperties}
                      onClick={() => navigate("results", { experimentId: exp.id })}
                      className="group cursor-pointer transition-colors hover:bg-accent/5 hover:shadow-[inset_3px_0_0_0_var(--color-accent)]"
                    >
                      <td className="whitespace-nowrap px-4 py-3">
                        <span className="rounded bg-surface-container px-2 py-1 font-code-sm text-code-sm text-text-secondary">
                          {exp.id}
                        </span>
                        {exp.isBest && (
                          <span className="ml-2 inline-flex items-center gap-1 rounded-full bg-success-bg px-2 py-0.5 font-label-sm text-label-sm text-success-text">
                            <Icon name="star" size={12} /> Best
                          </span>
                        )}
                      </td>
                      <td className="max-w-[300px] truncate px-4 py-3" title={exp.goalText}>
                        {exp.goalText}
                      </td>
                      <td className="px-4 py-3 text-text-secondary">{exp.datasetId}</td>
                      <td className="px-4 py-3">
                        {exp.metricValue !== null ? (
                          formatMetric(exp)
                        ) : (
                          <span className="text-text-muted">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <Badge tone={STATUS_TONE[exp.status] || "neutral"}>
                          {exp.status}
                        </Badge>
                      </td>
                      <td className="whitespace-nowrap px-4 py-3 text-right text-text-muted">
                        {new Date(exp.createdAtMs).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex items-center justify-between border-t border-border bg-background px-4 py-3">
              <span className="font-label-sm text-label-sm text-text-muted">
                Showing {experiments.length} experiment{experiments.length !== 1 ? "s" : ""}
              </span>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}
