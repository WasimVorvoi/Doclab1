import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { invoke } from "../lib/tauri";
import { AppShell } from "../components/AppShell";
import { Icon } from "../components/Icon";
import { Badge } from "../components/Badge";
import { useRouter } from "../router";
import type { ExperimentSummary } from "../types/tauri";

function formatMetric(exp: ExperimentSummary): string {
  if (exp.metricValue === null) return "Pending";
  if (exp.primaryMetric === "rouge_l") return exp.metricValue.toFixed(2);
  return `${(exp.metricValue * 100).toFixed(1)}%`;
}

function metricLabel(metric: string | null): string {
  if (metric === "rouge_l") return "ROUGE-L";
  if (metric === "accuracy") return "Accuracy";
  return metric?.replace(/_/g, " ") || "Metric";
}

export function Models() {
  const { navigate } = useRouter();
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    invoke<ExperimentSummary[]>("list_experiments")
      .then(setExperiments)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);

  const completed = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return experiments
      .filter((exp) => exp.status === "complete")
      .filter((exp) => {
        if (!needle) return true;
        return [exp.id, exp.goalText, exp.datasetId, exp.primaryMetric]
          .filter(Boolean)
          .some((value) => String(value).toLowerCase().includes(needle));
      });
  }, [experiments, query]);

  if (loading) {
    return (
      <AppShell title="Models">
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <Icon name="hourglass_empty" size={48} className="mx-auto mb-4 animate-pulse text-primary" />
            <p className="font-headline-md text-headline-md text-text-primary">
              Loading saved prototypes...
            </p>
          </div>
        </div>
      </AppShell>
    );
  }

  if (error) {
    return (
      <AppShell title="Models">
        <div className="mx-auto max-w-[720px] p-8">
          <div className="rounded-lg border border-warning-text/20 bg-warning-bg p-6">
            <div className="mb-3 flex items-center gap-2 text-warning-text">
              <Icon name="warning" />
              <h3 className="font-headline-md text-headline-md">Saved artifacts unavailable</h3>
            </div>
            <p className="font-body-md text-text-primary">{error}</p>
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell title="Models">
      <div className="mx-auto max-w-[1080px] space-y-8 p-8">
        <section>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
            <div>
              <h3 className="font-headline-lg text-headline-lg text-text-primary">
                Saved prototype artifacts
              </h3>
              <p className="mt-1 font-body-md text-text-muted">
                Completed local runs with metrics and model cards.
              </p>
            </div>
            <div className="relative">
              <Icon
                name="search"
                size={18}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted"
              />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search prototypes..."
                className="w-64 rounded-lg border border-border bg-surface py-1.5 pl-9 pr-4 font-body-md text-body-md transition-colors focus:border-outline-variant focus:outline-none"
              />
            </div>
          </div>

          {completed.length === 0 ? (
            <div className="rounded-lg border border-border bg-surface p-12 text-center">
              <Icon name="model_training" size={48} className="mx-auto mb-4 text-text-muted" />
              <h3 className="mb-2 font-headline-md text-headline-md text-text-primary">
                No completed prototypes yet
              </h3>
              <p className="mb-4 font-body-md text-text-muted">
                Complete a local run to see its saved artifact here.
              </p>
              <button
                onClick={() => navigate("home")}
                className="rounded bg-primary px-4 py-2 font-headline-md text-headline-md text-on-primary shadow-sm transition-colors hover:bg-inverse-surface"
              >
                Start Prototype
              </button>
            </div>
          ) : (
            <div className="stagger grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              {completed.map((exp, i) => (
                <button
                  key={exp.id}
                  style={{ "--i": i + 1 } as CSSProperties}
                  onClick={() => navigate("results", { experimentId: exp.id })}
                  className="flex flex-col rounded-xl border border-border bg-surface p-5 text-left transition-all hover:-translate-y-0.5 hover:border-outline-variant hover:shadow-sm"
                >
                  <div className="mb-3 flex items-start justify-between gap-3">
                    <div>
                      <h4 className="font-headline-md text-headline-md text-primary">
                        {exp.id}
                      </h4>
                      <p className="mt-0.5 font-label-sm text-label-sm text-text-muted">
                        {exp.datasetId}
                      </p>
                    </div>
                    <Badge tone={exp.isBest ? "success" : "neutral"}>
                      {exp.isBest ? "Best" : "Saved"}
                    </Badge>
                  </div>
                  <p className="mb-4 flex-1 font-body-md text-body-md text-text-secondary">
                    {exp.goalText}
                  </p>
                  <div className="mt-auto flex items-center gap-4 border-t border-border pt-4">
                    <div className="flex-1">
                      <div className="mb-1 font-label-sm text-label-sm text-text-muted">
                        {metricLabel(exp.primaryMetric)}
                      </div>
                      <div className="font-code-sm text-code-sm text-primary">
                        {formatMetric(exp)}
                      </div>
                    </div>
                    {exp.checkpointPath && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate("results", { experimentId: exp.id, focusTry: "true" });
                        }}
                        className="rounded border border-border bg-surface px-3 py-1.5 font-label-sm text-label-sm text-text-primary transition-colors hover:bg-surface-muted"
                      >
                        Try
                      </button>
                    )}
                    <span className="flex items-center gap-1 font-body-md text-body-md text-primary">
                      Open card <Icon name="arrow_forward" size={16} />
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
    </AppShell>
  );
}
