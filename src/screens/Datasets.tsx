import { useEffect, useState, type CSSProperties } from "react";
import { invoke } from "@tauri-apps/api/core";
import { AppShell } from "../components/AppShell";
import { Icon } from "../components/Icon";
import { useRouter } from "../router";
import type { Dataset } from "../types/tauri";

/** Each data modality gets its own hue; returned as a `--color-accent`
 *  override so the card's hairline, chip, and hover all re-tint to match. */
function modalityAccent(dataType: string): string {
  switch (dataType.toLowerCase()) {
    case "image":
      return "#4f46e5"; // indigo
    case "text":
      return "#b45309"; // amber
    default:
      return "#0f766e"; // teal — tabular
  }
}

export function Datasets() {
  const { navigate, params } = useRouter();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  // Picking a dataset pins it on the Home composer; carry any typed goal back.
  function selectDataset(dataset: Dataset) {
    navigate("home", {
      attachedDataset: dataset,
      prefillGoal: (params.goal as string) ?? "",
      // Preserve any uploaded file so attaching a dataset doesn't drop it.
      upload: (params.upload as string) ?? null,
    });
  }

  useEffect(() => {
    invoke<Dataset[]>("query_datasets", {
      keyword: null,
      dataType: null,
      taskType: null,
    })
      .then(setDatasets)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <AppShell title="Datasets">
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <Icon name="hourglass_empty" size={48} className="mx-auto mb-4 animate-pulse text-accent" />
            <p className="font-headline-md text-headline-md text-text-primary">
              Loading datasets...
            </p>
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell title="Datasets">
      <div className="mx-auto max-w-[1080px] p-8">
        <div className="mb-6">
          <div className="flex items-center gap-2.5">
            <span className="h-7 w-1 rounded-full bg-accent" />
            <h1 className="font-headline-lg text-headline-lg text-text-primary">
              Curated Datasets
            </h1>
          </div>
          <p className="mt-1 font-body-md text-text-muted">
            Approved public and de-identified datasets for prototyping. The agent
            selects from this registry only.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {datasets.map((dataset, i) => (
            <div
              key={dataset.id}
              role="button"
              tabIndex={0}
              data-reveal
              style={
                {
                  "--reveal-delay": `${(i % 2) * 80}ms`,
                  "--color-accent": modalityAccent(dataset.dataType),
                } as CSSProperties
              }
              onClick={() => selectDataset(dataset)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  selectDataset(dataset);
                }
              }}
              className="card-accent group relative cursor-pointer overflow-hidden rounded-lg border border-border bg-surface p-5 transition-all hover:-translate-y-0.5 hover:border-accent/40 hover:shadow-md hover:shadow-black/5"
            >
              <div className="mb-3 flex items-start justify-between">
                <h3 className="font-headline-md text-headline-md text-text-primary">
                  {dataset.name}
                </h3>
                <span className="inline-flex shrink-0 items-center rounded-full border border-accent/25 bg-accent/10 px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-accent">
                  {dataset.dataType}
                </span>
              </div>

              <div className="mb-3 rounded border border-surface-tint bg-log-bg px-3 py-2">
                <code className="font-code-sm text-code-sm text-log-text">
                  {dataset.hfId}@{dataset.revision.substring(0, 7)}
                </code>
              </div>

              <div className="mb-3 space-y-2 font-label-sm text-label-sm text-text-muted">
                <div className="flex items-center gap-2">
                  <Icon name="category" size={14} />
                  <span>Task: {dataset.taskTypes.join(", ")}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Icon name="storage" size={14} />
                  <span>Size: {dataset.size}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Icon name="gavel" size={14} />
                  <span>License: {dataset.license}</span>
                </div>
              </div>

              <p className="mb-3 font-body-md text-sm text-text-secondary">
                {dataset.description}
              </p>

              {dataset.limitations && (
                <details
                  className="font-label-sm text-label-sm text-text-muted"
                  onClick={(e) => e.stopPropagation()}
                >
                  <summary className="cursor-pointer">Limitations</summary>
                  <p className="mt-2 text-text-secondary">{dataset.limitations}</p>
                </details>
              )}

              <div className="mt-4 flex items-center justify-end gap-1 border-t border-border pt-3 font-label-sm text-label-sm font-semibold text-accent opacity-0 transition-opacity group-hover:opacity-100">
                Use this dataset
                <Icon name="arrow_forward" size={16} />
              </div>
            </div>
          ))}
        </div>

        {datasets.length === 0 && (
          <div className="rounded-lg border border-border bg-surface p-12 text-center">
            <Icon name="dataset" size={48} className="mx-auto mb-4 text-text-muted" />
            <h3 className="mb-2 font-headline-md text-headline-md text-text-primary">
              No datasets found
            </h3>
            <p className="font-body-md text-text-muted">
              The curated marketplace is empty.
            </p>
          </div>
        )}
      </div>
    </AppShell>
  );
}
