import { useEffect, useState, useRef } from "react";
import type { CSSProperties } from "react";
import { invoke } from "../lib/tauri";
import { openPath } from "@tauri-apps/plugin-opener";
import { open } from "@tauri-apps/plugin-dialog";
import ReactMarkdown from "react-markdown";
import { AppShell } from "../components/AppShell";
import { Icon } from "../components/Icon";
import { Badge } from "../components/Badge";
import { useRouter } from "../router";
import { friendlyError } from "../lib/errors";
import { useCountUp } from "../hooks/useCountUp";
import type { ExperimentDetail, PredictInput, PredictionResult } from "../types/tauri";

function modelFamilyLabel(modelType?: string | null): string {
  if (modelType === "cnn") return "Medical image classifier";
  if (modelType === "lora_t5_small") return "Medical text summarizer";
  if (modelType === "xgboost" || modelType === "logistic_regression") {
    return "Structured-data predictor";
  }
  return modelType || "Unknown";
}

/** Animated metric value: counts up from 0 on mount. */
function CountValue({
  target,
  decimals = 0,
  suffix = "",
  delay = 0,
}: {
  target: number;
  decimals?: number;
  suffix?: string;
  delay?: number;
}) {
  const v = useCountUp(target, { decimals, delay });
  return (
    <span>
      {v}
      {suffix}
    </span>
  );
}

export function Results() {
  const { params, navigate } = useRouter();
  const experimentId = params.experimentId as string;
  const focusTry = params.focusTry as string | undefined;
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelCardError, setModelCardError] = useState<string | null>(null);

  // Try prototype state
  const [selectedImagePath, setSelectedImagePath] = useState<string | null>(null);
  const [textInput, setTextInput] = useState("");
  const [tabularInput, setTabularInput] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);

  const tryRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!experimentId) {
      navigate("home");
      return;
    }

    invoke<ExperimentDetail>("get_experiment", { id: experimentId })
      .then(setDetail)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [experimentId, navigate]);

  useEffect(() => {
    if (focusTry === "true" && tryRef.current && detail) {
      setTimeout(() => {
        tryRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    }
  }, [focusTry, detail]);

  if (loading) {
    return (
      <AppShell title="Results">
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <Icon name="hourglass_empty" size={48} className="mx-auto mb-4 animate-pulse text-primary" />
            <p className="font-headline-md text-headline-md text-text-primary">
              Loading results...
            </p>
          </div>
        </div>
      </AppShell>
    );
  }

  if (!detail) {
    return (
      <AppShell title="Results">
        <div className="mx-auto max-w-[720px] p-8">
          <div className={`rounded-lg border p-6 ${
            error
              ? "border-warning-text/20 bg-warning-bg"
              : "border-error bg-error-bg"
          }`}>
            <p className="font-body-md text-text-primary">
              {error ? `Could not load experiment: ${error}` : "Experiment not found."}
            </p>
            <button
              onClick={() => navigate("experiments")}
              className="mt-4 rounded border border-border bg-surface px-4 py-2 font-body-md text-text-primary transition-colors hover:bg-surface-muted"
            >
              View All Experiments
            </button>
          </div>
        </div>
      </AppShell>
    );
  }

  async function openModelCard() {
    if (!detail?.modelCardPath) return;
    setModelCardError(null);
    try {
      await openPath(detail.modelCardPath);
    } catch (e) {
      setModelCardError(e instanceof Error ? e.message : String(e));
    }
  }

  async function selectImage() {
    try {
      const selected = await open({
        multiple: false,
        filters: [
          {
            name: "Images",
            extensions: ["png", "jpg", "jpeg", "bmp", "gif"],
          },
        ],
      });
      if (selected && typeof selected === "string") {
        setSelectedImagePath(selected);
        setPrediction(null);
        setPredictError(null);
      }
    } catch (e) {
      setPredictError(e instanceof Error ? e.message : String(e));
    }
  }

  async function runPrediction() {
    if (!detail || !experimentId) return;

    const modality = detail.modelType === "cnn" ? "image" : detail.modelType === "lora_t5_small" ? "text" : "tabular";

    let input: PredictInput;
    if (modality === "image") {
      if (!selectedImagePath) {
        setPredictError("Please select an image first");
        return;
      }
      input = { inputType: "image_path", value: selectedImagePath };
    } else if (modality === "text") {
      if (!textInput.trim()) {
        setPredictError("Please enter text to summarize");
        return;
      }
      input = { inputType: "text", value: textInput };
    } else if (modality === "tabular") {
      if (!tabularInput.trim()) {
        setPredictError("Please enter JSON input");
        return;
      }
      input = { inputType: "tabular_json", value: tabularInput };
    } else {
      setPredictError("Unknown modality");
      return;
    }

    setPredicting(true);
    setPredictError(null);
    setPrediction(null);

    try {
      const result = await invoke<PredictionResult>("run_predict", {
        experimentId,
        input,
      });
      setPrediction(result);
    } catch (e) {
      setPredictError(friendlyError(e instanceof Error ? e.message : String(e)));
    } finally {
      setPredicting(false);
    }
  }

  const metricValue = detail.metricValue || 0;
  const baselineValue = detail.baselineMetric || 0;
  const isSimilarityMetric = detail.primaryMetric === "rouge_l";
  const sanityPassed = metricValue > baselineValue;
  const metricPercent = (metricValue * 100).toFixed(1);
  const baselinePercent = (baselineValue * 100).toFixed(1);
  const delta = (metricValue - baselineValue) * 100;
  const deltaStr = delta > 0 ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`;
  const metricLabel = isSimilarityMetric ? "ROUGE-L" : "Accuracy";
  const metricTarget = isSimilarityMetric ? metricValue : Number(metricPercent);

  return (
    <AppShell title="Results">
      <div className="mx-auto max-w-[1080px] px-8 py-6 pb-16">
        {/* Header */}
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div className="animate-fade-in-up">
            <div className="mb-2 flex items-center gap-2">
              <Badge tone={detail.status === "complete" ? "success" : "error"}>
                {detail.status === "complete" ? "Complete" : "Failed"}
              </Badge>
              <span className="flex items-center gap-1 font-label-sm text-label-sm text-text-muted">
                <Icon name="schedule" size={14} /> {new Date(detail.updatedAtMs).toLocaleString()}
              </span>
            </div>
            <h2 className="font-headline-lg text-headline-lg text-text-primary">
              Prototype {detail.status === "complete" ? "complete" : "failed"}
            </h2>
            <p className="mt-1 text-text-secondary">
              {detail.goalText}
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => navigate("experiments")}
              className="rounded border border-border-strong bg-surface px-4 py-2 font-headline-md text-headline-md text-text-primary transition-colors hover:bg-surface-muted"
            >
              View experiment
            </button>
            {detail.modelCardPath && (
              <button
                onClick={openModelCard}
                className="rounded bg-primary px-4 py-2 font-headline-md text-headline-md text-on-primary shadow-sm transition-colors hover:bg-inverse-surface"
              >
                Open model card
              </button>
            )}
          </div>
        </div>

        {detail.status === "complete" && (
          <>
            {/* Summary strip */}
            <div className="stagger mb-6 flex flex-wrap gap-x-12 gap-y-4 border-b border-border pb-6">
              <div className="flex flex-col gap-0.5" style={{ "--i": 1 } as CSSProperties}>
                <div className="font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  Model family
                </div>
                <div className="text-headline-md text-text-primary font-headline-md">
                  {modelFamilyLabel(detail.modelType)}
                </div>
              </div>
              <div className="flex flex-col gap-0.5" style={{ "--i": 2 } as CSSProperties}>
                <div className="font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  {metricLabel}
                </div>
                <div className="flex items-baseline gap-2 text-headline-md text-text-primary font-headline-md">
                  <CountValue
                    target={metricTarget}
                    decimals={isSimilarityMetric ? 2 : 1}
                    suffix={isSimilarityMetric ? "" : "%"}
                    delay={250}
                  />
                  {!isSimilarityMetric && (
                    <span className="font-label-sm text-success-text">{deltaStr} vs baseline</span>
                  )}
                </div>
              </div>
              <div className="flex flex-col gap-0.5" style={{ "--i": 3 } as CSSProperties}>
                <div className="font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  Status
                </div>
                <div className="flex items-baseline gap-2 text-headline-md text-text-primary font-headline-md">
                  Saved
                  <Icon
                    name="check_circle"
                    size={18}
                    className="pop text-success-text"
                    style={{ animationDelay: "0.5s" } as CSSProperties}
                  />
                </div>
              </div>
              <div className="flex flex-col gap-0.5" style={{ "--i": 4 } as CSSProperties}>
                <div className="font-label-sm text-label-sm uppercase tracking-wider text-text-muted">
                  Experiment ID
                </div>
                <div className="font-code-sm text-headline-md text-text-primary">
                  {detail.id}
                </div>
              </div>
            </div>

            {/* Sanity message */}
            <div className={`mb-6 flex items-center gap-2 rounded-lg border px-4 py-3 ${
              isSimilarityMetric
                ? "border-border bg-surface text-text-secondary"
                : sanityPassed
                  ? "border-success-text/20 bg-success-bg text-success-text"
                  : "border-warning-text/20 bg-warning-bg text-warning-text"
            }`}>
              <Icon name={isSimilarityMetric ? "summarize" : sanityPassed ? "verified" : "warning"} size={18} />
              <p className="font-body-md text-sm">
                {isSimilarityMetric
                  ? `ROUGE-L is a reference-summary similarity score (${metricValue.toFixed(2)} on a 0–1 scale). Review the fixed examples in the model card qualitatively.`
                  : sanityPassed
                  ? `Sanity check passed — the model scores above the majority-class baseline (${metricPercent}% vs. ${baselinePercent}%), so it is learning real signal.`
                  : `Warning: model performance (${metricPercent}%) is close to or below the baseline (${baselinePercent}%). The model may not be learning signal.`
                }
              </p>
            </div>

            {/* Model card */}
            <div className="overflow-hidden rounded-xl border border-border">
              <div className="flex items-center justify-between border-b border-border px-5 py-4">
                <div className="flex items-center gap-2">
                  <Icon name="article" className="text-text-secondary" />
                  <h3 className="font-headline-lg text-headline-lg text-text-primary">
                    Model card
                  </h3>
                </div>
                {detail.modelCardPath && (
                  <button
                    onClick={openModelCard}
                    className="flex items-center gap-1 font-label-sm text-label-sm text-text-muted underline underline-offset-4 transition-colors hover:text-primary"
                  >
                    <Icon name="open_in_new" size={16} /> Open markdown
                  </button>
                )}
              </div>

              {modelCardError && (
                <div className="border-b border-error bg-error-bg px-5 py-3 font-label-sm text-label-sm text-error-text">
                  Could not open model card: {modelCardError}
                </div>
              )}

              {detail.modelCardContent ? (
                <div className="prose prose-sm max-w-none p-6">
                  <ReactMarkdown>{detail.modelCardContent}</ReactMarkdown>
                </div>
              ) : (
                <div className="p-6">
                  <p className="font-body-md text-text-muted">
                    Model card not available for this experiment.
                  </p>
                </div>
              )}
            </div>

            {/* Try prototype section */}
            {detail.checkpointPath ? (
              <div ref={tryRef} className="mt-6 overflow-hidden rounded-xl border border-border">
                <div className="flex items-center justify-between border-b border-border bg-surface px-5 py-4">
                  <div className="flex items-center gap-2">
                    <Icon name="science" className="text-text-secondary" />
                    <h3 className="font-headline-lg text-headline-lg text-text-primary">
                      Try this prototype
                    </h3>
                  </div>
                </div>

                <div className="p-6">
                  <div className="mb-4 rounded-lg border border-warning-text/20 bg-warning-bg px-4 py-3">
                    <p className="font-label-sm text-label-sm text-warning-text">
                      <strong>Research only</strong> — Use only de-identified, synthetic, or public test inputs. Not for diagnosis or treatment.
                    </p>
                  </div>

                  {detail.modelType === "cnn" && (
                    <div className="space-y-4">
                      <div>
                        <label className="mb-2 block font-label-md text-label-md text-text-primary">
                          Select test image
                        </label>
                        <div className="flex items-center gap-3">
                          <button
                            onClick={selectImage}
                            disabled={predicting}
                            className="rounded border border-border bg-surface px-4 py-2 font-body-md text-text-primary transition-colors hover:bg-surface-muted disabled:opacity-50"
                          >
                            Choose image...
                          </button>
                          {selectedImagePath && (
                            <span className="font-code-sm text-code-sm text-text-secondary truncate max-w-md">
                              {selectedImagePath.split("/").pop()}
                            </span>
                          )}
                        </div>
                      </div>

                      <button
                        onClick={runPrediction}
                        disabled={!selectedImagePath || predicting}
                        className="rounded bg-primary px-6 py-2 font-headline-md text-headline-md text-on-primary shadow-sm transition-colors hover:bg-inverse-surface disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {predicting ? "Running..." : "Run prediction"}
                      </button>
                    </div>
                  )}

                  {detail.modelType === "lora_t5_small" && (
                    <div className="space-y-4">
                      <div>
                        <label className="mb-2 block font-label-md text-label-md text-text-primary">
                          Enter text to summarize
                        </label>
                        <textarea
                          value={textInput}
                          onChange={(e) => setTextInput(e.target.value)}
                          disabled={predicting}
                          placeholder="Paste medical education text here..."
                          className="w-full rounded border border-border bg-surface px-4 py-3 font-body-md text-text-primary placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
                          rows={6}
                        />
                      </div>

                      <button
                        onClick={runPrediction}
                        disabled={!textInput.trim() || predicting}
                        className="rounded bg-primary px-6 py-2 font-headline-md text-headline-md text-on-primary shadow-sm transition-colors hover:bg-inverse-surface disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {predicting ? "Generating..." : "Generate summary"}
                      </button>
                    </div>
                  )}

                  {(detail.modelType === "xgboost" || detail.modelType === "logistic_regression") && (
                    <div className="space-y-4">
                      <div>
                        <label className="mb-2 block font-label-md text-label-md text-text-primary">
                          Enter feature values as JSON
                        </label>
                        <p className="mb-2 font-label-sm text-label-sm text-text-muted">
                          Paste a JSON object with feature names and values matching the training dataset schema.
                          Missing keys default to 0; use numeric/boolean values.
                        </p>
                        <p className="mb-3 font-code-sm text-code-sm text-text-secondary">
                          Example: {`{"age": 65, "glucose": 120, "readmitted": 0}`}
                        </p>
                        <textarea
                          value={tabularInput}
                          onChange={(e) => setTabularInput(e.target.value)}
                          disabled={predicting}
                          placeholder='{"age": 50, "glucose": 100}'
                          className="w-full rounded border border-border bg-surface px-4 py-3 font-code-sm text-code-sm text-text-primary placeholder:text-text-muted focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
                          rows={4}
                        />
                      </div>

                      <button
                        onClick={runPrediction}
                        disabled={!tabularInput.trim() || predicting}
                        className="rounded bg-primary px-6 py-2 font-headline-md text-headline-md text-on-primary shadow-sm transition-colors hover:bg-inverse-surface disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {predicting ? "Running..." : "Run prediction"}
                      </button>
                    </div>
                  )}

                  {predictError && (
                    <div className="mt-4 rounded-lg border border-error bg-error-bg px-4 py-3">
                      <p className="font-body-md text-error-text">{predictError}</p>
                    </div>
                  )}

                  {prediction && (
                    <div className="mt-4 rounded-lg border border-success-text/20 bg-success-bg p-4">
                      <div className="mb-2 flex items-center gap-2">
                        <Icon name="check_circle" className="text-success-text" />
                        <h4 className="font-headline-md text-headline-md text-success-text">
                          Prediction result
                        </h4>
                      </div>
                      <div className="space-y-2">
                        <p className="font-body-md text-text-primary">
                          <strong>Prediction:</strong> {prediction.prediction}
                        </p>
                        {prediction.confidence !== null && (
                          <p className="font-body-md text-text-primary">
                            <strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(1)}%
                          </p>
                        )}
                        <p className="font-label-sm text-label-sm text-text-muted">
                          {prediction.detail}
                        </p>
                        <p className="font-label-sm text-label-sm text-warning-text">
                          ⚠️ {prediction.warning}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="mt-6 rounded-lg border border-border bg-surface-muted px-6 py-4">
                <p className="font-body-md text-text-muted">
                  Retrain this prototype to enable local testing. Older experiments don't have saved checkpoints.
                </p>
              </div>
            )}
          </>
        )}

        {detail.status === "failed" && (
          <div className="rounded-lg border border-error bg-error-bg p-6">
            <div className="flex items-center gap-2 mb-4">
              <Icon name="error" className="text-error" />
              <h3 className="font-headline-md text-headline-md text-error">
                Experiment Failed
              </h3>
            </div>
            <p className="font-body-md text-text-primary mb-4">
              {friendlyError(detail.errorCode)}
            </p>
            <details className="mt-2">
              <summary className="cursor-pointer font-label-sm text-label-sm text-text-muted">
                Technical details
              </summary>
              <p className="mt-2 font-body-md text-text-secondary">
                <strong>Error code:</strong> {detail.errorCode || "unknown"}
              </p>
              <p className="mt-1 font-body-md text-text-secondary">
                {detail.errorMessage || "Worker failed without error details"}
              </p>
              {detail.workerStderr && (
                <pre className="mt-2 rounded bg-log-bg p-3 font-code-sm text-code-sm text-log-text overflow-x-auto">
                  {detail.workerStderr}
                </pre>
              )}
            </details>
          </div>
        )}
      </div>
    </AppShell>
  );
}
