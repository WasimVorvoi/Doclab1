// TypeScript types for Tauri command responses
// Matches Rust structs in src-tauri/src/experiments.rs and src-tauri/src/marketplace.rs
// Note: Rust uses snake_case, frontend uses camelCase (serde renames via #[serde(rename_all = "camelCase")])

export interface Dataset {
  id: string;
  name: string;
  hfId: string;
  revision: string;
  dataType: string;
  taskTypes: string[];
  labelColumn: string;
  category: string;
  description: string;
  modality: string;
  license: string;
  size: string;
  limitations: string;
}

export interface WorkerPlan {
  schema_version: number;
  dataset_id: string;
  hf_id: string;
  revision: string;
  label_column: string;
  model_type: string;
  framework: string;
  device: string;
  modality: "tabular" | "image" | "text";
  seed: number;
  split: number[];
  primary_metric: string;
  goal_text?: string;
  summary?: string;
  text_column?: string;
  local_csv?: string;
}

export interface PlanPreview {
  goalText: string;
  dataset: Dataset;
  plan: WorkerPlan;
  summary: string;
}

export interface ExperimentSummary {
  id: string;
  createdAtMs: number;
  updatedAtMs: number;
  status: string;
  goalText: string;
  datasetId: string;
  primaryMetric: string | null;
  metricValue: number | null;
  baselineMetric: number | null;
  isBest: boolean;
  checkpointPath: string | null;
}

export interface ExperimentDetail extends ExperimentSummary {
  modelType: string | null;
  framework: string | null;
  device: string | null;
  planPath: string;
  metricsPath: string | null;
  errorPath: string | null;
  modelCardPath: string | null;
  modelCardContent: string | null;
  workerStdout: string | null;
  workerStderr: string | null;
  errorCode: string | null;
  errorMessage: string | null;
  metrics: unknown | null;
  error: unknown | null;
}

export interface PredictInput {
  inputType: "image_path" | "text" | "tabular_json";
  value: string;
}

export interface PredictionResult {
  modality: string;
  prediction: string;
  confidence: number | null;
  detail: string;
  warning: string;
}

export interface AgentStatus {
  mode: "rules" | "hybrid" | "llm";
  llmConfigured: boolean;
  provider?: "openai" | "anthropic";
}
