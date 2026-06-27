// Rule-based goal parsing for M5
// Extracts task type, modality, and metric hint from goal text

export interface Intent {
  task_type: "predict" | "classify" | "detect" | "summarize" | "generate";
  modality: "tabular" | "image" | "text";
  metric_hint: "accuracy" | "auc" | "rouge" | null;
  goal_text: string;
  llm_used?: boolean;
  llm_fallback?: boolean;
}

export function parseGoal(goal: string): Intent {
  const lower = goal.toLowerCase();

  // Task type detection
  let task_type: Intent["task_type"] = "classify";
  if (lower.includes("predict")) {
    task_type = "predict";
  } else if (lower.includes("summarize")) {
    task_type = "summarize";
  } else if (lower.includes("detect")) {
    task_type = "detect";
  } else if (lower.includes("generate")) {
    task_type = "generate";
  }

  // Modality detection
  let modality: Intent["modality"] = "tabular";
  if (
    lower.includes("image") ||
    lower.includes("x-ray") ||
    lower.includes("xray") ||
    lower.includes("scan") ||
    lower.includes("photo")
  ) {
    modality = "image";
  } else if (
    lower.includes("text") ||
    lower.includes("summarize") ||
    lower.includes("note") ||
    lower.includes("document")
  ) {
    modality = "text";
  }

  // Metric hint based on modality
  let metric_hint: Intent["metric_hint"] = null;
  if (modality === "tabular" || modality === "image") {
    metric_hint = "accuracy";
  } else if (modality === "text") {
    metric_hint = "rouge";
  }

  return {
    task_type,
    modality,
    metric_hint,
    goal_text: goal,
  };
}
