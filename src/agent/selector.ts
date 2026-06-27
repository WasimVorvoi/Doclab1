// Dataset selection from curated marketplace
// Queries Tauri backend and ranks results

import { invoke } from "../lib/tauri";
import type { Dataset } from "../types/tauri";
import type { Intent } from "./parser";

export interface DatasetSelection {
  dataset_id: string;
  dataset_name: string;
  rationale: string;
  llm_used?: boolean;
  llm_fallback?: boolean;
}

export async function selectDataset(intent: Intent): Promise<DatasetSelection> {
  // Extract keywords from goal text
  const keywords = extractKeywords(intent.goal_text);

  // Query curated marketplace via Tauri
  const datasets = await invoke<Dataset[]>("query_datasets", {
    keyword: keywords[0] || null,
    dataType: intent.modality,
    taskType: intent.task_type,
  });

  if (datasets.length === 0) {
    throw new Error(
      `No ${intent.modality} datasets found for task "${intent.task_type}". Try a different goal or check the curated marketplace.`
    );
  }

  // Rust query_datasets already ranks by keyword score; take the best match
  const best = datasets[0];

  return {
    dataset_id: best.id,
    dataset_name: best.name,
    rationale: `Matched "${best.name}" from curated registry: ${best.description}`,
  };
}

export function extractKeywords(goal: string): string[] {
  const lower = goal.toLowerCase();
  const keywords = [
    "readmission",
    "heart",
    "diabetes",
    "chest",
    "pneumonia",
    "medical",
    "text",
    "xray",
    "x-ray",
  ];
  return keywords.filter((k) => lower.includes(k));
}
