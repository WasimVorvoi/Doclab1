// Agent orchestration: parse → select → profile → create plan
// Entry point for the M5 agent layer

import { invoke } from "../lib/tauri";
import { parseGoal, type Intent } from "./parser";
import { extractKeywords, type DatasetSelection } from "./selector";
import { profileDataset, type DataProfile } from "./profiler";
import type { PlanPreview, Dataset, AgentStatus } from "../types/tauri";

export interface AgentResult {
  intent: Intent;
  selection: DatasetSelection;
  profile: DataProfile;
  planPreview: PlanPreview;
}

export async function runAgent(goalText: string): Promise<AgentResult> {
  // 1. Parse intent (hybrid: try LLM if ambiguous, fallback to rules)
  const intent = await parseIntentHybrid(goalText);
  console.log("[Agent] intent.json:", intent);

  // 2. Query marketplace (unchanged - always Rust)
  const candidates = await invoke<Dataset[]>("query_datasets", {
    keyword: extractKeywords(intent.goal_text)[0] || null,
    dataType: intent.modality,
    taskType: intent.task_type,
  });

  // 3. Select dataset (hybrid: try LLM if multiple candidates)
  const selection = await selectDatasetHybrid(intent, candidates);
  console.log("[Agent] dataset_selection.json:", selection);

  // 4. Create plan via Tauri create_plan command
  const planPreview = await invoke<PlanPreview>("create_plan", {
    goalText,
    datasetId: selection.dataset_id,
  });
  console.log("[Agent] plan created:", planPreview);

  // 5. Profile selected dataset metadata for the fixed artifact contract
  const profile = await profileDataset(planPreview.dataset);
  console.log("[Agent] data_profile.json:", profile);

  return { intent, selection, profile, planPreview };
}

async function parseIntentHybrid(goalText: string): Promise<Intent> {
  const status = await invoke<AgentStatus>("get_agent_status");

  // Always try rules first in hybrid mode
  const rulesIntent = parseGoal(goalText);

  if (status.mode === "rules" || !status.llmConfigured) {
    return rulesIntent;
  }

  // In LLM mode, always use LLM
  if (status.mode === "llm") {
    try {
      const llmIntent = await invoke<Intent>("agent_parse_intent", { goalText });
      return { ...llmIntent, llm_used: true };
    } catch (err) {
      console.warn("[Agent] LLM parse failed, falling back to rules:", err);
      return { ...rulesIntent, llm_fallback: true };
    }
  }

  // Hybrid mode: use LLM only if ambiguous
  const isAmbiguous = goalText.length > 120 || hasConflictingKeywords(goalText);
  if (isAmbiguous) {
    try {
      const llmIntent = await invoke<Intent>("agent_parse_intent", { goalText });
      return { ...llmIntent, llm_used: true };
    } catch (err) {
      console.warn("[Agent] LLM parse failed, using rules:", err);
      return { ...rulesIntent, llm_fallback: true };
    }
  }

  return rulesIntent;
}

async function selectDatasetHybrid(
  intent: Intent,
  candidates: Dataset[]
): Promise<DatasetSelection> {
  if (candidates.length === 0) {
    throw new Error(
      `No ${intent.modality} datasets found for task "${intent.task_type}". Try a different goal or check the curated marketplace.`
    );
  }

  const status = await invoke<AgentStatus>("get_agent_status");

  // Single candidate: no need for LLM
  if (candidates.length === 1) {
    const dataset = candidates[0];
    return {
      dataset_id: dataset.id,
      dataset_name: dataset.name,
      rationale: `Only ${intent.modality} dataset for ${intent.task_type}: ${dataset.description}`,
    };
  }

  // Rules mode or no LLM: use existing ranking
  if (status.mode === "rules" || !status.llmConfigured) {
    const best = candidates[0];
    return {
      dataset_id: best.id,
      dataset_name: best.name,
      rationale: `Top-ranked match: ${best.description}`,
    };
  }

  // LLM or hybrid mode with multiple candidates: ask LLM to pick
  try {
    const llmSelection = await invoke<{ dataset_id: string; rationale: string }>(
      "agent_pick_dataset",
      { goalText: intent.goal_text, candidates }
    );
    const dataset = candidates.find((d) => d.id === llmSelection.dataset_id)!;
    return {
      dataset_id: llmSelection.dataset_id,
      dataset_name: dataset.name,
      rationale: llmSelection.rationale,
      llm_used: true,
    };
  } catch (err) {
    console.warn("[Agent] LLM selection failed, using top-ranked:", err);
    const best = candidates[0];
    return {
      dataset_id: best.id,
      dataset_name: best.name,
      rationale: `Top-ranked match (LLM fallback): ${best.description}`,
      llm_fallback: true,
    };
  }
}

function hasConflictingKeywords(goal: string): boolean {
  const lower = goal.toLowerCase();
  const hasImage = /image|x-ray|xray|scan|photo/.test(lower);
  const hasText = /text|summarize|note|document/.test(lower);
  const hasTabular = /predict|readmission|diabetes|heart/.test(lower);
  return [hasImage, hasText, hasTabular].filter(Boolean).length > 1;
}
