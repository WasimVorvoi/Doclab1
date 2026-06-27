// Lightweight dataset profiling for the plan artifacts.
// Full profiling happens in the worker; this keeps the staged UI deterministic.

import type { Dataset } from "../types/tauri";

export interface DataProfile {
  schema: string[];
  label_column: string;
  row_count: number;
  missing_percent: number;
}

export async function profileDataset(dataset: Dataset): Promise<DataProfile> {
  const dataType = dataset.dataType.toLowerCase();

  return {
    schema: schemaFor(dataset),
    label_column: dataset.labelColumn,
    row_count: rowCountHint(dataset.size),
    missing_percent: dataType === "tabular" ? 2.5 : 0,
  };
}

function schemaFor(dataset: Dataset): string[] {
  const dataType = dataset.dataType.toLowerCase();
  if (dataType === "image") return ["image", dataset.labelColumn];
  if (dataType === "text") return ["Text", dataset.labelColumn];
  return ["clinical_features", "encounter_features", dataset.labelColumn];
}

function rowCountHint(size: string): number {
  const compact = size.match(/~?(\d+(?:\.\d+)?)\s*k/i);
  if (compact) return Math.round(Number(compact[1]) * 1000);

  const firstNumber = size.match(/\d+/);
  return firstNumber ? Number(firstNumber[0]) : 0;
}
