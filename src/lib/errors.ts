// Maps the worker's closed error-code set (worker/doclab_worker/errors.py) plus
// agent/orchestration failures to calm, human-readable copy. Keeps raw codes
// and messages available for the technical "details" disclosure.

const FRIENDLY: Record<string, string> = {
  dataset_missing:
    "The dataset couldn't be loaded. It may not be cached locally yet — try prefetching it, or check your connection.",
  bad_plan:
    "The training plan was rejected. Try rephrasing your goal or picking a different dataset.",
  train_failed:
    "Training didn't finish. The data or settings may not suit this model — try a different dataset.",
  oom: "The run ran out of memory. Try a smaller dataset or close other apps and retry.",
  unknown: "Something went wrong during the run. You can retry, or pick a different dataset.",
};

export function friendlyError(code?: string | null): string {
  if (code && FRIENDLY[code]) return FRIENDLY[code];
  return FRIENDLY.unknown;
}
