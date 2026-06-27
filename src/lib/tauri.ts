import { invoke as tauriInvoke } from "@tauri-apps/api/core";

const BACKEND_UNAVAILABLE =
  "DocLab backend is only available inside the Tauri desktop app. Run `npm run tauri dev` for live datasets, experiments, and training.";

function hasTauriInternals(): boolean {
  if (typeof window === "undefined") return false;
  const maybeWindow = window as Window & { __TAURI_INTERNALS__?: unknown };
  return (
    "__TAURI_INTERNALS__" in maybeWindow &&
    maybeWindow.__TAURI_INTERNALS__ !== undefined
  );
}

export function backendUnavailableMessage(): string {
  return BACKEND_UNAVAILABLE;
}

export async function invoke<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  if (!hasTauriInternals()) {
    throw new Error(BACKEND_UNAVAILABLE);
  }
  return tauriInvoke<T>(command, args);
}
