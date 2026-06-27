import { useState, useEffect, type ReactNode } from "react";
import { invoke } from "../lib/tauri";
import { AppShell } from "../components/AppShell";
import { Icon } from "../components/Icon";
import type { AgentStatus } from "../types/tauri";

interface SettingsState {
  workspaceName: string;
  institution: string;
  preferredDatasets: string[];
  verbosity: "Minimal" | "Standard" | "Debug";
}

const DEFAULT_SETTINGS: SettingsState = {
  workspaceName: "DocLab ML Env",
  institution: "Community Research Lab",
  preferredDatasets: ["Diabetes 130-US", "PadChest (public)"],
  verbosity: "Standard",
};

const DATASET_OPTIONS = [
  "Diabetes 130-US",
  "PadChest (public)",
  "Medical Education Summaries",
];

function loadSettings(): SettingsState {
  try {
    const raw = window.localStorage.getItem("doclab.settings");
    if (!raw) return DEFAULT_SETTINGS;
    const parsed = JSON.parse(raw) as Partial<SettingsState>;
    return {
      workspaceName: parsed.workspaceName || DEFAULT_SETTINGS.workspaceName,
      institution: parsed.institution || DEFAULT_SETTINGS.institution,
      preferredDatasets: parsed.preferredDatasets?.length
        ? parsed.preferredDatasets
        : DEFAULT_SETTINGS.preferredDatasets,
      verbosity: parsed.verbosity || DEFAULT_SETTINGS.verbosity,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function Section({
  icon,
  title,
  children,
}: {
  icon: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-border bg-surface-container-lowest p-6">
      <h3 className="mb-4 flex items-center gap-2 font-headline-md text-headline-md text-primary">
        <Icon name={icon} className="text-text-muted" />
        <span>{title}</span>
      </h3>
      {children}
    </section>
  );
}

function Label({ children, htmlFor }: { children: ReactNode; htmlFor?: string }) {
  return (
    <label
      htmlFor={htmlFor}
      className="block font-label-sm text-label-sm text-text-secondary"
    >
      {children}
    </label>
  );
}

const inputCls =
  "w-full rounded-lg border border-border bg-background px-4 py-2 text-text-primary focus:border-border-strong focus:outline-none transition-colors";

export function Settings() {
  const [savedSettings, setSavedSettings] = useState<SettingsState>(loadSettings);
  const [settings, setSettings] = useState<SettingsState>(loadSettings);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [worker, setWorker] = useState<{
    state: "idle" | "checking" | "ok" | "error";
    text: string;
  }>({ state: "idle", text: "" });
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);

  useEffect(() => {
    invoke<AgentStatus>("get_agent_status")
      .then(setAgentStatus)
      .catch((err) => console.error("Failed to get agent status:", err));
  }, []);

  const dirty = JSON.stringify(settings) !== JSON.stringify(savedSettings);
  const nextDataset = DATASET_OPTIONS.find(
    (dataset) => !settings.preferredDatasets.includes(dataset),
  );

  function updateSettings(patch: Partial<SettingsState>) {
    setSettings((current) => ({ ...current, ...patch }));
    setSaveMessage(null);
  }

  function removeDataset(dataset: string) {
    updateSettings({
      preferredDatasets: settings.preferredDatasets.filter((item) => item !== dataset),
    });
  }

  function addDataset() {
    if (!nextDataset) return;
    updateSettings({
      preferredDatasets: [...settings.preferredDatasets, nextDataset],
    });
  }

  function discardChanges() {
    setSettings(savedSettings);
    setSaveMessage("Changes discarded.");
    window.setTimeout(() => setSaveMessage(null), 1800);
  }

  function saveSettings() {
    window.localStorage.setItem("doclab.settings", JSON.stringify(settings));
    setSavedSettings(settings);
    setSaveMessage("Settings saved.");
    window.setTimeout(() => setSaveMessage(null), 1800);
  }

  async function checkWorker() {
    setWorker({ state: "checking", text: "" });
    try {
      const out = await invoke<string>("worker_healthcheck");
      setWorker({ state: "ok", text: out });
    } catch (e) {
      setWorker({ state: "error", text: String(e) });
    }
  }

  return (
    <AppShell title="Settings">
      <div className="mx-auto w-full max-w-[1080px] space-y-6 p-8">
        {/* Workspace */}
        <Section icon="domain" title="Workspace">
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="workspace-name">Workspace name</Label>
                <input
                  id="workspace-name"
                  className={inputCls}
                  value={settings.workspaceName}
                  onChange={(e) => updateSettings({ workspaceName: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="research-institution">Research institution</Label>
                <input
                  id="research-institution"
                  className={inputCls}
                  value={settings.institution}
                  onChange={(e) => updateSettings({ institution: e.target.value })}
                />
              </div>
            </div>
          </div>
        </Section>

        {/* Agent config */}
        <Section icon="smart_toy" title="Agent configuration">
          <div className="space-y-6">
            <div className="space-y-2">
              <Label>Planning mode</Label>
              <div className="rounded-lg border border-border bg-surface-muted p-3">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-text-primary">
                    {agentStatus?.mode === "rules" && "Rule-based (curated registry)"}
                    {agentStatus?.mode === "hybrid" && "Hybrid (rules + AI-assisted)"}
                    {agentStatus?.mode === "llm" && "AI-powered (LLM-first)"}
                    {!agentStatus && "Loading..."}
                  </span>
                  {agentStatus?.llmConfigured ? (
                    <span className="flex items-center gap-1 text-success-text">
                      <Icon name="check_circle" size={16} />
                      API key configured
                    </span>
                  ) : (
                    <span className="text-text-muted">No API key</span>
                  )}
                </div>
                {agentStatus?.provider && (
                  <p className="mt-2 text-sm text-text-muted">
                    Provider: {agentStatus.provider}
                  </p>
                )}
              </div>
              <p className="text-sm text-text-muted">
                Set via DOCLAB_AGENT_MODE environment variable. AI planning uses your API
                key locally via Rust; training never leaves this device.
              </p>
            </div>
            <div className="space-y-2">
              <Label>Preferred datasets</Label>
              <div className="flex flex-wrap gap-2">
                {settings.preferredDatasets.map((dataset) => (
                  <span
                    key={dataset}
                    className="inline-flex items-center rounded-full border border-border bg-surface-muted px-3 py-1 font-label-sm text-label-sm text-text-secondary"
                  >
                    {dataset}
                    <button
                      type="button"
                      onClick={() => removeDataset(dataset)}
                      className="ml-2 transition-colors hover:text-error-text"
                      aria-label={`Remove ${dataset}`}
                    >
                      <Icon name="close" size={14} />
                    </button>
                  </span>
                ))}
                <button
                  type="button"
                  onClick={addDataset}
                  disabled={!nextDataset}
                  className="inline-flex items-center rounded-full border border-dashed border-border-strong px-3 py-1 font-label-sm text-label-sm text-text-muted transition-colors hover:bg-surface-muted disabled:cursor-not-allowed disabled:opacity-40"
                >
                  <Icon name="add" size={14} className="mr-1" /> Add dataset
                </button>
              </div>
            </div>
            <div className="space-y-2">
              <Label>Log verbosity</Label>
              <div className="flex items-center gap-4">
                {(["Minimal", "Standard", "Debug"] as const).map((verbosity) => (
                  <label key={verbosity} className="flex cursor-pointer items-center gap-2">
                    <input
                      type="radio"
                      name="verbosity"
                      checked={settings.verbosity === verbosity}
                      onChange={() => updateSettings({ verbosity })}
                      className="border-border text-primary focus:ring-primary"
                    />
                    <span className="text-text-primary">{verbosity}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </Section>

        {/* System / worker */}
        <Section icon="dns" title="System & worker">
          <div className="space-y-3">
            <p className="font-body-md text-body-md text-text-secondary">
              Verify the local Python training worker is reachable from the Rust
              backend.
            </p>
            <div className="flex items-center gap-3">
              <button
                onClick={checkWorker}
                disabled={worker.state === "checking"}
                className="rounded-lg bg-primary px-4 py-2 font-label-sm text-label-sm text-on-primary transition-colors hover:bg-inverse-surface disabled:opacity-50"
              >
                {worker.state === "checking" ? "Checking..." : "Check worker"}
              </button>
              {worker.state === "ok" && (
                <span className="flex items-center gap-1 font-label-sm text-label-sm text-success-text">
                  <Icon name="check_circle" size={16} /> Worker reachable
                </span>
              )}
              {worker.state === "error" && (
                <span className="flex items-center gap-1 font-label-sm text-label-sm text-error-text">
                  <Icon name="error" size={16} /> Not reachable
                </span>
              )}
            </div>
            {worker.text && (
              <pre className="max-w-2xl overflow-x-auto whitespace-pre-wrap rounded-lg border border-outline-variant bg-log-bg p-3 font-code-sm text-code-sm text-log-text">
                {worker.text}
              </pre>
            )}
          </div>
        </Section>

        {/* Security */}
        <Section icon="security" title="Security & access">
          <div className="space-y-2">
            <Label>Your role</Label>
            <div className="flex items-center justify-between rounded-lg border border-border bg-surface-muted p-3">
              <div className="flex items-center gap-2">
                <Icon name="admin_panel_settings" className="text-primary" />
                <span className="font-semibold text-text-primary">
                  Administrator
                </span>
              </div>
              <span className="rounded-full border border-success-text/20 bg-success-bg px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-success-text">
                Active
              </span>
            </div>
            <p className="mt-1 font-label-sm text-label-sm text-text-muted">
              Full access to workspace settings, dataset registry, and local
              prototype artifacts. No PHI or private uploads are ever permitted.
            </p>
          </div>
        </Section>

        {/* Actions */}
        <div className="flex flex-wrap items-center justify-end gap-4 border-t border-border pt-6">
          {saveMessage && (
            <span className="mr-auto font-label-sm text-label-sm text-success-text">
              {saveMessage}
            </span>
          )}
          <button
            onClick={discardChanges}
            disabled={!dirty}
            className="rounded-lg border border-border-strong px-6 py-2 font-label-sm text-label-sm text-text-primary transition-colors hover:bg-surface-muted disabled:cursor-not-allowed disabled:opacity-40"
          >
            Discard changes
          </button>
          <button
            onClick={saveSettings}
            disabled={!dirty}
            className="rounded-lg bg-primary px-6 py-2 font-label-sm text-label-sm text-on-primary transition-colors hover:bg-inverse-surface disabled:cursor-not-allowed disabled:opacity-40"
          >
            Save settings
          </button>
        </div>
      </div>
    </AppShell>
  );
}
