import type { ReactNode } from "react";

export type BadgeTone =
  | "success"
  | "warning"
  | "error"
  | "neutral"
  | "running";

const TONES: Record<BadgeTone, string> = {
  success: "bg-success-bg text-success-text border-success-text/20",
  warning: "bg-warning-bg text-warning-text border-warning-text/20",
  error: "bg-error-bg text-error-text border-error/20",
  neutral: "bg-surface-container text-text-secondary border-border",
  running: "bg-surface-container text-text-secondary border-border",
};

export function Badge({
  tone = "neutral",
  children,
  className = "",
}: {
  tone?: BadgeTone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-medium border ${TONES[tone]} ${className}`}
    >
      {tone === "running" && (
        <span className="w-1.5 h-1.5 rounded-full bg-outline animate-pulse" />
      )}
      {children}
    </span>
  );
}
