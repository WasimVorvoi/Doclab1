import { Icon } from "./Icon";

/** Global, always-visible safety banner. Required by the spec on every screen. */
export function Disclaimer() {
  return (
    <div className="mb-16 flex items-center gap-2 border-t border-warning-text/15 bg-warning-bg px-8 py-2 text-warning-text md:mb-0">
      <Icon name="shield" size={16} />
      <p className="font-label-sm text-label-sm">
        Research &amp; prototyping only — not for clinical care. Curated public
        data only; no PHI, EHR, or private patient uploads.
      </p>
    </div>
  );
}
