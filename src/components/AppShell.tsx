import type { ReactNode } from "react";
import { Sidebar } from "./Sidebar";
import { TopAppBar } from "./TopAppBar";
import { Disclaimer } from "./Disclaimer";
import { useScrollReveal } from "../hooks/useScrollReveal";

/**
 * Persistent application chrome: fixed sidebar + top bar + scrollable canvas +
 * the always-on safety disclaimer pinned to the bottom of the content column.
 */
export function AppShell({
  title,
  showSearch = false,
  children,
}: {
  title?: string;
  showSearch?: boolean;
  children: ReactNode;
}) {
  // Reveal [data-reveal] elements as they scroll into #doclab-main.
  useScrollReveal();

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar />
      <div className="flex h-screen min-w-0 flex-1 flex-col md:ml-[220px]">
        <TopAppBar title={title} showSearch={showSearch} />
        <main id="doclab-main" className="flex-1 overflow-y-auto">
          <div className="page-enter min-h-full">{children}</div>
        </main>
        <Disclaimer />
      </div>
    </div>
  );
}
