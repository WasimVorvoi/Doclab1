import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

/** Every screen in the M4 shell. Kept as a string union so navigation is typo-proof. */
export type Route =
  | "home"
  | "plan"
  | "training"
  | "results"
  | "experiments"
  | "models"
  | "datasets"
  | "settings";

interface RouterValue {
  route: Route;
  /** Optional payload passed between screens (e.g. the goal text from Home → Plan). */
  params: Record<string, unknown>;
  navigate: (route: Route, params?: Record<string, unknown>) => void;
}

const RouterContext = createContext<RouterValue | null>(null);

export function RouterProvider({ children }: { children: ReactNode }) {
  const [route, setRoute] = useState<Route>("home");
  const [params, setParams] = useState<Record<string, unknown>>({});

  const navigate = useCallback((next: Route, nextParams: Record<string, unknown> = {}) => {
    setParams(nextParams);
    setRoute(next);
    // Keep the scroll position sane when swapping full screens.
    requestAnimationFrame(() => {
      document.getElementById("doclab-main")?.scrollTo({ top: 0 });
    });
  }, []);

  const value = useMemo(
    () => ({ route, params, navigate }),
    [route, params, navigate],
  );

  return (
    <RouterContext.Provider value={value}>
      {children}
    </RouterContext.Provider>
  );
}

export function useRouter(): RouterValue {
  const ctx = useContext(RouterContext);
  if (!ctx) throw new Error("useRouter must be used within <RouterProvider>");
  return ctx;
}
