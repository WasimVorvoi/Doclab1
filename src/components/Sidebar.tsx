import { Icon } from "./Icon";
import { useRouter, type Route } from "../router";

interface NavItem {
  route: Route;
  label: string;
  icon: string;
}

const PRIMARY: NavItem[] = [
  { route: "home", label: "New Prototype", icon: "add_box" },
  { route: "experiments", label: "Experiments", icon: "science" },
  { route: "datasets", label: "Datasets", icon: "database" },
  { route: "models", label: "Models", icon: "model_training" },
  { route: "settings", label: "Settings", icon: "settings" },
];

const FOOTER = [
  { label: "Documentation", icon: "description" },
  { label: "Support", icon: "help" },
];

export function Sidebar() {
  const { route, navigate } = useRouter();

  return (
    <aside className="fixed left-0 top-0 bottom-0 z-40 hidden md:flex w-[220px] h-screen flex-col border-r border-border bg-background">
      {/* Brand */}
      <div className="p-5 flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-accent text-accent-on flex items-center justify-center shrink-0 shadow-sm shadow-accent/20">
          <Icon name="biotech" size={20} />
        </div>
        <div className="leading-tight">
          <h1 className="font-headline-md text-headline-md text-primary tracking-tight">
            DocLab
          </h1>
          <p className="font-label-sm text-label-sm text-text-muted">
            Healthcare ML
          </p>
        </div>
      </div>

      {/* Primary CTA */}
      <div className="px-4 pb-4">
        <button
          onClick={() => navigate("home")}
          className="group w-full flex items-center justify-center gap-2 bg-accent text-accent-on rounded-full py-2.5 font-headline-md text-headline-md shadow-sm shadow-accent/20 transition-all duration-300 ease-out hover:bg-accent-hover hover:shadow-md hover:shadow-accent/30 hover:-translate-y-0.5 active:translate-y-0 active:scale-[0.98]"
        >
          <Icon name="add" size={18} className="transition-transform duration-300 group-hover:rotate-90" />
          New Prototype
        </button>
      </div>

      {/* Primary nav */}
      <nav className="flex-1 overflow-y-auto px-3 space-y-1">
        {PRIMARY.map((item) => {
          const active = route === item.route;
          return (
            <button
              key={item.route}
              onClick={() => navigate(item.route)}
              className={`group relative flex w-full items-center gap-3 rounded-full px-3 py-2 text-left transition-all duration-300 ease-out active:scale-[0.98] ${
                active
                  ? "bg-accent/10 text-accent font-semibold"
                  : "text-text-muted hover:translate-x-0.5 hover:bg-surface-container/60 hover:text-text-secondary"
              }`}
            >
              {/* Animated active accent bar */}
              <span
                className={`absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full bg-accent transition-transform duration-300 ease-out ${
                  active ? "scale-y-100" : "scale-y-0"
                }`}
              />
              <Icon
                name={item.icon}
                size={20}
                fill={active}
                className="transition-transform duration-300 ease-out group-hover:scale-110"
              />
              <span className="font-body-md text-body-md">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Footer nav */}
      <div className="mt-auto p-3 border-t border-border space-y-1">
        {FOOTER.map((item) => (
          <a
            key={item.label}
            className="flex items-center gap-3 px-3 py-2 rounded-full text-text-muted hover:bg-surface-container hover:translate-x-0.5 transition-all duration-300 ease-out cursor-pointer"
          >
            <Icon name={item.icon} size={20} />
            <span className="font-label-sm text-label-sm">{item.label}</span>
          </a>
        ))}
      </div>
    </aside>
  );
}
