import type { ComponentType } from "react";
import { RouterProvider, useRouter, type Route } from "./router";
import { Home } from "./screens/Home";
import { Plan } from "./screens/Plan";
import { Training } from "./screens/Training";
import { Results } from "./screens/Results";
import { Experiments } from "./screens/Experiments";
import { Models } from "./screens/Models";
import { Datasets } from "./screens/Datasets";
import { Settings } from "./screens/Settings";

const SCREENS: Record<Route, ComponentType> = {
  home: Home,
  plan: Plan,
  training: Training,
  results: Results,
  experiments: Experiments,
  models: Models,
  datasets: Datasets,
  settings: Settings,
};

function CurrentScreen() {
  const { route } = useRouter();
  const Screen = SCREENS[route];
  // `key` forces a fresh mount per route so entry animations replay.
  return <Screen key={route} />;
}

export default function App() {
  return (
    <RouterProvider>
      <CurrentScreen />
    </RouterProvider>
  );
}
