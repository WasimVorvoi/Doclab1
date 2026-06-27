import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// Self-hosted fonts, bundled into the app. This is what makes startup fast:
// a local-first desktop app must not block first paint on a Google Fonts CDN
// round-trip (and icons must never fall back to raw ligature text offline).
import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "material-symbols/outlined.css";

// Imported last so its `.material-symbols-outlined` rules win the cascade.
import "./index.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
