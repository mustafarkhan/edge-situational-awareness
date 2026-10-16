import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Content-Security-Policy for the built app (not the dev server, whose hot reload needs inline scripts).
// Scripts only from this origin; map tiles only from the configured tile host; nothing framed or embedded.
const TILE_HOST = process.env.VITE_TILE_HOST ?? "https://tile.openstreetmap.org";
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'", // MapLibre positions markers and popups with inline styles
  `img-src 'self' data: blob: ${TILE_HOST}`,
  `connect-src 'self' ws: wss: ${TILE_HOST}`,
  "worker-src 'self' blob:", // MapLibre 6 loads its worker from this origin
  "child-src blob:",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join("; ");

const csp = {
  name: "csp",
  apply: "build",
  transformIndexHtml: (html) =>
    html.replace("<head>", `<head>\n    <meta http-equiv="Content-Security-Policy" content="${CSP}" />`),
};

export default defineConfig({ plugins: [react(), csp], server: { host: "127.0.0.1" } });
