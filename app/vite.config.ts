import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // MapLibre GL pesa ~1 MB minificado (~290 KB gzip) y ya va en su propio chunk.
  build: { chunkSizeWarningLimit: 1200 },
  // El worker de MapLibre 6 se carga como módulo (new Worker(url, { type: "module" })).
  worker: { format: "es" },
  server: {
    // Si el hot reload no detecta cambios dentro de Docker en macOS, cambiar a true.
    watch: { usePolling: false },
  },
});
