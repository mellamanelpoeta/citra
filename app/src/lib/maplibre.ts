// MapLibre 6 ubica su worker con import.meta.url (archivo hermano), lo que no sobrevive al
// empaquetado de Vite. Aquí Vite empaqueta el worker como módulo y le pasamos su URL.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

// Promesa que react-map-gl usa como mapLib: MapLibre sigue cargándose en un chunk aparte.
export const mapLib = import("maplibre-gl").then((m) => {
  m.setWorkerUrl(workerUrl);
  return m;
});
