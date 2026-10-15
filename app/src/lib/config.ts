// Constantes de la app.
export const APP_NOMBRE = "CITRA";
export const APP_LEMA = "Centro de Inteligencia Territorial, Riesgo y Actividad Económica";

// Fracción de AGEB que define el "top" (mismo criterio que visible_por_conteo en el contrato).
export const FRACCION_TOP = 0.1;

// Encuadre inicial: Ciudad de México [oeste, sur, este, norte].
export const CDMX_BBOX: [number, number, number, number] = [-99.365, 19.05, -98.94, 19.593];

// Basemap oscuro de CARTO: uso no comercial gratis con atribución. Se pide sin API key, así que CARTO
// puede ponerle marca de agua (términos vigentes desde septiembre de 2026).
export const BASEMAP_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

// Hasta dónde se puede mover la vista: la CDMX con un margen de ~3 km. Ver lib/camara.ts.
export const LIMITE_MAPA: [number, number, number, number] = [-99.395, 19.02, -98.91, 19.62];
export const ZOOM_MAX = 16;

// Contorno de la CDMX invertido (todo lo de fuera), del Marco Geoestadístico 2020 simplificado a ~50 m.
export const MASCARA_URL = `${import.meta.env.BASE_URL}cdmx-mascara.geojson`;

// Disposición: el mapa ocupa toda la ventana y el encabezado y el panel flotan encima (px).
export const MARGEN = 12;
export const ALTO_ENCABEZADO = 56;
export const PANEL = {
  inicial: 420,
  min: 340,
  minTabla: 460, // la tabla de prioridades no cabe en menos
  max: 860,
  mapaMin: 380, // ancho mínimo que el panel le deja al mapa
  amplio: 600, // a partir de aquí la tabla de prioridades muestra más columnas
};
