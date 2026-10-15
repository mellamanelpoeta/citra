// Única capa de carga de datos de la app. Lee los archivos estáticos de public/data/ que exporta
// el pipeline de Python (esquema en src/types/data.ts); los componentes solo ven AppData.

import type { BBox, MultiPolygon, Polygon, Position } from "geojson";
import type {
  AgebGeoJSON,
  AgebMetrics,
  AgebProps,
  AlcaldiaProps,
  AlcaldiasGeoJSON,
  ColoniaIndex,
  ColoniaProps,
  Colonias,
  ColoniasGeoJSON,
  Ficha,
  Giros,
  Meta,
  Prioridad,
  Prioridades,
  RuralGeoJSON,
  Validacion,
  ZonaValidacion,
  ZonaMetrics,
} from "../types/data";

export interface AppData {
  meta: Meta;
  geo: AgebGeoJSON;
  metrics: AgebMetrics;
  prioridades: Prioridades;
  colonias: Colonias;
  giros: Giros;
  alcaldiasGeo: AlcaldiasGeoJSON;
  alcaldiasMetrics: ZonaMetrics;
  coloniasGeo: ColoniasGeoJSON;
  coloniasMetrics: ZonaMetrics;
  rural: RuralGeoJSON;
  validacion: Validacion;
  ficha: Ficha;
  // Índices derivados (no forman parte del contrato).
  agebPorId: Map<string, AgebProps>;
  bboxPorAgeb: Map<string, BBox>;
  coloniaPorId: Map<string, ColoniaIndex>;
  prioridadPorId: Map<string, Prioridad>;
  alcaldiaPorId: Map<string, AlcaldiaProps>;
  alcaldiaPorNombre: Map<string, AlcaldiaProps>;
  bboxPorAlcaldia: Map<string, BBox>;
  coloniaGeoPorId: Map<string, ColoniaProps>;
  bboxPorColonia: Map<string, BBox>;
  validacionPorId: Map<string, ZonaValidacion>;
}

const ARCHIVOS = {
  meta: "meta.json",
  geo: "ageb.geojson",
  metrics: "ageb_metrics.json",
  prioridades: "prioridades.json",
  colonias: "colonias.json",
  giros: "giros.json",
  alcaldiasGeo: "alcaldias.geojson",
  alcaldiasMetrics: "alcaldias_metrics.json",
  coloniasGeo: "colonias.geojson",
  coloniasMetrics: "colonias_metrics.json",
  rural: "fuera_analisis.geojson",
  validacion: "validacion.json",
  ficha: "ficha.json",
} as const;

async function leer(archivo: string): Promise<unknown> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${archivo}`);
  if (!res.ok) throw new Error(`No se pudo cargar ${archivo} (HTTP ${res.status})`);
  return res.json();
}

// En desarrollo se valida contra el contrato con zod (import dinámico: no entra al bundle de producción).
async function validar(datos: Record<keyof typeof ARCHIVOS, unknown>) {
  if (!import.meta.env.DEV) return;
  const s = await import("./schemas");
  const esquemas = {
    meta: s.metaSchema,
    geo: s.agebGeoSchema,
    metrics: s.agebMetricsSchema,
    prioridades: s.prioridadesSchema,
    colonias: s.coloniasSchema,
    giros: s.girosSchema,
    alcaldiasGeo: s.alcaldiasGeoSchema,
    alcaldiasMetrics: s.zonaMetricsSchema,
    coloniasGeo: s.coloniasGeoSchema,
    coloniasMetrics: s.zonaMetricsSchema,
    rural: s.ruralGeoSchema,
    validacion: s.validacionSchema,
    ficha: s.fichaSchema,
  };
  for (const clave of Object.keys(esquemas) as (keyof typeof esquemas)[]) {
    const r = esquemas[clave].safeParse(datos[clave]);
    if (!r.success) {
      console.error(`[contrato] ${ARCHIVOS[clave]} no cumple el esquema`, r.error.issues.slice(0, 10));
      throw new Error(`${ARCHIVOS[clave]} no cumple el contrato de datos (ver consola)`);
    }
  }
}

function bboxDe(geom: Polygon | MultiPolygon): BBox {
  let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
  const anillos: Position[][] = geom.type === "Polygon" ? geom.coordinates : geom.coordinates.flat();
  for (const anillo of anillos)
    for (const [x, y] of anillo) {
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
  return [x0, y0, x1, y1];
}

export async function cargarDatos(): Promise<AppData> {
  const claves = Object.keys(ARCHIVOS) as (keyof typeof ARCHIVOS)[];
  const valores = await Promise.all(claves.map((c) => leer(ARCHIVOS[c])));
  const crudo = Object.fromEntries(claves.map((c, i) => [c, valores[i]])) as Record<keyof typeof ARCHIVOS, unknown>;
  await validar(crudo);

  const datos = crudo as unknown as Pick<AppData, keyof typeof ARCHIVOS>;
  const alcaldias = datos.alcaldiasGeo.features;
  return {
    ...datos,
    agebPorId: new Map(datos.geo.features.map((f) => [f.properties.cvegeo, f.properties])),
    bboxPorAgeb: new Map(datos.geo.features.map((f) => [f.properties.cvegeo, bboxDe(f.geometry)])),
    coloniaPorId: new Map(datos.colonias.map((c) => [c.cve_col, c])),
    prioridadPorId: new Map(datos.prioridades.map((p) => [p.cvegeo, p])),
    alcaldiaPorId: new Map(alcaldias.map((f) => [f.properties.cve_mun, f.properties])),
    alcaldiaPorNombre: new Map(alcaldias.map((f) => [f.properties.alcaldia, f.properties])),
    bboxPorAlcaldia: new Map(alcaldias.map((f) => [f.properties.cve_mun, bboxDe(f.geometry)])),
    coloniaGeoPorId: new Map(datos.coloniasGeo.features.map((f) => [f.properties.cve_col, f.properties])),
    bboxPorColonia: new Map(datos.coloniasGeo.features.map((f) => [f.properties.cve_col, bboxDe(f.geometry)])),
    validacionPorId: new Map(datos.validacion.zonas.map((z) => [z.cvegeo, z])),
  };
}
