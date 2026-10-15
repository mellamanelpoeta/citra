// Esquemas zod equivalentes a src/types/data.ts. Se usan para validar:
//  - en el navegador, solo en desarrollo (import dinámico: no entran al bundle de producción);
//  - desde la terminal, con scripts/valida-datos.ts.
// Las aserciones de tipo al final fallan en compilación si esquema y tipos se desalinean.

import { z } from "zod";
import type {
  AgebGeoJSON,
  AgebMetrics,
  AlcaldiasGeoJSON,
  Colonias,
  ColoniasGeoJSON,
  Giros,
  Meta,
  Prioridades,
  RuralGeoJSON,
  Validacion,
  ZonaMetrics,
} from "../types/data";

const prob = z.number().min(0).max(1);
const categoria = z.enum(["por_debajo", "dentro", "por_encima"]);

const veredicto = z.object({
  titulo: z.string(),
  enunciado: z.string(),
  estadistico: z.string(),
  resultado: z.enum(["confirmada", "rechazada", "no_concluyente"]),
  lectura: z.string(),
  evidencia: z.array(z.string()),
});

export const metaSchema = z.object({
  generado_en: z.iso.datetime({ offset: true }),
  version_contrato: z.literal("0.8"),
  anios: z.array(z.number().int()).min(1),
  anticipado: z.object({ anio: z.number().int(), datos: z.string(), vida_media_anios: z.number().positive() }),
  variable_objetivo: z.string(),
  modelo_esperado: z.enum(["M1", "M2"]),
  umbrales_categoria: z.object({ por_encima: prob, por_debajo: prob }),
  hipotesis: z.object({ A: veredicto, B: veredicto, C: veredicto }),
  fuentes: z.array(z.object({ nombre: z.string(), corte: z.string(), url: z.string() })),
  cobertura: z.object({
    agebs_urbanas: z.number().int().min(1),
    agebs_rurales: z.number().int().min(0),
    pct_area_urbana: prob,
    pct_robos_rurales: prob,
    pct_establecimientos_rurales: prob,
    pct_area_urbana_sin_colonia: prob,
    pct_robos_sin_colonia: prob,
  }),
});

const cvegeo = z.string().regex(/^\d{9}[\dA-Z]{4}$/, "cvegeo debe tener 13 caracteres");
const cveCol = z.string().min(1);
const position = z.array(z.number()).min(2);
const ring = z.array(position);

const coleccion = <T extends z.ZodType>(properties: T) =>
  z.object({
    type: z.literal("FeatureCollection"),
    features: z.array(
      z.object({
        type: z.literal("Feature"),
        geometry: z.discriminatedUnion("type", [
          z.object({ type: z.literal("Polygon"), coordinates: z.array(ring) }),
          z.object({ type: z.literal("MultiPolygon"), coordinates: z.array(z.array(ring)) }),
        ]),
        properties,
      }),
    ),
  });

export const agebGeoSchema = coleccion(
  z.object({
    cvegeo,
    alcaldia: z.string(),
    colonia: z.string(),
    cve_col: cveCol,
    poblacion_2020: z.number().int().min(0),
  }),
);

const metrica = z.object({
  observados: z.number().int().min(0),
  esperados: z.number().min(0),
  rr: z.number().min(0),
  p_exceso: prob,
  categoria,
  establecimientos: z.number().int().min(0),
  pct_efectivo: prob,
  delito_dominante: z.string().optional(),
  rank_conteo: z.number().int().min(1),
  rank_riesgo: z.number().int().min(1),
});

const metricaAnticipada = metrica.extend({ observados: z.number().min(0) });

export const agebMetricsSchema = z.record(
  cvegeo,
  z.object({ anticipado: metricaAnticipada, por_anio: z.record(z.string().regex(/^\d{4}$/), metrica) }),
);

export const prioridadesSchema = z.array(
  z.object({
    cvegeo,
    colonia: z.string(),
    cve_col: cveCol,
    alcaldia: z.string(),
    rank_riesgo: z.number().int().min(1),
    rank_conteo: z.number().int().min(1),
    p_exceso: prob,
    rr: z.number().min(0),
    esperados: z.number().min(0),
    persistencia: z.number().int().min(0),
    visible_por_conteo: z.boolean(),
    rank_riesgo_violencia: z.number().int().min(1),
    senal_sin_violencia: z.boolean(),
  }),
);

export const coloniasSchema = z.array(
  z.object({
    cve_col: cveCol,
    nombre: z.string(),
    alcaldia: z.string(),
    agebs: z.array(cvegeo),
    centroide: z.tuple([z.number(), z.number()]),
    bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]),
    categoria,
    rr: z.number().min(0),
    similares: z.array(cveCol),
  }),
);

const metricaZona = z.object({
  observados: z.number().min(0),
  esperados: z.number().min(0),
  rr: z.number().min(0),
  p_exceso: prob,
  categoria,
  establecimientos: z.number().int().min(0),
  rank_conteo: z.number().int().min(1),
  rank_riesgo: z.number().int().min(1),
});

export const zonaMetricsSchema = z.record(
  z.string().min(1),
  z.object({
    anticipado: metricaZona,
    por_anio: z.record(z.string().regex(/^\d{4}$/), metricaZona),
  }),
);

const resultadoLista = z.object({
  n: z.number().int().min(0),
  pct_mas_de_lo_esperado: prob,
  pct_por_encima: prob,
  robos: z.number().int().min(0),
  esperados: z.number().min(0),
  pct_exceso_ciudad: prob,
});

const vulnerabilidad = z.object({
  n: z.number().int().min(1),
  poblacion: z.number().int().min(0),
  escolaridad: z.number().min(0),
  pct_bajo_mediana_escolaridad: prob,
  pct_sin_basica: prob,
  pct_sin_salud: prob,
  pct_sin_internet: prob,
});

export const validacionSchema = z.object({
  anio: z.number().int(),
  datos: z.string(),
  resumen: z.object({ lista_riesgo: resultadoLista, lista_conteo: resultadoLista, no_visibles: resultadoLista, ciudad: resultadoLista }),
  vulnerabilidad: z.object({ lista_riesgo: vulnerabilidad, lista_conteo: vulnerabilidad, no_visibles: vulnerabilidad, ciudad: vulnerabilidad }),
  zonas: z.array(
    z.object({
      cvegeo,
      colonia: z.string(),
      cve_col: cveCol,
      alcaldia: z.string(),
      rank_riesgo: z.number().int().min(1),
      rank_conteo: z.number().int().min(1),
      en_lista_riesgo: z.boolean(),
      en_lista_conteo: z.boolean(),
      rr: z.number().min(0),
      p_exceso: prob,
      categoria,
      serie: z.array(z.object({ anio: z.number().int(), observados: z.number().int().min(0), esperados: z.number().min(0) })).min(1),
      esperados: z.number().min(0),
      observados: z.number().int().min(0),
      mas_de_lo_esperado: z.boolean(),
    }),
  ),
});

export const alcaldiasGeoSchema = coleccion(
  z.object({
    cve_mun: z.string().regex(/^\d{3}$/),
    alcaldia: z.string(),
    agebs_urbanas: z.number().int().min(0),
    agebs_rurales: z.number().int().min(0),
    pct_area_urbana: prob,
    prioritarias: z.number().int().min(0),
    denuncia_relativa: z.number().min(0).optional(),
    denuncia_relativa_ic: z.tuple([z.number(), z.number()]).optional(),
  }),
);

export const coloniasGeoSchema = coleccion(z.object({ cve_col: cveCol, nombre: z.string(), alcaldia: z.string() }));

export const ruralGeoSchema = coleccion(
  z.object({ cvegeo: z.string().regex(/^\d{5}[\dA-Z]{4}$/), alcaldia: z.string(), area_km2: z.number().positive() }),
);

export const girosSchema = z.array(
  z.object({
    id: z.string(),
    nombre: z.string(),
    scian: z.array(z.string()),
    irr: z.number().nullable(),
    mensaje: z.string(),
  }),
);

// ---------- Alineación esquema ↔ tipos (error de compilación si divergen) ----------
type Same<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;
const assertSame = <T extends true>(_: T) => {};
assertSame<Same<z.infer<typeof metaSchema>, Meta>>(true);
assertSame<Same<z.infer<typeof agebMetricsSchema>, AgebMetrics>>(true);
assertSame<Same<z.infer<typeof prioridadesSchema>, Prioridades>>(true);
assertSame<Same<z.infer<typeof coloniasSchema>, Colonias>>(true);
assertSame<Same<z.infer<typeof girosSchema>, Giros>>(true);
assertSame<Same<z.infer<typeof zonaMetricsSchema>, ZonaMetrics>>(true);
assertSame<Same<z.infer<typeof validacionSchema>, Validacion>>(true);
// La geometría GeoJSON se valida estructuralmente; basta con que sea asignable.
const _geo = (x: z.infer<typeof agebGeoSchema>): AgebGeoJSON => x;
const _alc = (x: z.infer<typeof alcaldiasGeoSchema>): AlcaldiasGeoJSON => x;
const _col = (x: z.infer<typeof coloniasGeoSchema>): ColoniasGeoJSON => x;
const _rur = (x: z.infer<typeof ruralGeoSchema>): RuralGeoJSON => x;
void [_geo, _alc, _col, _rur];
