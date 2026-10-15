// Tipos de los archivos de public/data/: contrato de datos (v0.9) entre el pipeline de Python y la app.
// Cualquier cambio se hace primero en el contrato y luego aquí (y en src/data/schemas.ts).

import type { FeatureCollection, MultiPolygon, Polygon } from "geojson";

// ---------- meta.json ----------
export type CategoriaRiesgo = "por_debajo" | "dentro" | "por_encima";

export interface Meta {
  generado_en: string; // ISO datetime
  version_contrato: "0.8";
  anios: number[]; // años observados, p. ej. [2020, ..., 2024]
  anticipado: Anticipacion; // (v0.5) año de la lista de prioridades (el siguiente al último observado)
  variable_objetivo: string; // p. ej. "Robo a negocio (con y sin violencia)"  (EDA)
  modelo_esperado: "M1" | "M2"; // modelo usado para Ê
  umbrales_categoria: { por_encima: number; por_debajo: number }; // sobre p_exceso; default 0.9 / 0.1
  hipotesis: Record<"A" | "B" | "C", Veredicto>;
  fuentes: { nombre: string; corte: string; url: string }[];
  cobertura: Cobertura; // (v0.4)
}

// La lista anticipa el año siguiente al último observado, con más peso a los años recientes (v0.5).
export interface Anticipacion {
  anio: number; // p. ej. 2025
  datos: string; // años usados, p. ej. "2020–2024"
  vida_media_anios: number; // cada año pesa la mitad que el siguiente cada `vida_media_anios`
}

// Qué deja fuera el análisis al usar solo AGEB urbanas (v0.4). Proporciones 0–1.
export interface Cobertura {
  agebs_urbanas: number;
  agebs_rurales: number;
  pct_area_urbana: number; // área de la CDMX cubierta por AGEB urbanas
  pct_robos_rurales: number; // robos de la variable objetivo (periodo) que caen en AGEB rurales
  pct_establecimientos_rurales: number; // establecimientos DENUE (corte fijo) en AGEB rurales
  pct_area_urbana_sin_colonia: number; // área urbana que no cubre ninguna colonia del catálogo
  pct_robos_sin_colonia: number; // robos del periodo en esa área: no aparecen en el nivel colonia
}

export interface Veredicto {
  titulo: string; // "¿Cuánto?" | "¿Cuáles?" | "¿Dónde?"
  enunciado: string;
  estadistico: string; // p. ej. "β = 0.78 [0.71, 0.85]"
  resultado: "confirmada" | "rechazada" | "no_concluyente";
  lectura: string; // el "entonces qué" en una frase (resumen ejecutivo, v0.8)
  evidencia: string[]; // 2–3 cifras que la sustentan, lenguaje llano
}

// ---------- ageb.geojson ----------
// FeatureCollection<Polygon | MultiPolygon, AgebProps>
export interface AgebProps {
  cvegeo: string; // clave AGEB INEGI (13 chars), id de unión con todo lo demás
  alcaldia: string;
  colonia: string; // colonia dominante (por área) para etiquetar
  cve_col: string; // clave de esa colonia en el catálogo CDMX (p. ej. "002-001")
  poblacion_2020: number;
}

export type AgebGeoJSON = FeatureCollection<Polygon | MultiPolygon, AgebProps>;

// ---------- ageb_metrics.json ----------
export interface AgebMetrics {
  [cvegeo: string]: {
    anticipado: MetricaAgebAnticipada; // lista del año siguiente (ranking y vista por defecto) (v0.5)
    por_anio: { [anio: string]: MetricaAgeb };
  };
}

export interface MetricaAgeb {
  observados: number; // O: carpetas de la variable objetivo
  esperados: number; // Ê fuera de muestra
  rr: number; // riesgo relativo contraído E[θ|O]
  p_exceso: number; // P(θ > 1 | O), 0–1
  categoria: CategoriaRiesgo; // derivada de p_exceso y umbrales de meta
  establecimientos: number;
  pct_efectivo: number; // % de establecimientos en giros intensivos en efectivo, 0–1 (EDA)
  delito_dominante?: string;
  rank_conteo: number; // 1 = más delitos crudos
  rank_riesgo: number; // 1 = mayor p_exceso
}

// Lista del año siguiente (v0.5): `observados` es el ritmo reciente (robos por año, ponderado) y `esperados` lo que le
// corresponde en un año por su actividad. v0.6 quita `anticipados` (era rr × esperados).
export type MetricaAgebAnticipada = MetricaAgeb;

// ---------- prioridades.json ----------
export interface Prioridad {
  cvegeo: string;
  colonia: string;
  cve_col: string;
  alcaldia: string;
  rank_riesgo: number;
  rank_conteo: number;
  p_exceso: number;
  rr: number;
  esperados: number; // robos que le corresponden en un año por su actividad (v0.5)
  persistencia: number; // nº de años con categoria = "por_encima"
  visible_por_conteo: boolean; // ¿está también en el top 10% por conteo crudo?
  rank_riesgo_violencia: number; // puesto por riesgo ajustado con solo robo con violencia (v0.3)
  senal_sin_violencia: boolean; // aviso: rank_riesgo_violencia fuera del top 25% (v0.3)
}
export type Prioridades = Prioridad[]; // top N (N ≈ 10% de AGEB), ordenado por rank_riesgo

// ---------- validacion.json (v0.5; v0.6 todas las AGEB y su serie) ----------
// "Si lo hubiéramos hecho en 2024": la misma receta con datos hasta el año anterior, comparada con lo que pasó. Todo
// sale de un modelo que no vio el año evaluado.
export interface ResultadoLista {
  n: number;
  pct_mas_de_lo_esperado: number; // 0–1: zonas con más robos que los esperados por su actividad ese año
  pct_por_encima: number; // 0–1: zonas con categoría "por_encima" ese año (más estricto)
  robos: number;
  esperados: number;
  pct_exceso_ciudad: number; // 0–1: parte del exceso de robo de la ciudad que cae en la lista
}
export interface ZonaValidacion {
  cvegeo: string;
  colonia: string;
  cve_col: string;
  alcaldia: string;
  rank_riesgo: number; // puesto en la lista armada con datos hasta el año anterior
  rank_conteo: number;
  en_lista_riesgo: boolean;
  en_lista_conteo: boolean;
  rr: number; // la señal: riesgo con datos hasta el año anterior
  p_exceso: number;
  categoria: CategoriaRiesgo; // categoría con datos hasta el año anterior
  serie: { anio: number; observados: number; esperados: number }[]; // años con que se armó la lista
  esperados: number; // el resultado: esperados ese año por su actividad
  observados: number; // lo que pasó ese año
  mas_de_lo_esperado: boolean;
}
// Quién vive en las zonas de cada lista (Censo 2020, v0.7): contexto de la zona, no de los negocios.
export interface Vulnerabilidad {
  n: number;
  poblacion: number;
  escolaridad: number; // años promedio, población de 15 años o más
  pct_bajo_mediana_escolaridad: number; // 0–1: AGEB con escolaridad por debajo de la mediana de la ciudad
  pct_sin_basica: number; // 0–1: población de 15 años o más sin secundaria terminada
  pct_sin_salud: number; // 0–1: población sin afiliación a servicios de salud
  pct_sin_internet: number; // 0–1: viviendas sin internet
}
export type GrupoValidacion = "lista_riesgo" | "lista_conteo" | "no_visibles" | "ciudad";
export interface Validacion {
  anio: number; // año evaluado, p. ej. 2024
  datos: string; // años con los que se armó la lista, p. ej. "2020–2023"
  resumen: Record<GrupoValidacion, ResultadoLista>;
  vulnerabilidad: Record<GrupoValidacion, Vulnerabilidad>;
  zonas: ZonaValidacion[]; // todas las AGEB con métricas, ordenadas por rank_riesgo
}

// ---------- colonias.json ----------
export interface ColoniaIndex {
  cve_col: string; // id único (los nombres se repiten entre alcaldías)
  nombre: string;
  alcaldia: string;
  agebs: string[]; // cvegeo que intersectan la colonia
  centroide: [number, number]; // [lon, lat]
  bbox: [number, number, number, number];
  categoria: CategoriaRiesgo; // agregado ponderado por establecimientos
  rr: number;
  similares: string[]; // cve_col de 5 colonias con exposición y composición similares
}
export type Colonias = ColoniaIndex[];

// ---------- giros.json ----------
export interface GrupoGiro {
  id: string; // p. ej. "efectivo", "alimentos_bebidas", "servicios"
  nombre: string; // etiqueta en lenguaje llano
  scian: string[]; // prefijos SCIAN incluidos (EDA)
  irr: number | null; // razón de tasas del coeficiente en M1 (null si no aplica)
  mensaje: string; // texto llano para "Consulta mi zona"
}
export type Giros = GrupoGiro[];

// ---------- Niveles alcaldía y colonia (v0.4) ----------
// Métricas de una zona agregada: O y Ê suman las de sus AGEB (en colonia, por interpolación areal) y la contracción
// se recalcula con parámetros propios del nivel. Puestos dentro del nivel.
export interface MetricaZona {
  observados: number; // en colonia puede ser fraccional (interpolación areal)
  esperados: number;
  rr: number;
  p_exceso: number;
  categoria: CategoriaRiesgo;
  establecimientos: number;
  rank_conteo: number;
  rank_riesgo: number;
}
// (v0.5) observados = ritmo reciente por año, esperados = lo que le corresponde en un año
export type MetricaZonaAnticipada = MetricaZona;
export interface ZonaMetrics {
  [id: string]: { anticipado: MetricaZonaAnticipada; por_anio: { [anio: string]: MetricaZona } };
}

// alcaldias.geojson: FeatureCollection<Polygon | MultiPolygon, AlcaldiaProps>; alcaldias_metrics.json: ZonaMetrics por cve_mun
export interface AlcaldiaProps {
  cve_mun: string; // "002" … "017"
  alcaldia: string;
  agebs_urbanas: number;
  agebs_rurales: number;
  pct_area_urbana: number; // 0–1
  prioritarias: number; // AGEB de prioridades.json en la alcaldía
  denuncia_relativa?: number; // r_alc de 04d (1 = media geométrica de las alcaldías)
  denuncia_relativa_ic?: [number, number];
}
export type AlcaldiasGeoJSON = FeatureCollection<Polygon | MultiPolygon, AlcaldiaProps>;

// colonias.geojson: catálogo completo; colonias_metrics.json: ZonaMetrics por cve_col (solo las de colonias.json)
export interface ColoniaProps {
  cve_col: string;
  nombre: string;
  alcaldia: string;
}
export type ColoniasGeoJSON = FeatureCollection<Polygon | MultiPolygon, ColoniaProps>;

// fuera_analisis.geojson: AGEB rurales, solo para dibujarlas como fuera del análisis
export interface RuralProps {
  cvegeo: string; // 9 caracteres
  alcaldia: string;
  area_km2: number;
}
export type RuralGeoJSON = FeatureCollection<Polygon | MultiPolygon, RuralProps>;

// ---------- ficha.json (v0.9) ----------
// Contexto descriptivo del detalle de cada AGEB: no entra al modelo ni cambia la lista. Proporciones 0–1.
export type GrupoComposicion =
  | "efectivo_barrio"
  | "efectivo_formal"
  | "comercio_menor"
  | "comercio_mayor"
  | "manufactura"
  | "alojamiento_alim"
  | "serv_prof_fin"
  | "serv_salud_educ"
  | "serv_otros"
  | "otros";
// Giros de alto efectivo (los dos grupos de arriba, desglosados)
export type GiroEfectivo =
  | "abarrotes"
  | "alim_otros"
  | "minisuper"
  | "licores"
  | "farmacias"
  | "gasolineras"
  | "restaurantes"
  | "bares"
  | "empeno_cambio"
  | "tortillerias";
// Censo 2020 (mismos indicadores que Vulnerabilidad); null si INEGI lo suprimió por confidencialidad
export interface CensoFicha {
  escolaridad: number | null;
  pct_sin_basica: number | null;
  pct_sin_salud: number | null;
  pct_sin_internet: number | null; // VPH_INTER sobre TVIVPARHAB
}
// Número de negocios en el DENUE 2019-11 y 2024-11 (actividad de 2018 y 2023), y los que salieron o entraron por id
export interface DinamicaFicha {
  antes: number;
  despues: number;
  salidas: number;
  entradas: number;
}
export interface FichaAgeb {
  composicion: Record<GrupoComposicion, number>; // parte de los establecimientos (DENUE 2023-11) en cada grupo de giro
  efectivo: Partial<Record<GiroEfectivo, number>>; // solo los giros presentes
  personal_ocupado: number; // estimado con el punto medio del estrato
  pct_grandes: number; // establecimientos de 51 personas o más
  densidad: number; // establecimientos por km²
  pct_densidad: number; // 0–1: parte de las AGEB con menor densidad
  censo: CensoFicha;
  dinamica?: DinamicaFicha; // falta en 6 AGEB sin negocios en alguno de los cortes
}
export interface IndicadorEnsu {
  valor: number;
  ee: number;
  rank: number; // 1 = la alcaldía con el valor más alto de las 16
}
export interface FichaAlcaldia {
  envipe: { denuncia: number; ee: number; contraida: number; contraida_ic: [number, number] }; // 04e, ediciones 2021–2025
  ensu: Record<"inseg_ciudad" | "inseg_calle" | "inseg_mercado" | "hog_vict_robo", IndicadorEnsu>; // promedio del año
}
export interface Ficha {
  ciudad: {
    composicion: Record<GrupoComposicion, number>;
    efectivo: Record<GiroEfectivo, number>;
    pct_grandes: number;
    densidad_mediana: number;
    censo: CensoFicha;
    dinamica: DinamicaFicha;
    cambio_mediano: number; // mediana por AGEB de despues / antes − 1
    envipe_denuncia: number;
    envipe_p_iguales: number; // p de la prueba "todas las alcaldías denuncian igual"
    anio_ensu: number;
  };
  agebs: { [cvegeo: string]: FichaAgeb };
  alcaldias: { [cve_mun: string]: FichaAlcaldia };
}
