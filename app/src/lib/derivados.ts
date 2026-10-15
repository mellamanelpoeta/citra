import { useMemo } from "react";
import { useData } from "../data/context";
import { FRACCION_TOP } from "./config";
import { useApp, type Periodo } from "./store";

// Sirve para AGEB y para las zonas agregadas (alcaldía, colonia): misma forma anticipado / por_anio.
// En la lista del año siguiente `observados` es el ritmo reciente (robos por año, ponderado).
export const metricaDe = <M, A extends M>(m: { anticipado: A; por_anio: Record<string, M> }, p: Periodo): M | A =>
  p === "anticipado" ? m.anticipado : m.por_anio[p];

export interface Resumen {
  nTop: number;
  topComercio: Set<string>; // las nTop AGEB con más establecimientos (paso "comercio" del recorrido)
  topConteo: Set<string>;
  topRiesgo: Set<string>;
  soloRiesgo: Set<string>; // prioritarias por riesgo, invisibles por conteo y "por encima de lo esperado"
  soloConteo: Set<string>; // top por conteo que dejan de serlo al ajustar
  totalObservados: number;
  porEncima: number;
}

// Conjuntos "top" del periodo elegido. Mismo criterio que visible_por_conteo (top 10%).
export function useResumen(): Resumen {
  const { metrics } = useData();
  const periodo = useApp((s) => s.periodo);
  return useMemo(() => {
    const filas = Object.entries(metrics).map(([id, m]) => [id, metricaDe(m, periodo)] as const);
    const nTop = Math.round(FRACCION_TOP * filas.length);
    const topConteo = new Set(filas.filter(([, m]) => m.rank_conteo <= nTop).map(([id]) => id));
    const topRiesgo = new Set(filas.filter(([, m]) => m.rank_riesgo <= nTop).map(([id]) => id));
    const porNegocios = [...filas].sort(([, a], [, b]) => b.establecimientos - a.establecimientos);
    return {
      nTop,
      topComercio: new Set(porNegocios.slice(0, nTop).map(([id]) => id)),
      topConteo,
      topRiesgo,
      // La lista es el 10% con más probabilidad de exceso, no un umbral: algunas no llegan a "por encima" (p ≥ 0.9) y en el
      // mapa se veían "dentro de lo esperado". Solo se marcan las que sí.
      soloRiesgo: new Set(filas.filter(([id, m]) => topRiesgo.has(id) && !topConteo.has(id) && m.categoria === "por_encima").map(([id]) => id)),
      soloConteo: new Set([...topConteo].filter((id) => !topRiesgo.has(id))),
      totalObservados: filas.reduce((s, [, m]) => s + m.observados, 0),
      porEncima: filas.filter(([, m]) => m.categoria === "por_encima").length,
    };
  }, [metrics, periodo]);
}

export const etiquetaPeriodo = (p: Periodo, anioAnticipado: number) => (p === "anticipado" ? `Lista ${anioAnticipado}` : p);

// Percentil (0–1, 1 = el valor más alto) de cada id según `valor`. Lo usa el mapa para el número de establecimientos,
// igual que el conteo usa su puesto.
export function percentiles(ids: string[], valor: (id: string) => number): Map<string, number> {
  const orden = [...ids].sort((a, b) => valor(b) - valor(a));
  return new Map(orden.map((id, i) => [id, 1 - i / ids.length]));
}

// Alcaldías centrales: concentran el comercio y el robo denunciado. El recorrido y la prueba las usan para mostrar que
// el conteo mira al centro y el riesgo ajustado se reparte por la ciudad.
export const ALCALDIAS_CENTRO = ["Cuauhtémoc", "Benito Juárez", "Miguel Hidalgo"];

// Reparto de un conjunto de AGEB por alcaldía: parte en las alcaldías centrales y las tres alcaldías con más zonas.
export function reparto(ids: string[], alcaldiaDe: (id: string) => string | undefined) {
  const n = new Map<string, number>();
  for (const id of ids) {
    const a = alcaldiaDe(id) ?? "";
    n.set(a, (n.get(a) ?? 0) + 1);
  }
  const centro = ALCALDIAS_CENTRO.reduce((s, a) => s + (n.get(a) ?? 0), 0) / ids.length;
  const principales = [...n.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([a]) => a);
  return { centro, principales };
}
