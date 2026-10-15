import type { CategoriaRiesgo } from "../types/data";

const nf = new Intl.NumberFormat("es-MX");
const nf1 = new Intl.NumberFormat("es-MX", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const pf = new Intl.NumberFormat("es-MX", { style: "percent", maximumFractionDigits: 0 });

export const entero = (x: number) => nf.format(Math.round(x));
export const decimal = (x: number) => nf1.format(x);
const nf2 = new Intl.NumberFormat("es-MX", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
export const decimal2 = (x: number) => nf2.format(x);
export const porcentaje = (x: number) => pf.format(x);

export const COLOR_CATEGORIA: Record<CategoriaRiesgo, string> = {
  por_debajo: "#3b82f6",
  dentro: "#94a3b8",
  por_encima: "#ef4444",
};
// Relleno del mapa: "dentro" y "por debajo" atenuados para que resalte "por encima".
export const MAPA_CATEGORIA: Record<CategoriaRiesgo, string> = {
  por_debajo: "#1e3a6e",
  dentro: "#2f3b52",
  por_encima: "#ef4444",
};
export const COLOR_SIN_DATOS = "#283246";
// AGEB rurales (fuera del análisis): fondo oscuro con rayado diagonal (MapaAgeb.tsx crea el patrón).
export const COLOR_RURAL = "#0b1120";
export const RAYA_RURAL = "#64748b";
export const COLOR_NUEVO = "#22d3ee";
// Prueba con el año anterior (v0.5): ¿la zona de la lista tuvo más robos de lo esperado? Mapa, leyenda y panel.
export const COLOR_ACIERTO = "#ef4444";
export const COLOR_FALLO = "#64748b";

// Etiquetas técnicas (vista SEDECO).
export const ETIQUETA_CATEGORIA: Record<CategoriaRiesgo, string> = {
  por_debajo: "Por debajo de lo esperado",
  dentro: "Dentro de lo esperado",
  por_encima: "Por encima de lo esperado",
};

// Lenguaje llano (vista "Consulta mi zona"): sin números.
export const MENSAJE_LLANO: Record<CategoriaRiesgo, { titulo: string; detalle: string }> = {
  por_encima: {
    titulo: "Tu zona tiene más robos de los que se esperarían para su nivel de actividad",
    detalle:
      "Comparada con zonas con un número y tipo de negocios parecidos, aquí se denuncian más robos a negocio, sobre todo en los años más recientes.",
  },
  dentro: {
    titulo: "Tu zona tiene los robos que se esperarían para su nivel de actividad",
    detalle: "Los robos a negocio denunciados aquí son parecidos a los de zonas con actividad económica similar.",
  },
  por_debajo: {
    titulo: "Tu zona tiene menos robos de los que se esperarían para su nivel de actividad",
    detalle:
      "Comparada con zonas con un número y tipo de negocios parecidos, aquí se denuncian menos robos a negocio, sobre todo en los años más recientes.",
  },
};

export const sinAcentos = (s: string) =>
  s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

// "Lomas De La Era" → "Lomas de la Era", "Polanco Ii Seccion" → "Polanco II Seccion", "C.t.m." → "C.T.M."
// (el catálogo capitaliza todo y a veces repite espacios).
const MINUSCULAS = new Set(["de", "del", "la", "las", "los", "el", "y"]);
const ROMANO = /^(i{1,3}|iv|v|vi{1,3}|ix|x|xi{1,3})$/;
const SIGLAS = /^([a-z]\.){2,}[a-z]?$/;
export const nombrePropio = (s: string) =>
  s
    .toLowerCase()
    .trim()
    .split(/\s+/)
    .map((p, i) =>
      ROMANO.test(p) || SIGLAS.test(p) ? p.toUpperCase() : i > 0 && MINUSCULAS.has(p) ? p : p.charAt(0).toUpperCase() + p.slice(1),
    )
    .join(" ");

// Aviso "señal dominada por robo sin violencia" (prioridades.json → senal_sin_violencia).
export const COLOR_AVISO = "#fbbf24";
export const AVISO_SIN_VIOLENCIA = {
  titulo: "Señal dominada por robo sin violencia",
  corto: "Revisar caso por caso: es prioritaria casi solo por robo sin violencia",
  detalle:
    "El robo sin violencia se denuncia de forma más desigual entre negocios y a veces se concentra en un establecimiento grande. " +
    "Antes de intervenir conviene revisar qué hay detrás.",
};

// Nombres de los niveles del mapa (contrato v0.4).
export const NIVEL_SINGULAR = { alcaldia: "alcaldía", colonia: "colonia", ageb: "AGEB" } as const;
export const NIVEL_PLURAL = { alcaldia: "alcaldías", colonia: "colonias", ageb: "AGEB" } as const;

// Supuesto de cobertura en lenguaje llano (meta.json → cobertura). Lo usan la leyenda, el tooltip y la página de contexto.
export const pctPequeno = (x: number) =>
  `${new Intl.NumberFormat("es-MX", { maximumFractionDigits: x < 0.01 ? 2 : 1 }).format(x * 100)}%`;
