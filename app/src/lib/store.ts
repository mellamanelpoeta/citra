import type { BBox } from "geojson";
import { useMemo } from "react";
import { create } from "zustand";
import type { Margenes } from "./camara";
import { ALTO_ENCABEZADO, MARGEN, PANEL } from "./config";

// Pasos de la narrativa + la vista del pequeño negocio. "contexto": página de entrada sin mapa (problema, definiciones,
// dificultades y hallazgos). "validar" (v0.5): cómo habría funcionado la lista un año atrás.
export type Vista = "contexto" | "anormal" | "validar" | "priorizar" | "consulta";
// Qué se pinta en el mapa. En "¿Qué es anormal?" lo elige el recorrido del panel (comercio → conteo → riesgo); en
// "¿Habría funcionado?", también el recorrido (lista por conteo → lista por riesgo).
export type Modo = "comercio" | "conteo" | "riesgo";
// En modo riesgo: 3 categorías o RR continuo.
export type Escala = "categoria" | "rr";
// "anticipado": el año siguiente al último observado (contrato v0.5); o un año observado ("2020", …). El mapa ya no
// elige año: siempre muestra la lista del año siguiente; la serie anual vive en el detalle de cada zona.
export type Periodo = "anticipado" | string;
// Unidad que se pinta en el mapa. El modelo es por AGEB; alcaldía y colonia agregan sus AGEB (contrato v0.4).
export type Nivel = "alcaldia" | "colonia" | "ageb";
export type ZonaSel = { nivel: "alcaldia" | "colonia"; id: string };
// Cómo se mueve el mapa hacia una zona: "acercar" (clic en el mapa) lleva siempre al mismo nivel de
// zoom; "mostrar" (listas, buscador) encuadra la zona completa.
export type TipoEncuadre = "acercar" | "mostrar";

interface Estado {
  vista: Vista;
  modo: Modo;
  escala: Escala;
  periodo: Periodo;
  resaltarCambios: boolean;
  // "¿Habría funcionado?": paso del recorrido a la vista (índice de PASOS_PRUEBA).
  pasoPrueba: number;
  nivel: Nivel;
  agebSel: string | null;
  // Alcaldía o colonia elegida en el mapa (vistas técnicas). La colonia de "Consulta mi zona" va aparte (coloniaSel).
  zonaSel: ZonaSel | null;
  coloniaSel: string | null;
  giroSel: string | null;
  // Petición de encuadre para el mapa; el contador fuerza el efecto aunque el bbox se repita.
  encuadre: { bbox: BBox; n: number; tipo: TipoEncuadre } | null;
  // Panel lateral: ancho elegido por el usuario (se recuerda entre visitas) y si está visible.
  panel: { ancho: number; abierto: boolean };

  irA: (v: Vista) => void;
  setModo: (m: Modo) => void;
  setEscala: (e: Escala) => void;
  setResaltarCambios: (b: boolean) => void;
  setPasoPrueba: (i: number) => void;
  setNivel: (n: Nivel) => void;
  seleccionarZona: (z: ZonaSel | null, bbox?: BBox, tipo?: TipoEncuadre) => void;
  seleccionarAgeb: (id: string | null, bbox?: BBox, tipo?: TipoEncuadre) => void;
  seleccionarColonia: (id: string | null, bbox?: BBox, tipo?: TipoEncuadre) => void;
  setGiro: (id: string | null) => void;
  setAnchoPanel: (ancho: number) => void;
  setPanelAbierto: (abierto: boolean) => void;
}

// Recorrido de "¿Habría funcionado?": qué lista pinta el mapa en cada paso y si se marcan en cian las zonas de la lista
// por riesgo que el conteo no veía. El último paso (la tabla) no fija el modo: ahí lo elige la propia tabla.
export const PASOS_PRUEBA: { t: string; modo?: Modo; nuevas: boolean }[] = [
  { t: "Conteo", modo: "conteo", nuevas: false },
  { t: "Riesgo", modo: "riesgo", nuevas: false },
  { t: "Periferia", modo: "riesgo", nuevas: true },
  { t: "Recursos", modo: "riesgo", nuevas: true },
  { t: "Tabla", nuevas: true },
];

// Cada paso propone un modo de mapa. "anormal" arranca en el comercio y "validar" en la lista por conteo: el recorrido
// del panel los lleva al riesgo.
const MODO_POR_VISTA: Record<Vista, Modo> = {
  contexto: "conteo",
  anormal: "comercio",
  validar: "conteo",
  priorizar: "riesgo",
  consulta: "riesgo",
};

// Nivel que propone cada paso: las listas son por AGEB y "Consulta mi zona" habla de colonias. En "¿Qué es anormal?"
// se conserva el que haya elegido el usuario (alcaldía, colonia o AGEB).
const NIVEL_POR_VISTA: Record<Vista, Nivel | null> = {
  contexto: null,
  anormal: null,
  validar: "ageb",
  priorizar: "ageb",
  consulta: "colonia",
};

const CLAVE_ANCHO = "citra.anchoPanel";

// El panel nunca deja al mapa con menos de PANEL.mapaMin px.
export const limitarAncho = (ancho: number) =>
  Math.round(Math.max(PANEL.min, Math.min(ancho, PANEL.max, window.innerWidth - PANEL.mapaMin)));

function anchoGuardado() {
  try {
    const v = Number(localStorage.getItem(CLAVE_ANCHO));
    if (v > 0) return limitarAncho(v);
  } catch {
    // Sin localStorage (modo privado estricto): ancho por defecto.
  }
  return limitarAncho(PANEL.inicial);
}

const pedirEncuadre = (s: Estado, bbox: BBox | undefined, tipo: TipoEncuadre) =>
  bbox ? { bbox, n: (s.encuadre?.n ?? 0) + 1, tipo } : s.encuadre;

export const useApp = create<Estado>((set) => ({
  vista: "contexto",
  modo: "conteo",
  escala: "categoria",
  periodo: "anticipado",
  resaltarCambios: true,
  pasoPrueba: 0,
  nivel: "ageb",
  agebSel: null,
  zonaSel: null,
  coloniaSel: null,
  giroSel: null,
  encuadre: null,
  panel: { ancho: anchoGuardado(), abierto: true },

  irA: (vista) =>
    set((s) => ({
      vista,
      modo: MODO_POR_VISTA[vista],
      pasoPrueba: 0,
      escala: vista === "consulta" ? "categoria" : s.escala,
      nivel: NIVEL_POR_VISTA[vista] ?? s.nivel,
      agebSel: null,
      zonaSel: null,
      panel: { ...s.panel, abierto: true },
    })),
  setModo: (modo) => set({ modo }),
  setEscala: (escala) => set({ escala }),
  setResaltarCambios: (resaltarCambios) => set({ resaltarCambios }),
  setPasoPrueba: (pasoPrueba) => set({ pasoPrueba }),
  setNivel: (nivel) => set((s) => ({ nivel, zonaSel: null, agebSel: nivel === "ageb" ? s.agebSel : null })),
  // Elegir una zona abre el panel: ahí está su detalle. El mapa pasa al nivel de la zona elegida.
  seleccionarZona: (zonaSel, bbox, tipo = "mostrar") =>
    set((s) => ({
      zonaSel,
      agebSel: null,
      nivel: zonaSel ? zonaSel.nivel : s.nivel,
      encuadre: pedirEncuadre(s, bbox, tipo),
      panel: zonaSel ? { ...s.panel, abierto: true } : s.panel,
    })),
  seleccionarAgeb: (agebSel, bbox, tipo = "mostrar") =>
    set((s) => ({
      agebSel,
      zonaSel: agebSel ? null : s.zonaSel,
      nivel: agebSel ? "ageb" : s.nivel,
      encuadre: pedirEncuadre(s, bbox, tipo),
      panel: agebSel ? { ...s.panel, abierto: true } : s.panel,
    })),
  seleccionarColonia: (coloniaSel, bbox, tipo = "mostrar") =>
    set((s) => ({
      coloniaSel,
      encuadre: pedirEncuadre(s, bbox, tipo),
      panel: coloniaSel ? { ...s.panel, abierto: true } : s.panel,
    })),
  setGiro: (giroSel) => set({ giroSel }),
  setAnchoPanel: (a) => {
    const ancho = limitarAncho(a);
    try {
      localStorage.setItem(CLAVE_ANCHO, String(ancho));
    } catch {
      // Sin localStorage: el ancho solo dura la sesión.
    }
    set((s) => ({ panel: { ...s.panel, ancho } }));
  },
  setPanelAbierto: (abierto) => set((s) => ({ panel: { ...s.panel, abierto } })),
}));

// Ancho real del panel: la tabla de prioridades pide un mínimo mayor que el resto de las vistas.
export const useAnchoPanel = () =>
  useApp((s) =>
    Math.max(s.panel.ancho, (s.vista === "priorizar" || s.vista === "validar") && !s.agebSel && !s.zonaSel ? PANEL.minTabla : PANEL.min),
  );

// Lo que el encabezado y el panel le tapan al mapa (px desde cada borde de la ventana).
export function useMargenesMapa(): Margenes {
  const ancho = useAnchoPanel();
  const abierto = useApp((s) => s.panel.abierto);
  return useMemo(
    () => ({
      top: 2 * MARGEN + ALTO_ENCABEZADO,
      right: abierto ? ancho + 2 * MARGEN : MARGEN,
      bottom: MARGEN,
      left: MARGEN,
    }),
    [ancho, abierto],
  );
}
