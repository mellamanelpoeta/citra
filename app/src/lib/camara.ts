// Geometría de la cámara del mapa. El mapa ocupa toda la ventana y el encabezado y el panel flotan
// encima, así que los encuadres y los límites se calculan sobre el "área útil": la parte del lienzo
// que no tapan. Se trabaja en Web Mercator normalizado (0–1, y crece hacia el sur).

import type { BBox } from "geojson";
import type { LngLat, TransformConstrainFunction } from "maplibre-gl";

export interface Margenes {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface Lienzo {
  ancho: number;
  alto: number;
  m: Margenes;
}

const TILE = 512;
const xDe = (lng: number) => (lng + 180) / 360;
const yDe = (lat: number) => {
  const s = Math.sin((lat * Math.PI) / 180);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
};
const lngDe = (x: number) => x * 360 - 180;
const latDe = (y: number) => (360 / Math.PI) * Math.atan(Math.exp((0.5 - y) * 2 * Math.PI)) - 90;
const escala = (zoom: number) => TILE * 2 ** zoom; // px por unidad Mercator

const caja = ([oeste, sur, este, norte]: BBox) => ({ x0: xDe(oeste), y0: yDe(norte), x1: xDe(este), y1: yDe(sur) });

const util = ({ ancho, alto, m }: Lienzo) => ({
  w: Math.max(1, ancho - m.left - m.right),
  h: Math.max(1, alto - m.top - m.bottom),
});

// Cuánto se desplaza (px) el centro del área útil respecto al centro del lienzo.
const desfase = ({ m }: Lienzo) => ({ dx: (m.left - m.right) / 2, dy: (m.top - m.bottom) / 2 });

export const conMargenExtra = (l: Lienzo, px: number): Lienzo => ({
  ...l,
  m: { top: l.m.top + px, right: l.m.right + px, bottom: l.m.bottom + px, left: l.m.left + px },
});

// Zoom con el que la caja cabe justa en el área útil.
export function zoomParaAjustar(b: BBox, l: Lienzo): number {
  const c = caja(b);
  const { w, h } = util(l);
  return Math.log2(Math.min(w / (c.x1 - c.x0), h / (c.y1 - c.y0)) / TILE);
}

// Cámara (centro del lienzo + zoom) que deja la caja centrada en el área útil.
export function camaraPara(b: BBox, l: Lienzo, zoom = zoomParaAjustar(b, l)) {
  const c = caja(b);
  const s = escala(zoom);
  const { dx, dy } = desfase(l);
  const center: [number, number] = [lngDe((c.x0 + c.x1) / 2 - dx / s), latDe((c.y0 + c.y1) / 2 - dy / s)];
  return { center, zoom };
}

// Restricción de la cámara: el área útil nunca sale del límite. El zoom mínimo es el que hace caber el
// límite completo; con más zoom la vista se desliza dentro de él, y en el eje que sobra queda centrada.
// maxBounds de MapLibre no sirve aquí porque se mide sobre todo el lienzo, incluida la parte bajo el panel.
export function restringir(limite: BBox, l: Lienzo, lng: number, lat: number, zoom: number, zoomMax: number) {
  const c = caja(limite);
  const z = Math.min(zoomMax, Math.max(zoom, zoomParaAjustar(limite, l)));
  const s = escala(z);
  const { w, h } = util(l);
  const { dx, dy } = desfase(l);
  const acotar = (v: number, a: number, b: number, medio: number) =>
    b - a <= 2 * medio ? (a + b) / 2 : Math.min(b - medio, Math.max(a + medio, v));
  const ux = acotar(xDe(lng) + dx / s, c.x0, c.x1, w / 2 / s);
  const uy = acotar(yDe(lat) + dy / s, c.y0, c.y1, h / 2 / s);
  return { center: [lngDe(ux - dx / s), latDe(uy - dy / s)] as [number, number], zoom: z };
}

// La misma restricción con la firma de transformConstrain. El lienzo se lee en cada llamada porque
// cambia al arrastrar el panel o al redimensionar la ventana.
export function crearRestriccion(
  LngLatCls: typeof LngLat,
  limite: BBox,
  lienzo: () => Lienzo,
  zoomMax: number,
): TransformConstrainFunction {
  return (lngLat, zoom) => {
    const r = restringir(limite, lienzo(), lngLat.lng, lngLat.lat, +zoom, zoomMax);
    return { center: new LngLatCls(...r.center), zoom: r.zoom };
  };
}

// Curva de las animaciones de encuadre: arranca y frena suave.
export const suavizar = (t: number) => (t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2);
