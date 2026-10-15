import { useId } from "react";
import colchon from "../data/colchon.json";
import { decimal, entero } from "../lib/format";

const ACENTO = "#38bdf8";
const GRIS = "#94a3b8";
const TENUE = "#475569";

// Colchón de densidad (hipótesis A, T7 partes 1b–1c): robos por km² contra negocios por km² en hexágonos H3, en escala
// log-log. La recta tiene el quiebre en el nodo estimado; la punteada sigue el ritmo de antes del quiebre. SVG propio,
// como MiniTendencia: 12 puntos no justifican una librería de gráficas.
// compacta: para el panel lateral (~380 px), con un lienzo más angosto para que el texto no se encoja.
export default function GraficaColchon({ compacta = false }: { compacta?: boolean }) {
  const W = compacta ? 380 : 560;
  const H = compacta ? 280 : 300;
  const m = { l: 44, r: compacta ? 12 : 20, t: 16, b: 38 };
  const area = `colchon-area-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
  const [x0, x1, y0, y1] = [Math.log10(4), Math.log10(4000), Math.log10(0.03), Math.log10(40)];
  const x = (v: number) => m.l + ((Math.log10(v) - x0) / (x1 - x0)) * (W - m.l - m.r);
  const y = (v: number) => m.t + (1 - (Math.log10(v) - y0) / (y1 - y0)) * (H - m.t - m.b);
  const ruta = (ps: { negocios_km2: number; robos_km2: number }[]) =>
    ps.map((p, i) => `${i ? "L" : "M"}${x(p.negocios_km2)},${y(p.robos_km2)}`).join("");
  const nodo = colchon.nodo_km2;
  const fin = [...colchon.sin_quiebre].reverse().find((p) => p.negocios_km2 <= 2500) ?? colchon.sin_quiebre[0];
  const p25 = colchon.percentiles_km2.p25;

  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Robos por km² contra negocios por km²">
        <defs>
          <clipPath id={area}>
            <rect x={m.l} y={m.t} width={W - m.l - m.r} height={H - m.t - m.b} />
          </clipPath>
        </defs>
        {/* Zona más densa: a la derecha del quiebre */}
        <rect x={x(nodo)} y={m.t} width={W - m.r - x(nodo)} height={H - m.t - m.b} fill={`${ACENTO}0f`} />
        {/* En la compacta la franja mide ~105 px: la etiqueta va en dos renglones, abajo y dentro de la franja */}
        {compacta ? (
          <text x={x(nodo) + 6} y={H - m.b - 18} className="fill-sky-300 text-[10px]">
            <tspan>Zonas con</tspan>
            <tspan x={x(nodo) + 6} dy={12}>
              más comercio
            </tspan>
          </text>
        ) : (
          <text x={W - m.r - 6} y={m.t + 12} textAnchor="end" className="fill-sky-300 text-[10px]">
            Zonas con más comercio
          </text>
        )}
        {[10, 100, 1000].map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={m.t} y2={H - m.b} stroke="rgb(255 255 255 / 0.06)" />
            <text x={x(v)} y={H - m.b + 13} textAnchor="middle" className="fill-slate-500 text-[9px]">
              {entero(v)}
            </text>
          </g>
        ))}
        {[0.1, 1, 10].map((v) => (
          <g key={v}>
            <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} stroke="rgb(255 255 255 / 0.06)" />
            <text x={m.l - 5} y={y(v) + 3} textAnchor="end" className="fill-slate-500 text-[9px]">
              {v < 1 ? decimal(v) : entero(v)}
            </text>
          </g>
        ))}
        <text x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle" className="fill-slate-400 text-[10px]">
          Negocios por km² (escala logarítmica)
        </text>
        <text transform={`translate(11 ${(m.t + H - m.b) / 2}) rotate(-90)`} textAnchor="middle" className="fill-slate-400 text-[10px]">
          Robos al año por km²
        </text>

        {/* Quiebre estimado */}
        <line x1={x(nodo)} x2={x(nodo)} y1={m.t} y2={H - m.b} stroke={ACENTO} strokeOpacity={0.5} strokeDasharray="3 3" />
        {/* Arriba a la izquierda del quiebre no hay puntos ni la punteada gris (que sale hacia la derecha) */}
        <text x={x(nodo) - 5} y={m.t + 12} textAnchor="end" className="fill-sky-300 text-[10px]">
          quiebre
        </text>

        <path d={ruta(colchon.sin_quiebre)} clipPath={`url(#${area})`} fill="none" stroke={GRIS} strokeWidth={1.5} strokeDasharray="5 4" />
        {!compacta && (
          <text x={x(fin.negocios_km2) - 4} y={y(fin.robos_km2) - 6} textAnchor="end" className="fill-slate-400 text-[10px]">
            si siguiera al mismo ritmo
          </text>
        )}
        <path d={ruta(colchon.recta)} clipPath={`url(#${area})`} fill="none" stroke={ACENTO} strokeWidth={2} />
        {colchon.puntos.map((p) => {
          const bajo = p.negocios_km2 < p25; // cuarto con menos negocios: medición poco confiable
          const c = bajo ? TENUE : "#f1f5f9";
          return (
            <g key={p.negocios_km2}>
              <line x1={x(p.negocios_km2)} x2={x(p.negocios_km2)} y1={y(p.ic[0])} y2={y(p.ic[1])} stroke={c} strokeOpacity={0.6} />
              <circle cx={x(p.negocios_km2)} cy={y(p.robos_km2)} r={3.4} fill={c} stroke="#0b1120" strokeWidth={1}>
                <title>{`${entero(p.negocios_km2)} negocios/km²: ${decimal(p.robos_km2)} robos al año por km² (${p.celdas} hexágonos)`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-slate-100" /> Promedio de ~{Math.round(colchon.celdas / colchon.puntos.length)} hexágonos
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-slate-600" /> Pocos negocios: dato poco confiable
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 bg-sky-400" /> Ajuste con quiebre
        </span>
        {compacta && (
          <span className="flex items-center gap-1.5">
            <span className="w-4 border-t-2 border-dashed border-slate-400" /> Si siguiera al mismo ritmo
          </span>
        )}
      </figcaption>
    </figure>
  );
}
