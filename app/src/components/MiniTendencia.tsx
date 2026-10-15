import { decimal, entero } from "../lib/format";

interface Punto {
  anio: string;
  observados: number;
  esperados: number;
}

// Prueba (v0.6): el año que el modelo no vio. Va después de una línea de corte y se pinta según lo que pasó.
interface Prueba extends Punto {
  mas_de_lo_esperado: boolean;
}

const NARANJA = "#f97316";
const GRIS = "#94a3b8";

// Observados vs. esperados por año y, en la prueba, el año evaluado al final. SVG propio: 5 puntos no justifican una
// librería de gráficas.
export default function MiniTendencia({ puntos, resaltar, prueba }: { puntos: Punto[]; resaltar?: string; prueba?: Prueba }) {
  const W = 320;
  const H = 110;
  const m = { l: 30, r: 16, t: 10, b: 20 };
  const todos = prueba ? [...puntos, prueba] : puntos;
  const max = Math.max(1, ...todos.flatMap((p) => [p.observados, p.esperados]));
  const x = (i: number) => m.l + (i * (W - m.l - m.r)) / Math.max(1, todos.length - 1);
  const y = (v: number) => m.t + (1 - v / max) * (H - m.t - m.b);
  const linea = (k: "observados" | "esperados") => puntos.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p[k])}`).join("");
  const u = puntos.length - 1;
  const xp = x(puntos.length);
  const colorPrueba = prueba?.mas_de_lo_esperado ? NARANJA : "#38bdf8";

  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Robos observados y esperados por año">
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={m.l} x2={W - m.r} y1={y(max * f)} y2={y(max * f)} stroke="rgb(255 255 255 / 0.08)" />
            <text x={m.l - 4} y={y(max * f) + 3} textAnchor="end" className="fill-slate-500 text-[9px]">
              {entero(max * f)}
            </text>
          </g>
        ))}
        {puntos.map((p, i) => (
          <g key={p.anio}>
            {p.anio === resaltar && <rect x={x(i) - 12} y={m.t - 4} width={24} height={H - m.t - m.b + 8} rx={3} fill="#38bdf81a" />}
            <text x={x(i)} y={H - 5} textAnchor="middle" className={`text-[9px] ${p.anio === resaltar ? "fill-sky-300" : "fill-slate-500"}`}>
              {p.anio}
            </text>
          </g>
        ))}
        {prueba && (
          <g>
            <rect x={xp - 13} y={m.t - 4} width={26} height={H - m.t - m.b + 8} rx={3} fill="#38bdf81a" />
            <line x1={(x(u) + xp) / 2} x2={(x(u) + xp) / 2} y1={m.t} y2={H - m.b} stroke="rgb(255 255 255 / 0.18)" strokeDasharray="2 2" />
            <text x={xp} y={H - 5} textAnchor="middle" className="fill-sky-300 text-[9px]">
              {prueba.anio}
            </text>
          </g>
        )}
        <path d={linea("esperados")} fill="none" stroke={GRIS} strokeWidth={1.5} strokeDasharray="4 3" />
        <path d={linea("observados")} fill="none" stroke={NARANJA} strokeWidth={2} />
        {puntos.map((p, i) => (
          <circle key={p.anio} cx={x(i)} cy={y(p.observados)} r={2.6} fill={NARANJA} />
        ))}
        {prueba && (
          <g>
            <path d={`M${x(u)},${y(puntos[u].esperados)}L${xp},${y(prueba.esperados)}`} stroke={GRIS} strokeWidth={1.5} strokeDasharray="4 3" />
            <path d={`M${x(u)},${y(puntos[u].observados)}L${xp},${y(prueba.observados)}`} stroke={NARANJA} strokeWidth={1.5} strokeDasharray="2 3" />
            <circle cx={xp} cy={y(prueba.esperados)} r={2.4} fill={GRIS} />
            <circle cx={xp} cy={y(prueba.observados)} r={3.6} fill={colorPrueba} stroke="#0b1120" strokeWidth={1.2}>
              <title>{`${prueba.anio}: ${entero(prueba.observados)} robos, ${decimal(prueba.esperados)} esperados por su actividad`}</title>
            </circle>
          </g>
        )}
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-400">
        <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-orange-500" /> Observados</span>
        <span className="flex items-center gap-1.5"><span className="h-0 w-4 border-t-2 border-dashed border-slate-400" /> Esperados por su actividad</span>
        {prueba && <span className="text-slate-500">El modelo no vio {prueba.anio}</span>}
      </figcaption>
    </figure>
  );
}
