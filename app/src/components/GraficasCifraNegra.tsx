import { decimal, decimal2 } from "../lib/format";

// Gráficas pequeñas de las pruebas de cifra negra (página Contexto). Cifras escritas a mano desde las salidas del
// pipeline, igual que ANALISIS en PaginaContexto: no viajan en los datos de la app. SVG propio, como GraficaColchon.

const ACENTO = "#38bdf8";
const GRIS = "#94a3b8";
const VIOLENCIA = "#f87171";
const REJILLA = "rgb(255 255 255 / 0.06)";

// pipeline/outputs/modelos/cifra_negra_alcaldia.md (04e): % de delitos que los hogares denuncian ante el MP por alcaldía
// de ocurrencia, ENVIPE 2021–2025, con su error estándar (puntos porcentuales).
const DENUNCIA_ALCALDIA: [string, number, number][] = [
  ["Azcapotzalco", 9.0, 1.2],
  ["Magdalena Contreras", 9.0, 1.4],
  ["Tláhuac", 9.3, 1.5],
  ["Venustiano Carranza", 9.5, 1.2],
  ["Iztapalapa", 9.6, 0.8],
  ["Álvaro Obregón", 9.7, 1.3],
  ["Miguel Hidalgo", 9.7, 1.3],
  ["Cuajimalpa", 9.8, 2.2],
  ["Gustavo A. Madero", 10.2, 1.5],
  ["Milpa Alta", 10.2, 2.9],
  ["Cuauhtémoc", 10.5, 1.0],
  ["Xochimilco", 10.8, 1.4],
  ["Benito Juárez", 11.2, 1.5],
  ["Iztacalco", 11.4, 1.4],
  ["Coyoacán", 13.5, 1.3],
  ["Tlalpan", 13.5, 1.6],
];
const DENUNCIA_CDMX = 10.4;

// Puntos con intervalo de 95% por alcaldía y la media de la ciudad: los intervalos se enciman.
export function GraficaAlcaldia() {
  const W = 320;
  const fila = 11;
  const m = { l: 108, r: 10, t: 14, b: 24 };
  const H = m.t + DENUNCIA_ALCALDIA.length * fila + m.b;
  const x = (v: number) => m.l + (v / 20) * (W - m.l - m.r);
  const y = (i: number) => m.t + (DENUNCIA_ALCALDIA.length - 1 - i) * fila + fila / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Porcentaje de delitos denunciados por alcaldía">
      {[0, 5, 10, 15, 20].map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={m.t} y2={H - m.b} stroke={REJILLA} />
          <text x={x(v)} y={H - m.b + 10} textAnchor="middle" className="fill-slate-500 text-[8px]">
            {v}%
          </text>
        </g>
      ))}
      <line x1={x(DENUNCIA_CDMX)} x2={x(DENUNCIA_CDMX)} y1={m.t} y2={H - m.b} stroke={GRIS} strokeDasharray="3 2" />
      {DENUNCIA_ALCALDIA.map(([nombre, d, ee], i) => (
        <g key={nombre}>
          <text x={m.l - 5} y={y(i) + 2.5} textAnchor="end" className="fill-slate-400 text-[7.5px]">
            {nombre}
          </text>
          <line x1={x(d - 1.96 * ee)} x2={x(d + 1.96 * ee)} y1={y(i)} y2={y(i)} stroke={ACENTO} strokeOpacity={0.45} strokeWidth={1.5} />
          <circle cx={x(d)} cy={y(i)} r={2.3} fill={ACENTO} />
        </g>
      ))}
      <text x={x(DENUNCIA_CDMX)} y={m.t - 5} textAnchor="middle" className="fill-slate-300 text-[8px]">
        CDMX {decimal(DENUNCIA_CDMX)}%
      </text>
      <text x={(m.l + W - m.r) / 2} y={H - 3} textAnchor="middle" className="fill-slate-500 text-[8px]">
        % de delitos denunciados (IC 95%)
      </text>
    </svg>
  );
}

// pipeline/outputs/modelos/sesgo_denuncia_quintiles.csv (04b): AGEB en quintiles por número de negocios (DENUE 2023-11).
// observada: denuncia esperada por la mezcla de tamaños (denuncia de la ENVE por tamaño, ponderada por los negocios de cada
// tamaño). necesaria: la que haría falta para rechazar A (cae 3.3% por duplicación), anclada en el quintil central.
const DENUNCIA_QUINTIL = [
  { negocios: 23, observada: 11.6, necesaria: 12.5 },
  { negocios: 83, observada: 11.5, necesaria: 11.8 },
  { negocios: 142, observada: 11.5, necesaria: 11.5 },
  { negocios: 225, observada: 11.4, necesaria: 11.2 },
  { negocios: 381, observada: 11.5, necesaria: 10.9 },
];

// Dos líneas por quintil: la denuncia observada (plana) y la que tendría que haber para sesgar el resultado (cae). Se
// cruzan en el quintil central, donde se ancla la necesaria.
export function GraficaTamano() {
  const W = 320;
  const H = 132;
  const m = { l: 30, r: 16, t: 24, b: 30 };
  const [y0, y1] = [10.5, 12.8];
  const y = (v: number) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const ancho = (W - m.l - m.r) / DENUNCIA_QUINTIL.length;
  const x = (i: number) => m.l + ancho * (i + 0.5);
  const linea = (clave: "observada" | "necesaria") =>
    DENUNCIA_QUINTIL.map((q, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(q[clave])}`).join(" ");
  const ultimo = DENUNCIA_QUINTIL.length - 1;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Denuncia esperada observada frente a la que sesgaría el resultado">
      {[11, 12].map((v) => (
        <g key={v}>
          <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} stroke={REJILLA} />
          <text x={m.l - 4} y={y(v) + 2.5} textAnchor="end" className="fill-slate-500 text-[8px]">
            {v}%
          </text>
        </g>
      ))}
      <path d={linea("necesaria")} fill="none" stroke={GRIS} strokeWidth={1.5} strokeDasharray="4 3" />
      <path d={linea("observada")} fill="none" stroke={ACENTO} strokeWidth={2} />
      {DENUNCIA_QUINTIL.map((q, i) => (
        <g key={q.negocios}>
          <circle cx={x(i)} cy={y(q.necesaria)} r={2.3} fill="#0b1120" stroke={GRIS} strokeWidth={1.2} />
          <circle cx={x(i)} cy={y(q.observada)} r={2.6} fill={ACENTO} />
          {/* la etiqueta de la observada va del lado contrario a la necesaria para no encimarse */}
          <text x={x(i)} y={y(q.observada) + (q.necesaria > q.observada ? 11 : -6)} textAnchor="middle"
            className="fill-sky-300 text-[8px] font-semibold">
            {decimal(q.observada)}
          </text>
          <text x={x(i)} y={H - m.b + 11} textAnchor="middle" className="fill-slate-400 text-[8px]">
            ~{q.negocios}
          </text>
        </g>
      ))}
      <text x={x(0) + 7} y={y(DENUNCIA_QUINTIL[0].necesaria) + 3} className="fill-slate-300 text-[8px]">
        {decimal(DENUNCIA_QUINTIL[0].necesaria)}
      </text>
      <text x={x(ultimo)} y={y(DENUNCIA_QUINTIL[ultimo].necesaria) + 11} textAnchor="middle" className="fill-slate-300 text-[8px]">
        {decimal(DENUNCIA_QUINTIL[ultimo].necesaria)}
      </text>
      <g transform={`translate(${m.l + 4} 9)`}>
        <line x1={0} x2={12} y1={-2.5} y2={-2.5} stroke={ACENTO} strokeWidth={2} />
        <text x={16} y={0} className="fill-slate-400 text-[8px]">
          Observada
        </text>
        <line x1={64} x2={76} y1={-2.5} y2={-2.5} stroke={GRIS} strokeWidth={1.5} strokeDasharray="4 3" />
        <text x={80} y={0} className="fill-slate-400 text-[8px]">
          La que sesgaría el resultado
        </text>
      </g>
      <text x={(m.l + W - m.r) / 2} y={H - 4} textAnchor="middle" className="fill-slate-500 text-[8px]">
        Negocios por zona (quintiles) · % de delitos denunciados esperado
      </text>
    </svg>
  );
}

// pipeline/outputs/modelos/robustez_C.csv (04c): correlación de rangos (Spearman) del riesgo de cada AGEB con el de todo
// el robo (V2b). Con violencia: rho_rr de "solo con violencia". Ruido: mediana de 200 réplicas que conservan al azar el 40%
// de las carpetas de V2b (la fracción con violencia), P5–P95 [0.81, 0.83].
const CORRELACION = [
  { etiqueta: "Solo con violencia", valor: 0.75, color: VIOLENCIA },
  { etiqueta: "40% de los robos al azar", valor: 0.82, color: GRIS },
];

export function GraficaTipoRobo() {
  const W = 320;
  const H = 82;
  const m = { l: 108, r: 34, t: 12 };
  const x = (v: number) => m.l + v * (W - m.l - m.r);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Correlación del orden por riesgo con el de todo el robo">
      <line x1={x(1)} x2={x(1)} y1={m.t - 4} y2={m.t + 44} stroke={GRIS} strokeDasharray="3 2" />
      <text x={x(1)} y={m.t - 6} textAnchor="middle" className="fill-slate-500 text-[7.5px]">
        idéntico
      </text>
      {CORRELACION.map((c, i) => (
        <g key={c.etiqueta}>
          <text x={m.l - 5} y={m.t + 8 + i * 22 + 3} textAnchor="end" className="fill-slate-400 text-[8px]">
            {c.etiqueta}
          </text>
          <rect x={m.l} y={m.t + 8 + i * 22 - 7} width={x(c.valor) - m.l} height={14} rx={2} fill={c.color} fillOpacity={0.85} />
          <text x={x(c.valor) - 4} y={m.t + 8 + i * 22 + 3} textAnchor="end" className="fill-slate-900 text-[9px] font-semibold">
            {decimal2(c.valor)}
          </text>
        </g>
      ))}
      <text x={W / 2} y={H - 4} textAnchor="middle" className="fill-slate-500 text-[8px]">
        Correlación con el orden de las zonas por riesgo de todo el robo
      </text>
    </svg>
  );
}

// pipeline/outputs/modelos/giros_violencia.csv (04h; los dos de efectivo coinciden con robustez_C.csv de 04c): razón de
// tasas por +10 pp de cada grupo, con y sin violencia (IC 95%), frente a servicios profesionales y financieros (= 1). Solo
// grupos donde +10 pp cabe en los datos (rango P10–P90 entre AGEB ≥ 10 puntos): fuera mayoreo, hoteles y "otros".
const IRR_GIRO = [
  { grupo: "Alto efectivo formal", efectivo: true, con: [1.45, 1.22, 1.58], sin: [1.21, 1.02, 1.34] },
  { grupo: "Alto efectivo tradicional", efectivo: true, con: [0.99, 0.83, 1.07], sin: [0.68, 0.59, 0.73] },
  { grupo: "Comercio al por menor", efectivo: false, con: [1.04, 0.86, 1.17], sin: [1.0, 0.83, 1.14] },
  { grupo: "Servicios personales", efectivo: false, con: [1.44, 1.16, 1.59], sin: [0.88, 0.74, 1.01] },
  { grupo: "Salud y educación", efectivo: false, con: [1.2, 1.04, 1.28], sin: [0.94, 0.8, 1.04] },
  { grupo: "Manufactura", efectivo: false, con: [1.2, 1.01, 1.33], sin: [0.84, 0.68, 0.95] },
];

// Con violencia frente a sin violencia (depende de denunciar), contra la línea de 1. Los dos
// grupos de efectivo arriba; el resto del tejido económico debajo de una línea, como comparación.
export function GraficaGiro() {
  const W = 320;
  const fila = 22;
  const separacion = 8;
  const m = { l: 108, r: 14, t: 18, b: 26 };
  const H = m.t + IRR_GIRO.length * fila + separacion + m.b;
  const [x0, x1] = [0.5, 1.7];
  const x = (v: number) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const yc = (i: number) => m.t + fila * (i + 0.5) + (i >= 2 ? separacion : 0);
  const series = [
    { clave: "con" as const, etiqueta: "Con violencia", color: VIOLENCIA, dy: -4 },
    { clave: "sin" as const, etiqueta: "Sin violencia", color: ACENTO, dy: 4 },
  ];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Razón de tasas por giro, con y sin violencia">
      {[0.6, 0.8, 1, 1.2, 1.4, 1.6].map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={m.t} y2={H - m.b} stroke={v === 1 ? GRIS : REJILLA} strokeDasharray={v === 1 ? "3 2" : undefined} />
          <text x={x(v)} y={H - m.b + 10} textAnchor="middle" className="fill-slate-500 text-[8px]">
            {decimal(v)}
          </text>
        </g>
      ))}
      <line x1={8} x2={W - m.r} y1={m.t + 2 * fila + separacion / 2} y2={m.t + 2 * fila + separacion / 2} stroke="rgb(255 255 255 / 0.12)" />
      {IRR_GIRO.map((g, i) => (
        <g key={g.grupo}>
          <text x={m.l - 5} y={yc(i) + 3} textAnchor="end" className={`text-[8px] ${g.efectivo ? "fill-slate-200 font-semibold" : "fill-slate-400"}`}>
            {g.grupo}
          </text>
          {series.map((s) => {
            const [v, lo, hi] = g[s.clave];
            return (
              <g key={s.clave} opacity={g.efectivo ? 1 : 0.7}>
                <line x1={x(lo)} x2={x(hi)} y1={yc(i) + s.dy} y2={yc(i) + s.dy} stroke={s.color} strokeOpacity={0.5} strokeWidth={1.5} />
                <circle cx={x(v)} cy={yc(i) + s.dy} r={2.4} fill={s.color} />
              </g>
            );
          })}
        </g>
      ))}
      {series.map((s, i) => (
        <g key={s.clave} transform={`translate(${m.l + i * 72} 9)`}>
          <circle cx={3} cy={-3} r={2.6} fill={s.color} />
          <text x={9} y={0} className="fill-slate-400 text-[8px]">
            {s.etiqueta}
          </text>
        </g>
      ))}
      <text x={W / 2} y={H - 3} textAnchor="middle" className="fill-slate-500 text-[8px]">
        Robo por +10 de cada 100 negocios (1 = igual que servicios profesionales)
      </text>
    </svg>
  );
}
