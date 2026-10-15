import type { ReactNode } from "react";
import { decimal, decimal2 } from "../lib/format";

// Gráficas pequeñas de las pruebas de causalidad (página Contexto): ¿el robo ahuyenta negocios? Cifras escritas a mano
// desde las salidas del pipeline, igual que GraficasCifraNegra: no viajan en los datos de la app.

const ACENTO = "#38bdf8";
const GRIS = "#94a3b8";
const VIOLENCIA = "#f87171";
const REJILLA = "rgb(255 255 255 / 0.06)";

// Título de la gráfica arriba, en HTML: se ajusta al ancho en vez de cortarse como un <text> del SVG.
function Figura({ titulo, children }: { titulo: string; children: ReactNode }) {
  return (
    <figure>
      <figcaption className="mb-1.5 text-[13px] leading-snug font-medium text-slate-300">{titulo}</figcaption>
      {children}
    </figure>
  );
}

type Fila = { etiqueta: string; valor: number; lo: number; hi: number; color?: string };

// Puntos con intervalo de 95%, uno por fila, contra una línea de referencia.
function Intervalos({ filas, dominio, marcas, referencia, textoReferencia, formato, eje, aria }: {
  filas: Fila[];
  dominio: [number, number];
  marcas: number[];
  referencia: number;
  textoReferencia: string;
  formato: (v: number) => string;
  eje: string;
  aria: string;
}) {
  const W = 320;
  const fila = 20;
  const m = { l: 108, r: 14, t: 16, b: 16 };
  const H = m.t + filas.length * fila + m.b;
  const x = (v: number) => m.l + ((v - dominio[0]) / (dominio[1] - dominio[0])) * (W - m.l - m.r);
  const y = (i: number) => m.t + i * fila + fila / 2;
  return (
    <Figura titulo={eje}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={aria}>
        {marcas.map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={m.t} y2={H - m.b} stroke={v === referencia ? GRIS : REJILLA} strokeDasharray={v === referencia ? "3 2" : undefined} />
            <text x={x(v)} y={H - m.b + 10} textAnchor="middle" className="fill-slate-500 text-[8px]">
              {formato(v)}
            </text>
          </g>
        ))}
        <text x={x(referencia)} y={m.t - 5} textAnchor="middle" className="fill-slate-300 text-[8px]">
          {textoReferencia}
        </text>
        {filas.map((f, i) => {
          const color = f.color ?? ACENTO;
          return (
            <g key={f.etiqueta}>
              <text x={m.l - 5} y={y(i) + 3} textAnchor="end" className="fill-slate-400 text-[8px]">
                {f.etiqueta}
              </text>
              <line x1={x(f.lo)} x2={x(f.hi)} y1={y(i)} y2={y(i)} stroke={color} strokeOpacity={0.5} strokeWidth={1.5} />
              <circle cx={x(f.valor)} cy={y(i)} r={2.6} fill={color} />
              <text x={x(f.hi) + 4} y={y(i) + 3} className="fill-slate-200 text-[8px] font-semibold">
                {formato(f.valor)}
              </text>
            </g>
          );
        })}
      </svg>
    </Figura>
  );
}

// pipeline/outputs/resultados.md (A, especificación sin controles) y modelos/robustez_C.csv (04c, prueba 3): β con los
// negocios de nov. 2023 y con los de ene. 2016, anteriores al robo analizado. Se muestra como 2^β: cuánto se multiplica
// el robo al duplicar los negocios.
const BETA_CORTE = [
  { etiqueta: "Negocios de nov. 2023", beta: [0.877, 0.808, 0.952] },
  { etiqueta: "Negocios de ene. 2016", beta: [0.838, 0.779, 0.912] },
];

// Si el robo hubiera ahuyentado negocios, el corte de 2016 (antes del robo) daría otra curva: da casi la misma.
export function GraficaCorte() {
  const filas = BETA_CORTE.map(({ etiqueta, beta: [b, lo, hi] }, i) => ({
    etiqueta,
    valor: 2 ** b,
    lo: 2 ** lo,
    hi: 2 ** hi,
    color: i ? GRIS : ACENTO,
  }));
  return (
    <Intervalos
      filas={filas}
      dominio={[1.6, 2.05]}
      marcas={[1.6, 1.7, 1.8, 1.9, 2]}
      referencia={2}
      textoReferencia="proporcional"
      formato={(v) => `×${decimal2(v)}`}
      eje="Cuánto se multiplica el robo al duplicar los negocios (IC 95%)"
      aria="Relación entre robo y negocios con los negocios de 2023 y de 2016"
    />
  );
}

// pipeline/outputs/dinamica/rotacion_quintiles.csv (08b): % de los negocios de 2019 que salen antes de 2024 y negocios
// nuevos como % de los de 2019, por quintil de riesgo relativo previo (2016–2018), con su mediana. Tasas crudas, sin
// controles; la prueba con controles es la de GraficaFlujos.
const FLUJOS_QUINTIL = [
  { rr: 0.28, salen: 34.3, entran: 34.0 },
  { rr: 0.5, salen: 35.6, entran: 32.6 },
  { rr: 0.73, salen: 34.4, entran: 31.8 },
  { rr: 1.02, salen: 35.2, entran: 30.1 },
  { rr: 1.89, salen: 36.0, entran: 37.8 },
];

// Pares de barras por quintil: las salidas son planas; las entradas suben en las zonas con más robo previo.
export function GraficaSalidas() {
  const W = 320;
  const H = 100;
  const m = { l: 22, r: 10, t: 22, b: 18 };
  const y = (v: number) => H - m.b - (v / 40) * (H - m.t - m.b);
  const ancho = (W - m.l - m.r) / FLUJOS_QUINTIL.length;
  const series = [
    { clave: "salen" as const, etiqueta: "Salen", color: VIOLENCIA, dx: -0.19 },
    { clave: "entran" as const, etiqueta: "Entran", color: ACENTO, dx: 0.19 },
  ];
  return (
    <Figura titulo="% de los negocios de 2019 que salen o entran hacia 2024, por robo previo (2016–2018)">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Porcentaje de negocios que salen y que entran según el robo previo de la zona">
        {[0, 20, 40].map((v) => (
          <g key={v}>
            <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} stroke={REJILLA} />
            <text x={m.l - 4} y={y(v) + 2.5} textAnchor="end" className="fill-slate-500 text-[8px]">
              {v}%
            </text>
          </g>
        ))}
        {FLUJOS_QUINTIL.map((q, i) => {
          const xc = m.l + ancho * (i + 0.5);
          return (
            <g key={q.rr}>
              {series.map((s) => (
                <g key={s.clave}>
                  <rect x={xc + s.dx * ancho - ancho * 0.17} y={y(q[s.clave])} width={ancho * 0.34} height={y(0) - y(q[s.clave])} rx={2} fill={s.color} fillOpacity={0.85} />
                  <text x={xc + s.dx * ancho} y={y(q[s.clave]) - 3} textAnchor="middle" className="fill-slate-200 text-[7.5px] font-semibold">
                    {Math.round(q[s.clave])}
                  </text>
                </g>
              ))}
              <text x={xc} y={y(0) + 10} textAnchor="middle" className="fill-slate-400 text-[8px]">
                riesgo ×{decimal(q.rr)}
              </text>
            </g>
          );
        })}
        {series.map((s, i) => (
          <g key={s.clave} transform={`translate(${m.l + 4 + i * 48} 9)`}>
            <rect x={0} y={-6} width={7} height={7} rx={1.5} fill={s.color} fillOpacity={0.85} />
            <text x={10} y={0} className="fill-slate-400 text-[8px]">
              {s.etiqueta}
            </text>
          </g>
        ))}
      </svg>
    </Figura>
  );
}

// pipeline/outputs/dinamica/rotacion_D2.csv y robustez_D.csv (07, 08): razón de tasas por +1 DE de log rr_pre (riesgo
// relativo previo ×2.2), con controles y efectos fijos de alcaldía.
const FLUJOS: Fila[] = [
  { etiqueta: "Negocios que salen", valor: 0.994, lo: 0.983, hi: 1.005 },
  { etiqueta: "Negocios que entran", valor: 1.024, lo: 1.004, hi: 1.043 },
  { etiqueta: "Saldo de negocios", valor: 1.011, lo: 1.002, hi: 1.019 },
];

export function GraficaFlujos() {
  return (
    <Intervalos
      filas={FLUJOS}
      dominio={[0.96, 1.06]}
      marcas={[0.96, 0.98, 1, 1.02, 1.04, 1.06]}
      referencia={1}
      textoReferencia="igual que las demás"
      formato={decimal2}
      eje="Zonas con 2.2 veces más riesgo en 2016–2018 que las demás (IC 95%)"
      aria="Razón de salidas, entradas y saldo de negocios según el riesgo previo"
    />
  );
}

// pipeline/outputs/dinamica/rotacion_robustez.csv (08): bajas que el INEGI registró entre censos (2019-11 → 2023-11, el
// 4% de los negocios) por tamaño, misma escala que GraficaFlujos.
const BAJAS_TAMANO: Fila[] = [
  { etiqueta: "0 a 10 personas", valor: 1.041, lo: 0.992, hi: 1.1 },
  { etiqueta: "11 a 50 personas", valor: 1.075, lo: 1.033, hi: 1.127, color: VIOLENCIA },
  { etiqueta: "51 o más personas", valor: 0.974, lo: 0.908, hi: 1.024 },
];

// La única señal en contra: los negocios medianos.
export function GraficaBajasTamano() {
  return (
    <Intervalos
      filas={BAJAS_TAMANO}
      dominio={[0.9, 1.15]}
      marcas={[0.9, 0.95, 1, 1.05, 1.1, 1.15]}
      referencia={1}
      textoReferencia="igual que las demás"
      formato={decimal2}
      eje="Bajas entre censos, zonas con 2.2 veces más riesgo 2016–18 (IC 95%)"
      aria="Razón de bajas de negocios entre censos por tamaño según el riesgo previo"
    />
  );
}
