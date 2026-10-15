import { useState, type ReactNode } from "react";
import ejemplo from "../data/ejemplo_colonia.json";
import { entero, porcentaje } from "../lib/format";

const ACENTO = "#38bdf8";
// Orden de ejemplo.giros: comercio, manufactura, servicios, alto efectivo tradicional, alto efectivo formal. Sin rojo ni azul,
// que en el resto de la app significan riesgo.
const COLOR_GIRO = ["#2dd4bf", "#94a3b8", "#a78bfa", "#fbbf24", "#f472b6"];
// Radio por estrato de personal ocupado (0 a 5 … 251 y más); del 4 en adelante son 51 personas o más.
const RADIO = [1.8, 2.6, 3.4, 4.2, 5.6, 6.8, 8];
const RADIO_BASE = 2.2;
const GRANDE = 4;

// Grandes primero para que los chicos queden encima y no se tapen.
const PUNTOS = [...ejemplo.puntos].sort((a, b) => b[3] - a[3]);

const PASOS: { titulo: string; texto: ReactNode }[] = [
  { titulo: "Cuántos hay", texto: "la exposición, el dato principal." },
  {
    titulo: "De qué giro son",
    texto: (
      <>
        comercio, manufactura, servicios… y dos grupos de alto efectivo: tradicional (abarrotes, tortillerías, carnicerías, fruterías, licores) y formal
        (minisúpers, farmacias, gasolineras, restaurantes, bares, casas de empeño).
      </>
    ),
  },
  { titulo: "De qué tamaño", texto: "qué proporción tiene 51 personas o más." },
];

// Las tres formas de describir la actividad económica, ilustradas con los establecimientos reales de una colonia: el
// mismo mapa de puntos se colorea por giro y luego se escala por tamaño. SVG propio, como GraficaColchon. Pasos a la
// izquierda y mapa a la derecha en pantallas medianas; `nota` va debajo de los pasos.
export default function EjemploActividad({ nota }: { nota?: ReactNode }) {
  const [paso, setPaso] = useState(0);
  const n = ejemplo.puntos.length;

  return (
    <div className="grid gap-x-6 md:grid-cols-2 md:items-start">
      <div>
        <ol className="mt-2 space-y-1 text-sm text-slate-300">
          {PASOS.map((p, i) => (
            <li key={p.titulo}>
              <button
                onClick={() => setPaso(i)}
                aria-pressed={paso === i}
                className={`flex w-full gap-2 rounded-lg px-2 py-1 text-left transition-colors ${
                  paso === i ? "bg-accent/10 ring-1 ring-accent/40" : "opacity-60 hover:opacity-100"
                }`}
              >
                <span
                  className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold ${
                    paso === i ? "bg-accent text-slate-900" : "bg-white/10 text-slate-300"
                  }`}
                >
                  {i + 1}
                </span>
                <span>
                  <b className="text-slate-100">{p.titulo}:</b> {p.texto}
                </span>
              </button>
            </li>
          ))}
        </ol>
        {nota}
      </div>

      <figure className="mt-3 md:mt-2 rounded-lg border border-white/10 bg-black/20 p-3">
        <div className="flex items-baseline justify-between gap-2 text-[11px]">
          <span className="font-medium text-slate-300">
            Ejemplo: {ejemplo.colonia}, {ejemplo.alcaldia}
          </span>
          <span className="text-slate-500">DENUE {ejemplo.corte}</span>
        </div>
        <svg viewBox={`0 0 ${ejemplo.ancho} ${ejemplo.alto}`} className="mt-2 w-full" role="img" aria-label={`Establecimientos de ${ejemplo.colonia}`}>
          <path d={ejemplo.poligono} fill="rgb(255 255 255 / 0.03)" stroke="rgb(255 255 255 / 0.35)" strokeWidth={1} strokeLinejoin="round" />
          {PUNTOS.map(([x, y, g, e], i) => {
            const r = paso === 2 ? RADIO[e] : RADIO_BASE;
            const color = paso === 0 ? ACENTO : COLOR_GIRO[g];
            return (
              <circle
                key={i}
                r={1}
                vectorEffect="non-scaling-stroke"
                style={{
                  transform: `translate(${x}px, ${y}px) scale(${r})`,
                  fill: color,
                  fillOpacity: paso === 2 ? 0.7 : 0.85,
                  stroke: paso === 2 && e >= GRANDE ? "#f8fafc" : "rgb(0 0 0 / 0.4)",
                  strokeWidth: paso === 2 && e >= GRANDE ? 1.2 : 0.5,
                  transition: "transform 600ms ease, fill 600ms ease, fill-opacity 600ms ease",
                }}
              />
            );
          })}
        </svg>

        <figcaption className="mt-2 min-h-[3.5rem] text-xs text-slate-400">
          {paso === 0 && (
            <p>
              <span className="text-lg font-semibold tabular-nums text-slate-50">{entero(n)}</span> establecimientos en{" "}
              {ejemplo.area_km2.toLocaleString("es-MX")} km². Cada punto es uno; este número es lo que el modelo usa como exposición.
            </p>
          )}
          {paso === 1 && (
            <ul className="flex flex-wrap gap-x-3 gap-y-1">
              {ejemplo.giros.map((g, i) => (
                <li key={g.id} className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: COLOR_GIRO[i] }} />
                  <span className="text-slate-300">{g.nombre}</span>
                  <span className="tabular-nums text-slate-500">{entero(g.n)}</span>
                </li>
              ))}
            </ul>
          )}
          {paso === 2 && (
            <div className="space-y-1.5">
              <div className="flex items-end gap-2">
                <span>Personas:</span>
                {ejemplo.estratos.map((s, i) => (
                  <span key={s.etiqueta} className="flex flex-col items-center" title={`${s.etiqueta} personas: ${s.n}`}>
                    <svg width={RADIO[i] * 2 + 3} height={RADIO[i] * 2 + 3}>
                      <circle
                        cx={RADIO[i] + 1.5}
                        cy={RADIO[i] + 1.5}
                        r={RADIO[i]}
                        fill="#cbd5e1"
                        fillOpacity={0.6}
                        stroke={i >= GRANDE ? "#f8fafc" : "none"}
                        strokeWidth={1.2}
                      />
                    </svg>
                  </span>
                ))}
                <span className="text-slate-500">de 0–5 a 251 y más</span>
              </div>
              <p>
                <span className="font-semibold tabular-nums text-slate-100">
                  {entero(ejemplo.n_grandes)} ({porcentaje(ejemplo.n_grandes / n)})
                </span>{" "}
                tienen 51 personas o más (con anillo blanco).
              </p>
            </div>
          )}
        </figcaption>

        <div className="mt-2 flex justify-end gap-3 text-xs font-medium">
          {paso > 0 && (
            <button onClick={() => setPaso(paso - 1)} className="text-slate-400 hover:text-slate-200">
              ← Anterior
            </button>
          )}
          {paso < PASOS.length - 1 ? (
            <button onClick={() => setPaso(paso + 1)} className="text-accent hover:underline">
              Siguiente: {PASOS[paso + 1].titulo.toLowerCase()} →
            </button>
          ) : (
            <button onClick={() => setPaso(0)} className="text-accent hover:underline">
              Volver a empezar ↺
            </button>
          )}
        </div>
      </figure>
    </div>
  );
}
