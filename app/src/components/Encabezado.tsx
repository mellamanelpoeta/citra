import { useData } from "../data/context";
import { ALTO_ENCABEZADO, APP_LEMA, APP_NOMBRE } from "../lib/config";
import { useApp, type Vista } from "../lib/store";

export default function Encabezado() {
  const { vista, irA } = useApp();
  const { meta, validacion } = useData();
  // La prueba un año atrás va antes de la lista: justifica usarla (contrato v0.5).
  const PASOS: { v: Vista; t: string; d: string }[] = [
    { v: "contexto", t: "Contexto", d: "Problema y hallazgos" },
    { v: "anormal", t: "¿Qué es anormal?", d: "Del comercio al riesgo" },
    { v: "validar", t: "¿Habría funcionado?", d: `Prueba con ${validacion.anio}` },
    { v: "priorizar", t: "¿Qué priorizar?", d: `Zonas para ${meta.anticipado.anio}` },
  ];

  return (
    <header
      className="vidrio absolute inset-x-3 top-3 z-30 flex items-center gap-4 rounded-2xl px-4"
      style={{ height: ALTO_ENCABEZADO }}
    >
      <div className="flex items-center gap-2.5">
        <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" className="h-7 w-7" />
        <div className="leading-tight">
          <div className="font-semibold text-slate-50">{APP_NOMBRE}</div>
          <div className="text-[10px] text-slate-400">{APP_LEMA}</div>
        </div>
      </div>

      <nav className="ml-4 flex items-center gap-1" aria-label="Pasos del análisis">
        {PASOS.map((p, i) => {
          const activo = vista === p.v;
          return (
            <button
              key={p.v}
              onClick={() => irA(p.v)}
              aria-current={activo ? "step" : undefined}
              className={`flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-left transition-colors ${activo ? "bg-white/10 shadow-inner" : "hover:bg-white/5"}`}
            >
              <span
                className={`flex h-5 w-5 items-center justify-center rounded-full text-[11px] font-bold ${
                  activo ? "bg-accent text-ink-950 shadow-[0_0_12px_rgb(56_189_248/0.5)]" : "border border-slate-500/40 text-slate-400"
                }`}
              >
                {i + 1}
              </span>
              <span className="leading-tight">
                <span className={`block text-xs font-medium ${activo ? "text-slate-50" : "text-slate-300"}`}>{p.t}</span>
                <span className="block text-[10px] text-slate-500">{p.d}</span>
              </span>
            </button>
          );
        })}
      </nav>

      <button
        onClick={() => irA("consulta")}
        className={`ml-auto rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors ${
          vista === "consulta" ? "border-accent bg-accent text-ink-950" : "border-accent/50 bg-accent/5 text-accent hover:bg-accent/15"
        }`}
      >
        Consulta mi zona
      </button>
    </header>
  );
}
