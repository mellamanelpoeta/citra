import type { CSSProperties, ReactNode } from "react";
import type { CategoriaRiesgo } from "../types/data";
import { COLOR_CATEGORIA, ETIQUETA_CATEGORIA } from "../lib/format";

export function InsigniaCategoria({ categoria, texto }: { categoria: CategoriaRiesgo; texto?: string }) {
  const c = COLOR_CATEGORIA[categoria];
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium" style={{ background: `${c}26`, color: c, border: `1px solid ${c}66` }}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: c }} />
      {texto ?? ETIQUETA_CATEGORIA[categoria]}
    </span>
  );
}

export function Tarjeta({ children, className = "", style }: { children: ReactNode; className?: string; style?: CSSProperties }) {
  return (
    <div className={`velo rounded-xl p-3 ${className}`} style={style}>
      {children}
    </div>
  );
}

export function Dato({ etiqueta, valor, nota }: { etiqueta: string; valor: ReactNode; nota?: ReactNode }) {
  return (
    <Tarjeta className="!p-2.5">
      <div className="text-[11px] text-slate-400">{etiqueta}</div>
      <div className="text-lg font-semibold tabular-nums text-slate-100">{valor}</div>
      {nota && <div className="text-[11px] text-slate-500">{nota}</div>}
    </Tarjeta>
  );
}

export function Seccion({ titulo, children, accion }: { titulo: string; children: ReactNode; accion?: ReactNode }) {
  return (
    <section className="mt-4">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">{titulo}</h3>
        {accion}
      </div>
      {children}
    </section>
  );
}
