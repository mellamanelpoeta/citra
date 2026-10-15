import { useMemo, useState } from "react";
import { useData } from "../data/context";
import { COLOR_CATEGORIA, MENSAJE_LLANO, nombrePropio, sinAcentos } from "../lib/format";
import { useApp } from "../lib/store";
import type { CategoriaRiesgo } from "../types/data";
import { InsigniaCategoria, Seccion, Tarjeta } from "./ui";

const NIVELES: CategoriaRiesgo[] = ["por_debajo", "dentro", "por_encima"];
const ETIQUETA_NIVEL: Record<CategoriaRiesgo, string> = { por_debajo: "Menos", dentro: "Lo esperado", por_encima: "Más" };

// Vista del pequeño negocio: lenguaje llano y sin números técnicos.
export default function PanelConsulta() {
  const { colonias, coloniaPorId, giros } = useData();
  const { coloniaSel, giroSel, seleccionarColonia, setGiro } = useApp();
  const [q, setQ] = useState("");

  // Índice de búsqueda sin acentos ni mayúsculas (sin geocodificación externa).
  const indice = useMemo(() => colonias.map((c) => ({ c, clave: sinAcentos(`${c.nombre} ${c.alcaldia}`) })), [colonias]);
  const resultados = useMemo(() => {
    const t = sinAcentos(q.trim());
    if (t.length < 2) return [];
    const palabras = t.split(/\s+/);
    // Primero las que contienen cada palabra completa ("seccion ii" antes que "seccion iii"), luego las que empiezan igual.
    const orden = (clave: string) => {
      const enteras = new Set(clave.split(/\s+/));
      return (palabras.every((p) => enteras.has(p)) ? 0 : 2) + (clave.startsWith(t) ? 0 : 1);
    };
    return indice
      .filter((x) => palabras.every((p) => x.clave.includes(p)))
      .sort((a, b) => orden(a.clave) - orden(b.clave))
      .slice(0, 8)
      .map((x) => x.c);
  }, [indice, q]);

  const col = coloniaSel ? coloniaPorId.get(coloniaSel) : undefined;
  const giro = giros.find((g) => g.id === giroSel);
  const elegir = (id: string) => {
    seleccionarColonia(id, coloniaPorId.get(id)?.bbox);
    setQ("");
  };

  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-accent">Consulta mi zona</div>
      <h2 className="mt-1 text-xl font-semibold leading-snug text-slate-50">¿Cómo está el robo a negocios en tu colonia?</h2>

      <div className="relative mt-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Escribe el nombre de tu colonia…"
          className="velo w-full rounded-lg px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-accent focus:outline-none"
          aria-label="Buscar colonia"
        />
        {resultados.length > 0 && (
          <ul className="vidrio absolute z-20 mt-1 w-full overflow-hidden rounded-lg !bg-ink-900/90">
            {resultados.map((c) => (
              <li key={c.cve_col}>
                <button onClick={() => elegir(c.cve_col)} className="w-full px-3 py-2 text-left hover:bg-white/10">
                  <span className="text-sm text-slate-100">{nombrePropio(c.nombre)}</span>
                  <span className="ml-2 text-xs text-slate-500">{c.alcaldia}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
        {q.trim().length >= 2 && resultados.length === 0 && (
          <p className="mt-1 text-xs text-slate-500">No encontramos esa colonia. Prueba con otra forma de escribirla.</p>
        )}
      </div>

      <label className="mt-3 block text-xs text-slate-400">
        ¿Qué tipo de negocio tienes?
        <select
          value={giroSel ?? ""}
          onChange={(e) => setGiro(e.target.value || null)}
          className="velo mt-1 w-full rounded-lg px-3 py-2 text-sm text-slate-100 [&>option]:bg-ink-900"
        >
          <option value="">Elige tu giro (opcional)</option>
          {giros.map((g) => (
            <option key={g.id} value={g.id}>
              {g.nombre}
            </option>
          ))}
        </select>
      </label>

      {!col && <p className="mt-4 text-sm text-slate-500">También puedes hacer clic en el mapa sobre tu zona.</p>}

      {col && (
        <>
          <Tarjeta className="mt-4">
            <div className="text-[11px] text-slate-500">{col.alcaldia}</div>
            <div className="text-lg font-semibold text-slate-50">{nombrePropio(col.nombre)}</div>
            <p className="mt-2 text-base font-medium leading-snug" style={{ color: COLOR_CATEGORIA[col.categoria] }}>
              {MENSAJE_LLANO[col.categoria].titulo}
            </p>
            <p className="mt-1 text-sm text-slate-400">{MENSAJE_LLANO[col.categoria].detalle}</p>
            {/* Escala visual de tres niveles, sin cifras */}
            <div className="mt-3 grid grid-cols-3 gap-1" aria-hidden>
              {NIVELES.map((n) => (
                <div key={n} className="text-center">
                  <div className="h-2 rounded-full" style={{ background: n === col.categoria ? COLOR_CATEGORIA[n] : "rgb(255 255 255 / 0.08)" }} />
                  <div className={`mt-1 text-[11px] ${n === col.categoria ? "font-semibold text-slate-200" : "text-slate-500"}`}>{ETIQUETA_NIVEL[n]}</div>
                </div>
              ))}
            </div>
          </Tarjeta>

          {giro && (
            <Tarjeta className="mt-2">
              <div className="text-[11px] text-slate-500">Para tu giro: {giro.nombre}</div>
              <p className="mt-1 text-sm text-slate-200">{giro.mensaje}</p>
            </Tarjeta>
          )}

          <Seccion titulo="Colonias parecidas a la tuya">
            <p className="-mt-1 mb-2 text-[11px] text-slate-500">Con un número y tipo de negocios similares.</p>
            <ul className="space-y-1">
              {col.similares
                .map((id) => coloniaPorId.get(id))
                .filter((c) => c !== undefined)
                .map((c) => (
                  <li key={c.cve_col}>
                    <button onClick={() => elegir(c.cve_col)} className="flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-white/5">
                      <span className="min-w-0">
                        <span className="block truncate text-sm text-slate-200">{nombrePropio(c.nombre)}</span>
                        <span className="block text-[11px] text-slate-500">{c.alcaldia}</span>
                      </span>
                      <InsigniaCategoria categoria={c.categoria} texto={ETIQUETA_NIVEL[c.categoria]} />
                    </button>
                  </li>
                ))}
            </ul>
          </Seccion>
        </>
      )}

      <p className="mt-4 border-t border-white/10 pt-3 text-[11px] leading-relaxed text-slate-500">
        Esta información se basa solo en robos denunciados ante la Fiscalía. Muchos robos a negocio no se denuncian, sobre todo en negocios
        pequeños, así que la cifra real es mayor en todas las zonas.
      </p>
    </div>
  );
}
