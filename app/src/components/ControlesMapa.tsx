import { useData } from "../data/context";
import { useResumen } from "../lib/derivados";
import {
  COLOR_ACIERTO,
  COLOR_FALLO,
  COLOR_NUEVO,
  COLOR_RURAL,
  COLOR_SIN_DATOS,
  MAPA_CATEGORIA,
  NIVEL_PLURAL,
  RAYA_RURAL,
  pctPequeno,
  porcentaje,
} from "../lib/format";
import { PASOS_PRUEBA, useApp, type Nivel } from "../lib/store";

// Unidad del mapa: alcaldía, colonia o AGEB. En "Consulta mi zona" siempre es colonia.
export function SelectorNivel() {
  const { nivel, setNivel, vista } = useApp();
  // La prueba con el año anterior es por AGEB, como la lista.
  if (vista === "consulta" || vista === "validar") return null;
  const opciones: { v: Nivel; t: string }[] = [
    { v: "alcaldia", t: "Alcaldía" },
    { v: "colonia", t: "Colonia" },
    { v: "ageb", t: "AGEB" },
  ];
  return (
    <div className="vidrio flex items-center gap-1 rounded-xl p-1" role="radiogroup" aria-label="Unidad del mapa">
      <span className="px-1.5 text-[11px] text-slate-400">Ver por</span>
      {opciones.map((o) => (
        <button
          key={o.v}
          role="radio"
          aria-checked={nivel === o.v}
          onClick={() => setNivel(o.v)}
          className={`rounded-lg px-2.5 py-1 text-xs transition-colors ${nivel === o.v ? "bg-white/15 font-semibold text-slate-100" : "text-slate-400 hover:bg-white/5"}`}
        >
          {o.t}
        </button>
      ))}
    </div>
  );
}

// Cuántas zonas cambian al ajustar: hace explícita la hipótesis C.
export function AvisoCambio() {
  const { modo, resaltarCambios, setResaltarCambios, vista, nivel } = useApp();
  const { nTop, soloRiesgo } = useResumen();
  const { validacion } = useData();
  if (vista === "consulta") return null;
  if (vista === "validar") {
    const l = modo === "conteo" ? validacion.resumen.lista_conteo : validacion.resumen.lista_riesgo;
    return (
      <div className="vidrio max-w-xs rounded-xl px-3 py-2 text-xs text-slate-300">
        De las <b className="text-slate-100">{l.n}</b> zonas de la lista por {modo === "conteo" ? "conteo" : "riesgo"} (el{" "}
        {porcentaje(l.n / validacion.zonas.length)} de la ciudad),{" "}
        <b style={{ color: COLOR_ACIERTO }}>{porcentaje(l.pct_mas_de_lo_esperado)}</b> tuvo más robos de los esperados en {validacion.anio}{" "}
        (toda la ciudad: {porcentaje(validacion.resumen.ciudad.pct_mas_de_lo_esperado)}).
      </div>
    );
  }
  // El top 10% y la comparación con el conteo (hipótesis C) son por AGEB: en los niveles agregados solo se explica la suma.
  if (nivel !== "ageb")
    return (
      <div className="vidrio max-w-xs rounded-xl px-3 py-2 text-xs text-slate-300">
        Cada {nivel === "alcaldia" ? "alcaldía" : "colonia"} suma los robos y lo esperado de sus AGEB urbanas
        {nivel === "colonia" && " (repartidos según el área que comparten)"}. Al agregar se diluyen los focos: para priorizar, baja a{" "}
        <b className="text-slate-100">AGEB</b>.
      </div>
    );
  // Comercio y conteo solo se ven en "¿Qué es anormal?", que no dibuja el contorno del top: no hay nada que explicar.
  if (modo !== "riesgo") return null;
  return (
    <div className="vidrio max-w-xs rounded-xl px-3 py-2 text-xs">
      <p className="text-slate-300">
        <b style={{ color: COLOR_NUEVO }}>{soloRiesgo.size}</b> zonas prioritarias con más robos de lo esperado{" "}
        <b className="text-slate-100">no aparecen</b> entre las <b className="text-slate-100">{nTop}</b> con más robos.
      </p>
      <label className="mt-1.5 flex cursor-pointer items-center gap-2 text-slate-400">
        <input type="checkbox" checked={resaltarCambios} onChange={(e) => setResaltarCambios(e.target.checked)} className="accent-cyan-400" />
        Resaltar las diferencias con el conteo
      </label>
    </div>
  );
}

function Muestra({ color, borde, punteado, rayado }: { color?: string; borde?: string; punteado?: boolean; rayado?: boolean }) {
  return (
    <span
      className="inline-block h-3 w-4 shrink-0 rounded-sm"
      style={{
        background: rayado
          ? `repeating-linear-gradient(135deg, ${COLOR_RURAL} 0 3px, ${RAYA_RURAL} 3px 4px)`
          : (color ?? "transparent"),
        border: borde ? `2px ${punteado ? "dashed" : "solid"} ${borde}` : undefined,
      }}
    />
  );
}

export function Leyenda() {
  const { meta, validacion } = useData();
  const { modo, escala, setEscala, resaltarCambios, pasoPrueba, vista, nivel } = useApp();
  const destacar = resaltarCambios && modo === "riesgo" && vista !== "consulta" && vista !== "validar" && nivel === "ageb";
  if (vista === "validar")
    return (
      <div className="vidrio w-60 rounded-xl p-3 text-xs">
        <div className="mb-2 font-semibold text-slate-200">
          Prueba {validacion.anio}: lista por {modo === "conteo" ? "conteo" : "riesgo"}
        </div>
        <ul className="space-y-1 text-slate-300">
          <li className="flex items-center gap-2"><Muestra color={COLOR_ACIERTO} /> Tuvo más robos de lo esperado</li>
          <li className="flex items-center gap-2"><Muestra color={COLOR_FALLO} /> No superó lo esperado</li>
          <li className="flex items-center gap-2"><Muestra color={COLOR_SIN_DATOS} /> Fuera de la lista</li>
          {PASOS_PRUEBA[pasoPrueba].nuevas && modo === "riesgo" && (
            <li className="flex items-center gap-2"><Muestra borde={COLOR_NUEVO} /> Mayor riesgo que el conteo no veía</li>
          )}
        </ul>
        <p className="mt-2 text-slate-500">Lista armada con {validacion.datos}, sin ver {validacion.anio}.</p>
      </div>
    );
  return (
    <div className="vidrio w-60 rounded-xl p-3 text-xs">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-semibold text-slate-200">
          {modo === "comercio" ? "Establecimientos (DENUE 2023)" : modo === "conteo" ? "Robos a negocio denunciados" : "Riesgo ajustado"}
        </span>
        {modo === "riesgo" && vista !== "consulta" && (
          <div className="flex overflow-hidden rounded border border-white/10 text-[10px]">
            {(["categoria", "rr"] as const).map((e) => (
              <button key={e} onClick={() => setEscala(e)} className={`px-1.5 py-0.5 ${escala === e ? "bg-white/15 text-slate-100" : "text-slate-400 hover:bg-white/5"}`}>
                {e === "categoria" ? "Categoría" : "RR"}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Los degradados repiten los cortes de colorRelleno (MapaAgeb.tsx): cambiar ambos a la vez. */}
      {modo === "comercio" && (
        <>
          <div className="h-2.5 rounded-sm" style={{ background: "linear-gradient(90deg,#1a2540,#2b2f5c 50%,#4c3a8f 75%,#8b5cf6 90%,#c4b5fd)" }} />
          <div className="mt-1 flex justify-between text-slate-400">
            <span>Pocos</span>
            <span>Muchos</span>
          </div>
          <p className="mt-1 text-slate-500">Percentil entre {NIVEL_PLURAL[nivel]}</p>
        </>
      )}
      {modo === "conteo" && (
        <>
          <div className="h-2.5 rounded-sm" style={{ background: "linear-gradient(90deg,#1a2540,#2c3352 50%,#6b3420 75%,#ea580c 90%,#ef4444)" }} />
          <div className="mt-1 flex justify-between text-slate-400">
            <span>Pocos</span>
            <span>Muchos</span>
          </div>
          <p className="mt-1 text-slate-500">Percentil entre {NIVEL_PLURAL[nivel]}</p>
        </>
      )}
      {modo === "riesgo" && escala === "categoria" && (
        <ul className="space-y-1 text-slate-300">
          <li className="flex items-center gap-2"><Muestra color={MAPA_CATEGORIA.por_encima} /> Más robos de lo esperado</li>
          <li className="flex items-center gap-2"><Muestra color={MAPA_CATEGORIA.dentro} /> Lo esperado</li>
          <li className="flex items-center gap-2"><Muestra color={MAPA_CATEGORIA.por_debajo} /> Menos robos de lo esperado</li>
        </ul>
      )}
      {modo === "riesgo" && escala === "rr" && (
        <>
          <div className="h-2.5 rounded-sm" style={{ background: "linear-gradient(90deg,#1d4ed8,#3b82f6 33%,#475569 50%,#f97316 67%,#dc2626)" }} />
          <div className="mt-1 flex justify-between text-slate-400">
            <span>×0.35</span>
            <span>×1</span>
            <span>×2.8</span>
          </div>
          <p className="mt-1 text-slate-500">Robos observados ÷ esperados (contraído)</p>
        </>
      )}

      <ul className="mt-2 space-y-1 border-t border-white/10 pt-2 text-slate-400">
        {vista !== "consulta" && vista !== "anormal" && nivel === "ageb" && (
          <li className="flex items-center gap-2"><Muestra borde="#f8fafc" /> Top 10% del modo activo</li>
        )}
        {destacar && (
          <>
            <li className="flex items-center gap-2"><Muestra borde={COLOR_NUEVO} /> Prioritaria, invisible por conteo</li>
            <li className="flex items-center gap-2"><Muestra borde="#e2e8f0" punteado /> Top por conteo que sale al ajustar</li>
          </>
        )}
        <li
          className="flex items-center gap-2"
          title={
            nivel === "colonia"
              ? `El catálogo de colonias no cubre el ${pctPequeno(meta.cobertura.pct_area_urbana_sin_colonia)} del área urbana (sobre todo pueblos del sur), donde ocurre el ${pctPequeno(meta.cobertura.pct_robos_sin_colonia)} del robo: esos robos sí cuentan en AGEB y alcaldía.`
              : undefined
          }
        >
          <Muestra color={COLOR_SIN_DATOS} /> {nivel === "colonia" ? "Sin actividad o fuera del catálogo de colonias" : "Sin actividad económica registrada"}
        </li>
        <li
          className="flex items-center gap-2"
          title={`Solo se analizan las ${meta.cobertura.agebs_urbanas.toLocaleString("es-MX")} AGEB urbanas. Las rurales concentran el ${pctPequeno(meta.cobertura.pct_robos_rurales)} del robo a negocio y el ${pctPequeno(meta.cobertura.pct_establecimientos_rurales)} de los establecimientos.`}
        >
          <Muestra rayado /> Zona rural: fuera del análisis
        </li>
      </ul>
    </div>
  );
}
