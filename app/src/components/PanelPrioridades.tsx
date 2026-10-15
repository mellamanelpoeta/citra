import { useMemo, useState } from "react";
import { useData } from "../data/context";
import { descargarCsv } from "../lib/csv";
import { PANEL } from "../lib/config";
import { AVISO_SIN_VIOLENCIA, COLOR_AVISO, COLOR_NUEVO, decimal, entero, nombrePropio, porcentaje } from "../lib/format";
import { useResumen } from "../lib/derivados";
import { useAnchoPanel, useApp } from "../lib/store";

// Paso 4 (SEDECO): ranking de prioridades.json, la lista para el año siguiente al último observado (v0.5).
export default function PanelPrioridades() {
  const { prioridades, meta, metrics, bboxPorAgeb } = useData();
  const { agebSel, seleccionarAgeb } = useApp();
  const anio = meta.anticipado.anio;
  const [soloOcultas, setSoloOcultas] = useState(false);
  const [sinAviso, setSinAviso] = useState(false);
  const [alcaldia, setAlcaldia] = useState("");
  // Con el panel ensanchado cabe la probabilidad de exceso, que el CSV ya incluye.
  const amplio = useAnchoPanel() >= PANEL.amplio;
  // Las mismas que el contorno cian del mapa: fuera del top por conteo y "por encima de lo esperado" (p ≥ 0.9). Las demás
  // no visibles por conteo están en la lista, pero con un exceso menos claro.
  const { soloRiesgo } = useResumen();

  const alcaldias = useMemo(() => [...new Set(prioridades.map((p) => p.alcaldia))].sort(), [prioridades]);
  const filas = useMemo(
    () =>
      prioridades.filter(
        (p) => (!soloOcultas || soloRiesgo.has(p.cvegeo)) && (!sinAviso || !p.senal_sin_violencia) && (!alcaldia || p.alcaldia === alcaldia),
      ),
    [prioridades, soloOcultas, sinAviso, alcaldia, soloRiesgo],
  );
  const ocultas = prioridades.filter((p) => soloRiesgo.has(p.cvegeo)).length;
  const ocultasDudosas = prioridades.filter((p) => !p.visible_por_conteo).length - ocultas;
  const conAviso = prioridades.filter((p) => p.senal_sin_violencia).length;
  const nAnios = meta.anios.length;

  const exportar = () =>
    descargarCsv(
      `prioridades_riesgo_ajustado_${anio}.csv`,
      filas.map((p) => ({
        rank_riesgo: p.rank_riesgo,
        cvegeo: p.cvegeo,
        colonia: p.colonia,
        cve_col: p.cve_col,
        alcaldia: p.alcaldia,
        riesgo_relativo: p.rr,
        prob_exceso: p.p_exceso,
        robos_por_anio_recientes: metrics[p.cvegeo].anticipado.observados,
        esperados_por_anio: p.esperados,
        persistencia_anios: p.persistencia,
        rank_conteo: p.rank_conteo,
        visible_por_conteo: p.visible_por_conteo ? "sí" : "no",
        rank_riesgo_solo_violencia: p.rank_riesgo_violencia,
        revisar_senal_sin_violencia: p.senal_sin_violencia ? "sí" : "no",
      })),
    );

  return (
    <div className="flex h-full flex-col">
      <div className="text-[11px] uppercase tracking-wider text-accent">¿Qué priorizar? · {anio}</div>
      <h2 className="mt-1 text-xl font-semibold leading-snug text-slate-50">
        {prioridades.length} zonas con más robo del que les corresponde por su actividad
      </h2>
      <p className="mt-1 text-sm text-slate-400">
        <b style={{ color: COLOR_NUEVO }}>{ocultas}</b> zonas con más robos de lo esperado no estarían en la lista si se priorizara por conteo de
        robos.{ocultasDudosas > 0 }
      </p>
      <p className="mt-1 text-xs text-slate-400" title={AVISO_SIN_VIOLENCIA.detalle}>
        <span style={{ color: COLOR_AVISO }}>⚠ {conAviso}</span> son prioritarias casi solo por robo sin violencia: revisarlas caso por caso
        antes de intervenir.
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
        <label className="flex cursor-pointer items-center gap-1.5 text-slate-300">
          <input type="checkbox" checked={soloOcultas} onChange={(e) => setSoloOcultas(e.target.checked)} className="accent-cyan-400" />
          Solo no visibles por conteo (en cian)
        </label>
        <label className="flex cursor-pointer items-center gap-1.5 text-slate-300">
          <input type="checkbox" checked={sinAviso} onChange={(e) => setSinAviso(e.target.checked)} className="accent-amber-400" />
          Sin ⚠
        </label>
        <select
          value={alcaldia}
          onChange={(e) => setAlcaldia(e.target.value)}
          className="velo rounded-md px-1.5 py-1 text-slate-200 [&>option]:bg-ink-900"
          aria-label="Filtrar por alcaldía"
        >
          <option value="">Todas las alcaldías</option>
          {alcaldias.map((a) => (
            <option key={a}>{a}</option>
          ))}
        </select>
        <button onClick={exportar} className="velo ml-auto rounded-md px-2 py-1 text-slate-200 transition-colors hover:bg-white/10">
          Exportar CSV ({filas.length})
        </button>
      </div>

      <div className="scroll-fino mt-2 min-h-0 flex-1 overflow-y-auto velo rounded-xl">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-ink-900/85 text-[11px] text-slate-400 backdrop-blur-md">
            <tr>
              <th className="px-2 py-1.5 font-medium">#</th>
              <th className="px-2 py-1.5 font-medium">Zona</th>
              <th className="px-2 py-1.5 text-right font-medium" title="Riesgo relativo: robos observados ÷ esperados (más peso a lo reciente)">RR</th>
              {amplio && (
                <th className="px-2 py-1.5 text-right font-medium" title="Robos por año (recientes) y los que le corresponden por su actividad">
                  Por año: hubo / esp.
                </th>
              )}
              {amplio && (
                <th className="px-2 py-1.5 text-right font-medium" title="Probabilidad de que el riesgo esté por encima de lo esperado">
                  P(exceso)
                </th>
              )}
              <th className="px-2 py-1.5 font-medium" title={`Años con riesgo por encima de lo esperado (de ${nAnios})`}>Persistencia</th>
              <th className="px-2 py-1.5 font-medium">Por conteo</th>
            </tr>
          </thead>
          <tbody>
            {filas.map((p) => (
              <tr
                key={p.cvegeo}
                onClick={() => seleccionarAgeb(p.cvegeo, bboxPorAgeb.get(p.cvegeo))}
                className={`cursor-pointer border-t border-white/5 transition-colors hover:bg-white/5 ${agebSel === p.cvegeo ? "bg-accent/10" : ""}`}
              >
                <td className="px-2 py-1.5 tabular-nums text-slate-500">{p.rank_riesgo}</td>
                <td className={`px-2 py-1.5 ${amplio ? "max-w-64" : "max-w-40"}`}>
                  <div className="flex items-center gap-1">
                    <span className="truncate text-slate-200">{nombrePropio(p.colonia)}</span>
                    {p.senal_sin_violencia && (
                      <span
                        className="shrink-0 text-[11px]"
                        style={{ color: COLOR_AVISO }}
                        title={`${AVISO_SIN_VIOLENCIA.corto}. Con solo robo con violencia quedaría en el lugar #${entero(p.rank_riesgo_violencia)}.`}
                        aria-label={AVISO_SIN_VIOLENCIA.titulo}
                      >
                        ⚠
                      </span>
                    )}
                  </div>
                  <div className="truncate text-[11px] text-slate-500">{p.alcaldia}</div>
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums text-orange-300">×{decimal(p.rr)}</td>
                {amplio && (
                  <td className="px-2 py-1.5 text-right tabular-nums text-slate-300">
                    {decimal(metrics[p.cvegeo].anticipado.observados)} <span className="text-slate-500">/ {decimal(p.esperados)}</span>
                  </td>
                )}
                {amplio && <td className="px-2 py-1.5 text-right tabular-nums text-slate-300">{porcentaje(p.p_exceso)}</td>}
                <td className="px-2 py-1.5">
                  <div className="flex gap-0.5" aria-label={`${p.persistencia} de ${nAnios} años`}>
                    {Array.from({ length: nAnios }, (_, i) => (
                      <span key={i} className={`h-2 w-1.5 rounded-sm ${i < p.persistencia ? "bg-red-400" : "bg-white/10"}`} />
                    ))}
                  </div>
                </td>
                <td className="px-2 py-1.5">
                  {p.visible_por_conteo ? (
                    <span className="text-[11px] text-slate-500">Visible (#{entero(p.rank_conteo)})</span>
                  ) : !soloRiesgo.has(p.cvegeo) ? (
                    <span className="text-[11px] text-slate-400" title="Tampoco está en el top por conteo, pero su exceso es menos claro (no llega a «por encima de lo esperado»)">
                      No visible
                    </span>
                  ) : (
                    <span className="rounded px-1.5 py-0.5 text-[11px] font-medium" style={{ color: COLOR_NUEVO, background: `${COLOR_NUEVO}1a` }}>
                      No visible
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
