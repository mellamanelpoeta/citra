import { useData } from "../data/context";
import { etiquetaPeriodo, metricaDe } from "../lib/derivados";
import { AVISO_SIN_VIOLENCIA, COLOR_AVISO, NIVEL_PLURAL, decimal, entero, nombrePropio, porcentaje } from "../lib/format";
import { useApp, type ZonaSel } from "../lib/store";
import MiniTendencia from "./MiniTendencia";
import { Dato, InsigniaCategoria, Seccion, Tarjeta } from "./ui";

// Detalle técnico (SEDECO) de una alcaldía o colonia: O y Ê suman los de sus AGEB urbanas (contrato v0.4).
export default function PanelZona({ zona }: { zona: ZonaSel }) {
  const datos = useData();
  const { meta, prioridades, agebPorId, bboxPorAgeb } = datos;
  const { periodo, seleccionarZona, seleccionarAgeb } = useApp();
  const esAlcaldia = zona.nivel === "alcaldia";
  const mm = (esAlcaldia ? datos.alcaldiasMetrics : datos.coloniasMetrics)[zona.id];
  const m = mm ? metricaDe(mm, periodo) : undefined;
  const ant = periodo === "anticipado" ? mm?.anticipado : undefined;
  const anioA = meta.anticipado.anio;
  const n = Object.keys(esAlcaldia ? datos.alcaldiasMetrics : datos.coloniasMetrics).length;
  const bbox = (esAlcaldia ? datos.bboxPorAlcaldia : datos.bboxPorColonia).get(zona.id);
  const alc = esAlcaldia ? datos.alcaldiaPorId.get(zona.id) : undefined;
  const col = esAlcaldia ? undefined : datos.coloniaGeoPorId.get(zona.id);
  const nombre = alc?.alcaldia ?? (col ? nombrePropio(col.nombre) : zona.id);
  const aprox = esAlcaldia ? "" : "≈ ";
  const cifra = (x: number) => `${aprox}${x < 10 ? decimal(x) : entero(x)}`;

  // AGEB prioritarias dentro de la zona. En colonia: las AGEB que la cubren (mismo criterio que colonias.json).
  const agebsCol = new Set(esAlcaldia ? [] : (datos.coloniaPorId.get(zona.id)?.agebs ?? []));
  const prior = prioridades.filter((p) => (esAlcaldia ? p.alcaldia === alc?.alcaldia : agebsCol.has(p.cvegeo)));

  return (
    <div>
      <div className="flex items-center justify-between">
        <button onClick={() => seleccionarZona(null)} className="text-xs text-slate-400 hover:text-slate-200">
          ← Volver
        </button>
        <button onClick={() => seleccionarZona(zona, bbox)} className="text-xs text-accent hover:underline">
          Centrar en el mapa
        </button>
      </div>
      <div className="mt-2 text-[11px] uppercase tracking-wider text-slate-500">{esAlcaldia ? "Alcaldía" : `${col?.alcaldia} · Colonia`}</div>
      <h2 className="text-xl font-semibold text-slate-50">{nombre}</h2>

      {!m || !mm ? (
        <Tarjeta className="mt-3 text-sm text-slate-300">
          Ninguna AGEB urbana con actividad económica registrada en el DENUE cubre esta colonia: no se calcula un nivel esperado de robos.
        </Tarjeta>
      ) : (
        <>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <InsigniaCategoria categoria={m.categoria} />
            <span className="text-[11px] text-slate-500">{etiquetaPeriodo(periodo, meta.anticipado.anio)}</span>
          </div>

          <div className="mt-3 grid grid-cols-2 gap-2">
            {ant ? (
              <>
                <Dato etiqueta="Robos por año (recientes)" valor={cifra(ant.observados)} nota="Los años recientes pesan más" />
                <Dato etiqueta="Le corresponden por su actividad" valor={cifra(ant.esperados)} nota="Robos por año · suma de sus AGEB" />
              </>
            ) : (
              <>
                <Dato
                  etiqueta="Robos observados"
                  valor={`${aprox}${entero(m.observados)}`}
                  nota={`#${entero(m.rank_conteo)} por conteo de ${entero(n)} ${NIVEL_PLURAL[zona.nivel]}`}
                />
                <Dato etiqueta="Esperados por su actividad" valor={cifra(m.esperados)} nota="Suma de sus AGEB" />
              </>
            )}
            <Dato
              etiqueta="Riesgo relativo"
              valor={<span className={m.rr > 1 ? "text-orange-400" : "text-sky-400"}>×{decimal(m.rr)}</span>}
              nota={`#${entero(m.rank_riesgo)} por riesgo`}
            />
            <Dato etiqueta="Prob. de exceso" valor={porcentaje(m.p_exceso)} nota="P(riesgo > esperado)" />
          </div>
          {ant && (
            <p className="mt-2 text-[11px] text-slate-500">
              Ritmo reciente: {cifra(ant.observados)} robos por año (#{entero(ant.rank_conteo)} por conteo de {entero(n)}{" "}
              {NIVEL_PLURAL[zona.nivel]}), con más peso a los años recientes.
            </p>
          )}
          {!esAlcaldia && (
            <p className="mt-2 text-[11px] text-slate-500">
              Los robos y lo esperado de cada AGEB se reparten entre las colonias que la cubren según el área que comparten: las cifras son
              aproximadas.
            </p>
          )}

          <Seccion titulo="Tendencia anual">
            <MiniTendencia
              puntos={meta.anios.map((a) => ({ anio: String(a), ...mm.por_anio[String(a)] }))}
              resaltar={periodo === "anticipado" ? undefined : periodo}
            />
          </Seccion>

          <Seccion titulo="Cobertura">
            <div className="grid grid-cols-2 gap-2">
              <Dato etiqueta="Establecimientos" valor={`${aprox}${entero(m.establecimientos)}`} nota="DENUE" />
              {alc ? (
                <Dato
                  etiqueta="AGEB urbanas"
                  valor={entero(alc.agebs_urbanas)}
                  nota={alc.agebs_rurales ? `${porcentaje(alc.pct_area_urbana)} del área; el resto es rural` : "100% del área"}
                />
              ) : (
                <Dato etiqueta="AGEB que la cubren" valor={entero(agebsCol.size)} />
              )}
            </div>
          </Seccion>
        </>
      )}

      <Seccion titulo={`Zonas prioritarias para ${anioA} aquí (${prior.length})`}>
        {prior.length === 0 ? (
          <p className="text-xs text-slate-500">Ninguna de las {entero(prioridades.length)} AGEB prioritarias está en esta {esAlcaldia ? "alcaldía" : "colonia"}.</p>
        ) : (
          <Tarjeta className="!p-1">
            {prior.slice(0, 10).map((p) => {
              return (
                <button
                  key={p.cvegeo}
                  onClick={() => seleccionarAgeb(p.cvegeo, bboxPorAgeb.get(p.cvegeo))}
                  className="flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-white/5"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm text-slate-200">
                      #{p.rank_riesgo} {nombrePropio(agebPorId.get(p.cvegeo)?.colonia ?? p.colonia)}
                      {p.senal_sin_violencia && (
                        <span className="ml-1 text-[11px]" style={{ color: COLOR_AVISO }} title={AVISO_SIN_VIOLENCIA.corto}>
                          ⚠
                        </span>
                      )}
                    </span>
                    <span className="block text-[11px] text-slate-500">AGEB {p.cvegeo}</span>
                  </span>
                  <span className="shrink-0 text-right text-[11px] tabular-nums text-slate-400">
                    ×{decimal(p.rr)}
                  </span>
                </button>
              );
            })}
            {prior.length > 10 && <p className="px-2 py-1 text-[11px] text-slate-500">y {prior.length - 10} más en la vista «¿Qué priorizar?»</p>}
          </Tarjeta>
        )}
      </Seccion>
    </div>
  );
}
