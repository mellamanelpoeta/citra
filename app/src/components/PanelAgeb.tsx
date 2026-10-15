import { useData } from "../data/context";
import { etiquetaPeriodo, metricaDe } from "../lib/derivados";
import { AVISO_SIN_VIOLENCIA, COLOR_ACIERTO, COLOR_AVISO, COLOR_FALLO, COLOR_NUEVO, decimal, entero, nombrePropio, porcentaje } from "../lib/format";
import { useApp } from "../lib/store";
import type { ZonaValidacion } from "../types/data";
import FichaAgeb from "./FichaAgeb";
import MiniTendencia from "./MiniTendencia";
import { Dato, InsigniaCategoria, Seccion, Tarjeta } from "./ui";

// Cifras pequeñas (robos por año) con un decimal; las grandes, enteras.
const cifra = (x: number) => (x < 10 ? decimal(x) : entero(x));

export default function PanelAgeb({ id }: { id: string }) {
  const { metrics, agebPorId, meta, bboxPorAgeb, prioridades, prioridadPorId, alcaldiaPorNombre, bboxPorAlcaldia, bboxPorColonia, validacionPorId } =
    useData();
  const { periodo, vista, modo, seleccionarAgeb, seleccionarZona } = useApp();
  const props = agebPorId.get(id);
  const mm = metrics[id];
  const m = mm ? metricaDe(mm, periodo) : undefined;
  const ant = periodo === "anticipado" ? mm?.anticipado : undefined;
  const periodoTxt = etiquetaPeriodo(periodo, meta.anticipado.anio);
  const prioridad = prioridadPorId.get(id);
  // En la prueba todo sale de validacion.json: el modelo que no vio el año evaluado (v0.6).
  const prueba = vista === "validar" ? validacionPorId.get(id) : undefined;
  const alc = props ? alcaldiaPorNombre.get(props.alcaldia) : undefined;
  const anioA = meta.anticipado.anio;

  return (
    <div>
      <div className="flex items-center justify-between">
        <button onClick={() => seleccionarAgeb(null)} className="text-xs text-slate-400 hover:text-slate-200">
          ← Volver
        </button>
        <button onClick={() => seleccionarAgeb(id, bboxPorAgeb.get(id))} className="text-xs text-accent hover:underline">
          Centrar en el mapa
        </button>
      </div>
      {/* Alcaldía y colonia llevan a su detalle en el nivel correspondiente del mapa */}
      <div className="mt-2 text-[11px] uppercase tracking-wider text-slate-500">
        {alc ? (
          <button onClick={() => seleccionarZona({ nivel: "alcaldia", id: alc.cve_mun }, bboxPorAlcaldia.get(alc.cve_mun))} className="uppercase hover:text-accent">
            {alc.alcaldia}
          </button>
        ) : (
          props?.alcaldia
        )}{" "}
        · AGEB {id}
      </div>
      <h2 className="text-xl font-semibold text-slate-50">
        {props ? (
          <button
            onClick={() => seleccionarZona({ nivel: "colonia", id: props.cve_col }, bboxPorColonia.get(props.cve_col))}
            className="text-left hover:text-accent"
            title="Ver la colonia"
          >
            {nombrePropio(props.colonia)}
          </button>
        ) : (
          id
        )}
      </h2>

      {prueba && mm ? (
        <Prueba z={prueba} porConteo={modo === "conteo"} />
      ) : !m || !mm ? (
        <Tarjeta className="mt-3 text-sm text-slate-300">
          Sin actividad económica registrada en el DENUE: no se calcula un nivel esperado de robos para esta zona.
        </Tarjeta>
      ) : (
        <>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <InsigniaCategoria categoria={m.categoria} />
            <span className="text-[11px] text-slate-500">{periodoTxt}</span>
          </div>
          {prioridad && (
            <div className="mt-1 text-[11px] text-slate-400">
              Zona prioritaria para {anioA}: #{prioridad.rank_riesgo} de {entero(prioridades.length)}
            </div>
          )}

          {prioridad?.senal_sin_violencia && (
            <div className="mt-3 rounded-lg border p-3 text-xs" style={{ borderColor: `${COLOR_AVISO}66`, background: `${COLOR_AVISO}14` }}>
              <div className="font-semibold" style={{ color: COLOR_AVISO }}>
                ⚠ {AVISO_SIN_VIOLENCIA.titulo}
              </div>
              <p className="mt-1 text-slate-300">
                Es prioritaria casi solo por robos sin violencia: contando solo los robos con violencia quedaría en el lugar{" "}
                <b className="text-slate-100">#{entero(prioridad.rank_riesgo_violencia)}</b>, fuera del 25% más alto. {AVISO_SIN_VIOLENCIA.detalle}
              </p>
            </div>
          )}

          {ant ? (
            <>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <Dato etiqueta="Robos por año (recientes)" valor={cifra(ant.observados)} nota={`#${entero(ant.rank_conteo)} por conteo`} />
                <Dato etiqueta="Le corresponden por su actividad" valor={cifra(ant.esperados)} nota={`Robos por año · modelo ${meta.modelo_esperado}`} />
                <Dato
                  etiqueta="Riesgo relativo"
                  valor={<span className={ant.rr > 1 ? "text-orange-400" : "text-sky-400"}>×{decimal(ant.rr)}</span>}
                  nota={`#${entero(ant.rank_riesgo)} por riesgo`}
                />
                <Dato etiqueta="Prob. de exceso" valor={porcentaje(ant.p_exceso)} nota="P(riesgo > esperado)" />
              </div>
              <p className="mt-2 text-[11px] text-slate-500">
                Con datos de {meta.anticipado.datos}; los años recientes pesan más.
              </p>
            </>
          ) : (
            <div className="mt-3 grid grid-cols-2 gap-2">
              <Dato etiqueta="Robos observados" valor={entero(m.observados)} nota={`#${entero(m.rank_conteo)} por conteo`} />
              <Dato etiqueta="Esperados por su actividad" valor={cifra(m.esperados)} nota={`Modelo ${meta.modelo_esperado}`} />
              <Dato
                etiqueta="Riesgo relativo"
                valor={<span className={m.rr > 1 ? "text-orange-400" : "text-sky-400"}>×{decimal(m.rr)}</span>}
                nota={`#${entero(m.rank_riesgo)} por riesgo`}
              />
              <Dato etiqueta="Prob. de exceso" valor={porcentaje(m.p_exceso)} nota="P(riesgo > esperado)" />
            </div>
          )}

          <Seccion titulo="Tendencia anual">
            <MiniTendencia
              puntos={meta.anios.map((a) => ({ anio: String(a), ...mm.por_anio[String(a)] }))}
              resaltar={periodo === "anticipado" ? undefined : periodo}
            />
          </Seccion>

          {m.delito_dominante && (
            <Seccion titulo="Delito dominante">
              <div className="text-sm text-slate-200">{m.delito_dominante}</div>
            </Seccion>
          )}

          <FichaAgeb id={id} conComparacion />
        </>
      )}
    </div>
  );
}

// Resultado de cada zona en la prueba: la señal (lista armada sin el año evaluado) contra lo que pasó ese año.
const RESULTADO = {
  acierto: { titulo: "Acierto", texto: "Se señaló y tuvo más robos de los que le correspondían.", color: COLOR_ACIERTO },
  falsa: { titulo: "Falsa alarma", texto: "Se señaló, pero no superó lo que le correspondía.", color: COLOR_FALLO },
  omitida: { titulo: "No se detectó", texto: "Tuvo más robos de los que le correspondían sin estar en la lista.", color: COLOR_NUEVO },
  nada: { titulo: "Sin señal y sin exceso", texto: "No se señaló y no superó lo que le correspondía.", color: COLOR_FALLO },
};

function Prueba({ z, porConteo }: { z: ZonaValidacion; porConteo: boolean }) {
  const { validacion } = useData();
  const anio = validacion.anio;
  const enLista = porConteo ? z.en_lista_conteo : z.en_lista_riesgo;
  const rank = porConteo ? z.rank_conteo : z.rank_riesgo;
  const r = RESULTADO[enLista ? (z.mas_de_lo_esperado ? "acierto" : "falsa") : z.mas_de_lo_esperado ? "omitida" : "nada"];
  return (
    <>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <InsigniaCategoria categoria={z.categoria} />
        <span className="text-[11px] text-slate-500">con datos de {validacion.datos}</span>
      </div>
      <div className="mt-1 text-[11px] text-slate-400">
        {enLista
          ? `Señalada para ${anio}: #${rank} de ${entero(validacion.resumen.lista_riesgo.n)} (lista por ${porConteo ? "conteo" : "riesgo"})`
          : `No señalada para ${anio} (#${entero(rank)} por ${porConteo ? "conteo" : "riesgo"})`}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <Dato
          etiqueta="Riesgo relativo"
          valor={<span className={z.rr > 1 ? "text-orange-400" : "text-sky-400"}>×{decimal(z.rr)}</span>}
          nota={`Con datos de ${validacion.datos}`}
        />
        <Dato etiqueta="Prob. de exceso" valor={porcentaje(z.p_exceso)} nota="P(riesgo > esperado)" />
      </div>

      <Seccion titulo={`¿Qué pasó en ${anio}?`}>
        <div className="rounded-lg border p-3 text-xs" style={{ borderColor: `${r.color}66`, background: `${r.color}14` }}>
          <div className="font-semibold" style={{ color: r.color }}>
            {r.titulo}
          </div>
          <p className="mt-1 text-slate-300">
            {r.texto} Hubo <b className="text-slate-100">{entero(z.observados)}</b> robos; por su actividad le correspondían{" "}
            <b className="text-slate-100">{cifra(z.esperados)}</b>.
          </p>
        </div>
      </Seccion>

      <Seccion titulo="Tendencia anual">
        <MiniTendencia
          puntos={z.serie.map((p) => ({ ...p, anio: String(p.anio) }))}
          prueba={{ anio: String(anio), observados: z.observados, esperados: z.esperados, mas_de_lo_esperado: z.mas_de_lo_esperado }}
        />
      </Seccion>

      <FichaAgeb id={z.cvegeo} conComparacion={false} />
    </>
  );
}
