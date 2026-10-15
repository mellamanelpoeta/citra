import { useMemo } from "react";
import colchon from "../data/colchon.json";
import { useData } from "../data/context";
import { ALCALDIAS_CENTRO, metricaDe, reparto, useResumen } from "../lib/derivados";
import { AVISO_SIN_VIOLENCIA, COLOR_AVISO, COLOR_NUEVO, decimal, entero, nombrePropio, porcentaje } from "../lib/format";
import { useApp, type Modo } from "../lib/store";
import GraficaColchon from "./GraficaColchon";
import { GraficaGiro } from "./GraficasCifraNegra";
import { BarraPasos, Paso, useRecorrido } from "./Recorrido";
import { Seccion, Tarjeta } from "./ui";

function Fila({ id, nota }: { id: string; nota: string }) {
  const { agebPorId, bboxPorAgeb, prioridadPorId } = useData();
  const seleccionarAgeb = useApp((s) => s.seleccionarAgeb);
  const p = agebPorId.get(id);
  const aviso = prioridadPorId.get(id)?.senal_sin_violencia;
  return (
    <button
      onClick={() => seleccionarAgeb(id, bboxPorAgeb.get(id))}
      className="flex w-full items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-white/5"
    >
      <span className="min-w-0">
        <span className="block truncate text-sm text-slate-200">
          {p ? nombrePropio(p.colonia) : id}
          {aviso && (
            <span className="ml-1 text-[11px]" style={{ color: COLOR_AVISO }} title={AVISO_SIN_VIOLENCIA.corto}>
              ⚠
            </span>
          )}
        </span>
        <span className="block text-[11px] text-slate-500">{p?.alcaldia}</span>
      </span>
      <span className="shrink-0 text-right text-[11px] tabular-nums text-slate-400">{nota}</span>
    </button>
  );
}

const PASOS: { modo: Modo; t: string }[] = [
  { modo: "comercio", t: "Comercio" },
  { modo: "conteo", t: "Robos denunciados" },
  { modo: "riesgo", t: "Riesgo ajustado" },
];

// Paso 2: recorrido del mapa del comercio al del riesgo. La sección a la vista en el panel decide qué pinta el mapa.
export default function PanelAnormal() {
  const { metrics, agebPorId, validacion } = useData();
  const { setModo, nivel, irA } = useApp();
  const { nTop, topConteo, topComercio, soloRiesgo, soloConteo } = useResumen();

  const c = useMemo(() => {
    const ids = Object.keys(metrics);
    const alc = (id: string) => agebPorId.get(id)?.alcaldia;
    const negocios = ids.reduce((s, id) => s + metrics[id].anticipado.establecimientos, 0);
    const negociosCentro = ids
      .filter((id) => ALCALDIAS_CENTRO.includes(alc(id) ?? ""))
      .reduce((s, id) => s + metrics[id].anticipado.establecimientos, 0);
    // Las 8 zonas más anómalas que el conteo no ve, y las 5 del top por conteo que salen al ajustar.
    const orden = (s: Set<string>, k: "rank_riesgo" | "rank_conteo") =>
      [...s].map((id) => ({ id, m: metricaDe(metrics[id], "anticipado") })).sort((a, b) => a.m[k] - b.m[k]);
    return {
      agebsCentro: reparto(ids, alc).centro,
      negociosCentro: negociosCentro / negocios,
      coinciden: [...topConteo].filter((id) => topComercio.has(id)).length,
      conteo: reparto([...topConteo], alc),
      nuevas: reparto([...soloRiesgo], alc),
      ocultas: orden(soloRiesgo, "rank_riesgo").slice(0, 8),
      salen: orden(soloConteo, "rank_conteo").slice(0, 5),
    };
  }, [metrics, agebPorId, topConteo, topComercio, soloRiesgo, soloConteo]);

  // Al volver del detalle de una zona el panel se monta de nuevo: se abre en el paso que estaba en el mapa.
  const { activo, ir, refPaso, barra } = useRecorrido(
    Math.max(0, PASOS.findIndex((p) => p.modo === useApp.getState().modo)),
    (i) => setModo(PASOS[i].modo),
  );
  const agregado = nivel !== "ageb";

  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-accent">¿Qué es anormal?</div>
      <h2 className="mt-1 text-xl font-semibold leading-snug text-slate-50">Del mapa del comercio al mapa del riesgo</h2>
      <p className="mt-1 text-sm text-slate-400">Baja por el panel: el mapa cambia en cada paso.</p>

      <BarraPasos barra={barra} etiquetas={PASOS.map((p) => p.t)} activo={activo} ir={ir} />

      <Paso n={1} titulo="Dónde está el comercio" refPaso={refPaso(0)}>
        <p>
          Cada zona se pinta según su número de establecimientos (DENUE, noviembre de 2023). El comercio se concentra en el centro:{" "}
          {ALCALDIAS_CENTRO.join(", ")} tienen el {porcentaje(c.agebsCentro)} de las zonas y el{" "}
          <b className="text-slate-100">{porcentaje(c.negociosCentro)} de los negocios</b>.
        </p>
        <p className="text-slate-400">Es la exposición: donde hay más negocios hay más oportunidades de robo.</p>
      </Paso>

      <Paso n={2} titulo="Dónde se denuncian más robos" refPaso={refPaso(1)}>
        <p>
          El mapa se parece al anterior. De las {nTop} zonas con más robos, <b className="text-slate-100">{c.coinciden}</b> están también entre
          las {nTop} con más negocios (al azar coincidirían ~{Math.round((nTop * nTop) / Object.keys(metrics).length)}).
        </p>
        <p>
          El <b className="text-slate-100">{porcentaje(c.conteo.centro)}</b> de esas zonas está en las tres alcaldías centrales. Con este mapa,
          SEDECO miraría sobre todo al centro.
        </p>
      </Paso>

      <Paso n={3} titulo="Dónde hay más robo del esperado" refPaso={refPaso(2)}>
        <p>
          Ahora cada zona se compara con los robos que le corresponderían por su número, giro y tamaño de negocios.{" "}
          <b style={{ color: COLOR_NUEVO }}>{soloRiesgo.size}</b> zonas prioritarias con más robos de lo esperado no estaban entre las {nTop} con
          más robos.
        </p>
        <p>
          Esas zonas nuevas están fuera del centro: solo el {porcentaje(c.nuevas.centro)} cae en las alcaldías centrales; la mayoría está en{" "}
          {c.nuevas.principales.join(", ")}. El conteo las dejaba invisibles.
        </p>

        <p>Lo esperado de cada zona toma en cuenta dos patrones de los datos: cuántos negocios tiene y de qué tipo son.</p>

        {/* Colchón de densidad (hipótesis A) */}
        <Seccion titulo="El robo crece más despacio que el comercio">
          <Tarjeta className="!p-3">
            <p className="text-sm text-slate-300">
              Lo esperado no sale de una regla de tres. Comparando zonas con los mismos giros, en la mitad de la ciudad con menos comercio
              duplicar los negocios multiplica el robo por ×{decimal(2 ** colchon.m1.beta_bajo)}; en la mitad con más comercio, por ×
              {decimal(2 ** colchon.m1.beta_alto)}. En las zonas con más comercio, cada negocio adicional se asocia con menos robo adicional.
            </p>
            <div className="mt-3">
              <GraficaColchon compacta />
            </div>
            <p className="mt-2 text-xs text-slate-400">
              Es una relación entre zonas, no un experimento: no prueba que hacer crecer el comercio de una zona reduzca su robo.
            </p>
          </Tarjeta>
        </Seccion>

        {/* pipeline/outputs/modelos/giros_violencia.csv (04h): IRR por +10 pp de cada grupo, con y sin violencia */}
        <Seccion titulo="Por qué importa el giro">
          <Tarjeta className="!p-3">
            <div className="space-y-2 text-sm text-slate-300">
              <p>
                No solo cuenta cuántos negocios hay, sino de qué tipo. Con el mismo número de negocios, las zonas con más negocios de alto
                efectivo formal (minisúpers, farmacias, gasolineras, restaurantes) tienen más robo, con y sin violencia. Por eso el robo
                esperado de cada zona considera su mezcla de giros: una zona no sale como anormal solo por tener muchos negocios de este
                tipo.
              </p>
              <p>
                Frente al resto del comercio al por menor, los de alto efectivo tradicional (abarrotes, tortillerías) tienen el mismo robo
                con violencia, pero menos sin violencia. Si fueran zonas más seguras, bajarían los dos: puede ser que denuncien menos ese
                robo, quizá por falta de los mecanismos y protocolos que los giros formales sí tienen, o que el mostrador lo dificulte.
              </p>
            </div>
            <div className="mt-3">
              <GraficaGiro />
            </div>
            <p className="mt-2 text-xs text-slate-400">En gris, el resto de los giros como comparación.</p>
          </Tarjeta>
        </Seccion>
        {agregado && (
          <p className="text-xs text-slate-400">
            Estás viendo el mapa por {nivel === "alcaldia" ? "alcaldía" : "colonia"}: al sumar AGEB se diluyen los focos. Las listas de abajo
            son por AGEB.
          </p>
        )}

        <Seccion titulo="Prioritarias que el conteo no ve">
          <Tarjeta className="!p-1">
            {c.ocultas.map(({ id, m }) => (
              <Fila key={id} id={id} nota={`×${decimal(m.rr)} · #${entero(m.rank_conteo)} por conteo`} />
            ))}
          </Tarjeta>
        </Seccion>

        <Seccion titulo="Muchos robos, pero los esperados por su actividad">
          <Tarjeta className="!p-1">
            {c.salen.map(({ id, m }) => (
              <Fila key={id} id={id} nota={`${decimal(m.observados)} robos/año · ×${decimal(m.rr)}`} />
            ))}
          </Tarjeta>
        </Seccion>
      </Paso>

      <button
        onClick={() => irA("validar")}
        className="mt-5 w-full rounded-lg border border-accent/50 bg-accent/10 px-3 py-2.5 text-sm font-medium text-accent hover:bg-accent/20"
      >
        ¿Habría funcionado? Probarlo con {validacion.anio} →
      </button>
    </div>
  );
}
