import { useMemo, useState, type ReactNode } from "react";
import { useData } from "../data/context";
import { PANEL } from "../lib/config";
import { ALCALDIAS_CENTRO, reparto } from "../lib/derivados";
import { COLOR_ACIERTO, COLOR_FALLO, COLOR_NUEVO, decimal, entero, nombrePropio, porcentaje } from "../lib/format";
import { PASOS_PRUEBA, useAnchoPanel, useApp } from "../lib/store";
import type { Vulnerabilidad } from "../types/data";
import { BarraPasos, Paso, useRecorrido } from "./Recorrido";
import { Tarjeta } from "./ui";

const COLOR_CONTEO = "#f97316";

function Barra({ etiqueta, valor, color, nota }: { etiqueta: string; valor: number; color: string; nota?: string }) {
  return (
    <div>
      <div className="mb-0.5 flex justify-between text-xs">
        <span className="text-slate-300">{etiqueta}</span>
        <span className="font-semibold tabular-nums text-slate-100">{porcentaje(valor)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-white/10">
        <div className="h-full rounded-full" style={{ width: `${valor * 100}%`, background: color }} />
      </div>
      {nota && <div className="mt-0.5 text-[11px] text-slate-500">{nota}</div>}
    </div>
  );
}

// Cifra grande con su lectura debajo: los pasos van en cifras en vez de párrafos. `chica` para filas de tres.
function Cifra({ valor, texto, color, chica = false }: { valor: ReactNode; texto: ReactNode; color?: string; chica?: boolean }) {
  return (
    <Tarjeta className="!p-2.5">
      <div className={`font-semibold tabular-nums leading-tight text-slate-100 ${chica ? "text-lg" : "text-2xl"}`} style={{ color }}>
        {valor}
      </div>
      <div className="mt-1 text-[11px] leading-snug text-slate-400">{texto}</div>
    </Tarjeta>
  );
}

// Indicadores del Censo 2020 que se comparan entre listas (paso 4).
const INDICADORES: { k: keyof Vulnerabilidad; t: string; nota: string; f: (x: number) => string }[] = [
  { k: "escolaridad", t: "Escolaridad promedio", nota: "años, de 15 años o más", f: (x) => decimal(x) },
  { k: "pct_sin_basica", t: "Sin secundaria terminada", nota: "de 15 años o más", f: porcentaje },
  { k: "pct_sin_salud", t: "Sin servicio de salud", nota: "IMSS, ISSSTE u otro", f: porcentaje },
  { k: "pct_sin_internet", t: "Viviendas sin internet", nota: "", f: porcentaje },
];

// Paso 3 (v0.5): "si lo hubiéramos hecho en 2024". La lista se arma igual que la del año siguiente, pero solo con los
// años anteriores (el modelo no vio el año evaluado), y se compara con lo que pasó. Es un recorrido: la lista por conteo,
// la lista por riesgo, dónde caen las zonas nuevas, quién vive ahí y, al final, la tabla zona por zona.
export default function PanelValidacion() {
  const { validacion, meta, bboxPorAgeb } = useData();
  const { agebSel, seleccionarAgeb, modo, setModo, setPasoPrueba } = useApp();
  const [soloFallos, setSoloFallos] = useState(false);
  const [soloOcultas, setSoloOcultas] = useState(false);
  const amplio = useAnchoPanel() >= PANEL.amplio;
  const { anio, datos, resumen: r, vulnerabilidad: v } = validacion;
  const porConteo = modo === "conteo";
  // La lista es el 10% de las AGEB de la ciudad: decirlo así da escala a "242 zonas".
  const parte = r.lista_riesgo.n / validacion.zonas.length;

  // Al volver del detalle de una zona el panel se monta de nuevo: se abre en el paso en que estaba.
  const { activo, ir, refPaso, barra } = useRecorrido(useApp.getState().pasoPrueba, (i) => {
    setPasoPrueba(i);
    const m = PASOS_PRUEBA[i].modo;
    if (m) setModo(m);
  });

  // Dónde caen las listas: la de conteo mira al centro; la de riesgo se reparte y sus zonas nuevas están en la periferia.
  const centro = useMemo(() => {
    const zs = validacion.zonas;
    const alc = new Map(zs.map((z) => [z.cvegeo, z.alcaldia]));
    const ids = (f: (z: (typeof zs)[number]) => boolean) => zs.filter(f).map((z) => z.cvegeo);
    return {
      conteo: reparto(ids((z) => z.en_lista_conteo), (id) => alc.get(id)),
      riesgo: reparto(ids((z) => z.en_lista_riesgo), (id) => alc.get(id)),
      nuevas: reparto(ids((z) => z.en_lista_riesgo && !z.en_lista_conteo), (id) => alc.get(id)),
    };
  }, [validacion]);

  const filas = useMemo(
    () =>
      validacion.zonas
        .filter((z) => (porConteo ? z.en_lista_conteo : z.en_lista_riesgo))
        .filter((z) => (!soloFallos || !z.mas_de_lo_esperado) && (!soloOcultas || !z.en_lista_conteo))
        .sort((a, b) => (porConteo ? a.rank_conteo - b.rank_conteo : a.rank_riesgo - b.rank_riesgo)),
    [validacion, porConteo, soloFallos, soloOcultas],
  );

  const columnas: { k: "lista_conteo" | "no_visibles" | "ciudad"; t: string; color: string }[] = [
    { k: "lista_conteo", t: "Lista por conteo", color: COLOR_CONTEO },
    { k: "no_visibles", t: "Mayor riesgo, fuera del conteo", color: COLOR_NUEVO },
    { k: "ciudad", t: "Ciudad", color: "#94a3b8" },
  ];

  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-accent">¿Habría funcionado? · Prueba {anio}</div>
      <h2 className="mt-1 text-xl font-semibold leading-snug text-slate-50">
        Con datos hasta {anio - 1} señalamos el {porcentaje(parte)} de las zonas de la ciudad: el{" "}
        <span style={{ color: COLOR_ACIERTO }}>{porcentaje(r.lista_riesgo.pct_mas_de_lo_esperado)}</span> tuvo más robos de los esperados en{" "}
        {anio}
      </h2>
      <p className="mt-2 text-sm text-slate-400">
        Son {entero(r.lista_riesgo.n)} de {entero(validacion.zonas.length)} AGEB. Armamos la lista igual que la de {meta.anticipado.anio},
        pero solo con {datos}: ni el modelo ni la lista vieron {anio}.
      </p>
      <p className="mt-2 text-sm text-slate-400">Baja por el panel: el mapa cambia en cada paso.</p>

      <BarraPasos barra={barra} etiquetas={PASOS_PRUEBA.map((p) => p.t)} activo={activo} ir={ir} />

      <Paso n={1} titulo="La regla simple: las zonas con más robos" refPaso={refPaso(0)} holgado>
        <p>
          Si SEDECO señalara las {entero(r.lista_conteo.n)} zonas con más robos denunciados de {datos}, ¿cuántas tendrían más robos de los
          esperados por su actividad en {anio}?
        </p>
        <div className="grid grid-cols-2 gap-2">
          <Cifra valor={porcentaje(r.lista_conteo.pct_mas_de_lo_esperado)} color={COLOR_ACIERTO} texto="de la lista por conteo" />
          <Cifra valor={porcentaje(r.ciudad.pct_mas_de_lo_esperado)} texto="de toda la ciudad (referencia)" />
        </div>
        <p className="text-xs text-slate-400">
          En el mapa solo se pintan las zonas de la lista: en rojo, las que en {anio} tuvieron más robos de los esperados.
        </p>
      </Paso>

      <Paso n={2} titulo="La lista por riesgo ajustado" refPaso={refPaso(1)} holgado>
        <p>
          Misma información, otro orden: por cuánto robo tienen por encima de lo que les correspondería por su número, giro y tamaño de
          negocios. Acierta más:
        </p>
        <Tarjeta className="space-y-3">
          <div className="text-[11px] text-slate-400">Zonas con más robos de los esperados en {anio}</div>
          <Barra etiqueta="Lista por riesgo ajustado" valor={r.lista_riesgo.pct_mas_de_lo_esperado} color={COLOR_ACIERTO} />
          <Barra etiqueta="Lista por conteo de robos" valor={r.lista_conteo.pct_mas_de_lo_esperado} color={COLOR_CONTEO} />
          <Barra etiqueta="Toda la ciudad (referencia)" valor={r.ciudad.pct_mas_de_lo_esperado} color={COLOR_FALLO} />
        </Tarjeta>
        <div>
          <div className="mb-2 text-[11px] text-slate-400">
            Las {entero(r.lista_riesgo.n)} zonas de la lista por riesgo, en {anio}
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Cifra valor={entero(r.lista_riesgo.robos)} texto="robos" chica />
            <Cifra valor={entero(r.lista_riesgo.esperados)} texto="les correspondían por su actividad" chica />
            <Cifra
              valor={porcentaje(r.lista_riesgo.pct_exceso_ciudad)}
              texto={`del exceso de robo de la ciudad (conteo: ${porcentaje(r.lista_conteo.pct_exceso_ciudad)})`}
              chica
            />
          </div>
        </div>
      </Paso>

      <Paso n={3} titulo="El conteo mira al centro; el riesgo encuentra la periferia" refPaso={refPaso(2)} holgado>
        <div>
          <div className="mb-2 text-[11px] text-slate-400">
            Zonas en {ALCALDIAS_CENTRO.join(", ")}, donde se concentra el comercio
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Cifra valor={porcentaje(centro.conteo.centro)} color={COLOR_CONTEO} texto="de la lista por conteo" />
            <Cifra valor={porcentaje(centro.riesgo.centro)} color={COLOR_ACIERTO} texto="de la lista por riesgo" />
          </div>
        </div>
        <Tarjeta className="space-y-2" style={{ borderColor: `${COLOR_NUEVO}55` }}>
          <div>
            <span className="text-2xl font-semibold tabular-nums" style={{ color: COLOR_NUEVO }}>
              {entero(r.no_visibles.n)}
            </span>{" "}
            <span className="text-slate-200">zonas de mayor riesgo no estaban en la lista por conteo</span>
            <span className="text-slate-400"> (en cian en el mapa)</span>
          </div>
          <ul className="list-disc space-y-1.5 pl-4 marker:text-slate-500">
            <li>
              Solo el {porcentaje(centro.nuevas.centro)} está en el centro: se reparten por la periferia, sobre todo en{" "}
              {centro.nuevas.principales.join(", ")}.
            </li>
            <li>
              No son relleno: el <b style={{ color: COLOR_ACIERTO }}>{porcentaje(r.no_visibles.pct_mas_de_lo_esperado)}</b> tuvo más robos de los
              esperados en {anio}, más que la lista por conteo ({porcentaje(r.lista_conteo.pct_mas_de_lo_esperado)}).
            </li>
          </ul>
        </Tarjeta>
        <p>Con el conteo, SEDECO no las habría visto.</p>
      </Paso>

      <Paso n={4} titulo="Quién vive en las zonas que el conteo no ve" refPaso={refPaso(3)} holgado>
        <Cifra
          valor={`${entero(v.no_visibles.poblacion)} personas`}
          color={COLOR_NUEVO}
          texto={`viven en las ${entero(v.no_visibles.n)} zonas de mayor riesgo que el conteo no veía`}
        />
        <p>Según el Censo 2020, tienen menos recursos que las de la lista por conteo:</p>
        <Tarjeta className="!p-2">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-[11px] text-slate-400">
                <th className="py-1 pr-2 text-left font-medium" />
                {columnas.map((c) => (
                  <th key={c.k} className="px-1 py-1 text-right font-medium" style={{ color: c.color }}>
                    {c.t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {INDICADORES.map((ind) => (
                <tr key={ind.k} className="border-t border-white/5">
                  <td className="py-1.5 pr-2">
                    <div className="text-slate-200">{ind.t}</div>
                    {ind.nota && <div className="text-[10px] text-slate-500">{ind.nota}</div>}
                  </td>
                  {columnas.map((c) => (
                    <td key={c.k} className={`px-1 py-1.5 text-right tabular-nums ${c.k === "no_visibles" ? "font-semibold text-slate-50" : "text-slate-300"}`}>
                      {ind.f(v[c.k][ind.k])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Tarjeta>
        <div>
          <div className="mb-2 text-[11px] text-slate-400">Zonas por debajo de la escolaridad mediana de la ciudad</div>
          <div className="grid grid-cols-2 gap-2">
            <Cifra valor={porcentaje(v.lista_conteo.pct_bajo_mediana_escolaridad)} color={COLOR_CONTEO} texto="de la lista por conteo" />
            <Cifra valor={porcentaje(v.no_visibles.pct_bajo_mediana_escolaridad)} color={COLOR_NUEVO} texto="de las que el conteo no veía" />
          </div>
          <p className="mt-2 text-xs text-slate-400">
            La lista por conteo se inclina a zonas con más recursos. Las que no veía están un poco por debajo del promedio de la ciudad; el
            contraste fuerte es con la lista por conteo.
          </p>
        </div>
        <Tarjeta className="space-y-1.5 border-l-2 !border-l-accent">
          <div className="font-semibold text-slate-100">Otra razón para priorizarlas</div>
          <p>
            Donde hay menos recursos, un mismo robo es más difícil de absorber, y son zonas que una lista por conteo deja fuera. Ajustar por la
            actividad no solo encuentra más robo del esperado: lo encuentra también en zonas con menos recursos.
          </p>
        </Tarjeta>
        <p className="text-xs text-slate-500">
          Los datos describen a quienes viven en la zona, no a los dueños ni a los empleados de los negocios, y no dicen que la falta de recursos
          cause el robo: son contexto para decidir. Indicadores del índice de marginación urbana de CONAPO, ponderados por población.
        </p>
      </Paso>

      <Paso n={5} titulo="Zona por zona" refPaso={refPaso(4)} holgado>
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div className="flex overflow-hidden rounded-lg border border-white/10" role="radiogroup" aria-label="Lista de la tabla y el mapa">
            {(["riesgo", "conteo"] as const).map((m) => (
              <button
                key={m}
                role="radio"
                aria-checked={modo === m}
                onClick={() => setModo(m)}
                className={`px-2.5 py-1 ${modo === m ? "bg-white/15 font-semibold text-slate-100" : "text-slate-400 hover:bg-white/5"}`}
              >
                Por {m}
              </button>
            ))}
          </div>
          <label className="flex cursor-pointer items-center gap-1.5 text-slate-300">
            <input type="checkbox" checked={soloFallos} onChange={(e) => setSoloFallos(e.target.checked)} className="accent-slate-400" />
            Solo las que no superaron lo esperado
          </label>
          {!porConteo && (
            <label className="flex cursor-pointer items-center gap-1.5 text-slate-300">
              <input type="checkbox" checked={soloOcultas} onChange={(e) => setSoloOcultas(e.target.checked)} className="accent-cyan-400" />
              Solo las que el conteo no veía
            </label>
          )}
        </div>

        <div className="velo rounded-xl">
          <table className="w-full text-left text-xs">
            {/* Pegada bajo la barra de pasos (que mide ~44 px y se pega 16 px por encima del borde). */}
            <thead className="sticky top-7 z-[5] bg-ink-900 text-[11px] text-slate-400">
              <tr>
                <th className="px-2 py-1.5 font-medium">#</th>
                <th className="px-2 py-1.5 font-medium">Zona</th>
                {amplio && (
                  <th className="px-2 py-1.5 text-right font-medium" title={`Riesgo relativo con datos hasta ${anio - 1}`}>
                    RR
                  </th>
                )}
                <th className="px-2 py-1.5 text-right font-medium" title={`Robos esperados en ${anio} por su actividad económica`}>
                  Esperados
                </th>
                <th className="px-2 py-1.5 text-right font-medium" title={`Robos denunciados en ${anio}`}>
                  Hubo
                </th>
                <th className="px-2 py-1.5 font-medium">¿Más de lo esperado?</th>
              </tr>
            </thead>
            <tbody>
              {filas.map((z) => (
                <tr
                  key={z.cvegeo}
                  onClick={() => seleccionarAgeb(z.cvegeo, bboxPorAgeb.get(z.cvegeo))}
                  className={`cursor-pointer border-t border-white/5 transition-colors hover:bg-white/5 ${agebSel === z.cvegeo ? "bg-accent/10" : ""}`}
                >
                  <td className="px-2 py-1.5 tabular-nums text-slate-500">{porConteo ? z.rank_conteo : z.rank_riesgo}</td>
                  <td className={`px-2 py-1.5 ${amplio ? "max-w-64" : "max-w-40"}`}>
                    <div className="flex items-center gap-1">
                      <span className="truncate text-slate-200">{nombrePropio(z.colonia)}</span>
                      {!porConteo && !z.en_lista_conteo && (
                        <span className="shrink-0 rounded px-1 text-[10px]" style={{ color: COLOR_NUEVO, background: `${COLOR_NUEVO}1a` }} title="Zona de mayor riesgo que no estaba en la lista por conteo">
                          mayor riesgo
                        </span>
                      )}
                    </div>
                    <div className="truncate text-[11px] text-slate-500">{z.alcaldia}</div>
                  </td>
                  {amplio && <td className="px-2 py-1.5 text-right tabular-nums text-orange-300">×{decimal(z.rr)}</td>}
                  <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">{decimal(z.esperados)}</td>
                  <td className="px-2 py-1.5 text-right font-semibold tabular-nums text-slate-100">{entero(z.observados)}</td>
                  <td className="px-2 py-1.5">
                    <span
                      className="rounded px-1.5 py-0.5 text-[11px] font-medium"
                      style={
                        z.mas_de_lo_esperado
                          ? { color: COLOR_ACIERTO, background: `${COLOR_ACIERTO}1f` }
                          : { color: "#94a3b8", background: "rgb(255 255 255 / 0.05)" }
                      }
                    >
                      {z.mas_de_lo_esperado ? "Sí" : "No"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Paso>
    </div>
  );
}
