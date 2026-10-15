import { useMemo, useState, type ReactNode } from "react";
import { useData } from "../data/context";
import { useResumen } from "../lib/derivados";
import { ALTO_ENCABEZADO, MARGEN } from "../lib/config";
import { COLOR_NUEVO, decimal, entero, nombrePropio, pctPequeno, porcentaje } from "../lib/format";
import { useApp } from "../lib/store";
import type { Veredicto } from "../types/data";
import EjemploActividad from "./EjemploActividad";
import { GraficaBajasTamano, GraficaCorte, GraficaFlujos, GraficaSalidas } from "./GraficasCausalidad";
import { GraficaAlcaldia, GraficaTamano, GraficaTipoRobo } from "./GraficasCifraNegra";
import { Tarjeta } from "./ui";

// Cifras del análisis que no viajan en los datos de la app (pipeline/outputs/resultados.md y diseño §3, §7, §11).
const ANALISIS = {
  pctExcluidoRoboNegocio: 0.61, // farderos + "S/V dentro de negocios", 2020–2024 (40,703 de 66,321 carpetas)
  pctSinViolencia: 0.6, // composición de la variable (V2b)
  cifraNegraGrandes: 0.728, // ENVE, CDMX
  cifraNegraMicro: 0.888,
};

const ESTILO_RESULTADO: Record<Veredicto["resultado"], { texto: string; clase: string }> = {
  confirmada: { texto: "Confirmada", clase: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40" },
  rechazada: { texto: "Rechazada", clase: "bg-rose-500/15 text-rose-300 border-rose-500/40" },
  no_concluyente: { texto: "No concluyente", clase: "bg-amber-500/15 text-amber-300 border-amber-500/40" },
};

// Veredictos de las hipótesis A/B/C leídos de meta.json, como resumen ejecutivo: la lectura en una frase, las cifras que
// la sustentan y, al pie, la hipótesis formal con su estadístico (v0.8).
function Hallazgos() {
  const { meta } = useData();
  return (
    <div className="grid gap-3 lg:grid-cols-3">
      {(["A", "B", "C"] as const).map((k) => {
        const h = meta.hipotesis[k];
        const r = ESTILO_RESULTADO[h.resultado];
        return (
          <Tarjeta key={k} className="flex flex-col !p-5">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent/10 text-xs font-bold text-accent ring-1 ring-accent/30">{k}</span>
                <span className="font-semibold text-slate-100">{h.titulo}</span>
              </div>
              <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${r.clase}`}>{r.texto}</span>
            </div>
            <p className="mt-4 text-base font-semibold leading-snug text-slate-50">{h.lectura}</p>
            <ul className="mt-4 space-y-3 text-sm text-slate-300">
              {h.evidencia.map((e) => (
                <li key={e} className="flex gap-2">
                  <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                  <span>{e}</span>
                </li>
              ))}
            </ul>
            <div className="mt-auto pt-5">
              <div className="border-t border-white/10 pt-3 text-xs text-slate-500">
                <span className="font-semibold text-slate-400">Hipótesis. </span>
                {h.enunciado}
                <div className="mt-1 font-mono text-[11px]">{h.estadistico}</div>
              </div>
            </div>
          </Tarjeta>
        );
      })}
    </div>
  );
}

function Bloque({ etiqueta, titulo, children }: { etiqueta: string; titulo: string; children: ReactNode }) {
  return (
    <section className="mt-14">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-accent">{etiqueta}</div>
      <h2 className="mt-1 text-2xl font-semibold text-slate-50">{titulo}</h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Cifra({ valor, texto }: { valor: string; texto: string }) {
  return (
    <div>
      <div className="text-2xl font-semibold tabular-nums text-slate-50">{valor}</div>
      <div className="text-xs text-slate-400">{texto}</div>
    </div>
  );
}

function Dificultad({ titulo, problema, respuesta, abierta, alternar }: {
  titulo: string;
  problema: ReactNode;
  respuesta: ReactNode;
  abierta: boolean;
  alternar: () => void;
}) {
  return (
    <Tarjeta className={`flex flex-col !p-4 ${abierta ? "ring-1 ring-accent/50" : ""}`}>
      <h3 className="font-semibold text-slate-100">{titulo}</h3>
      <p className="mt-2 text-sm text-slate-300">{problema}</p>
      <p className="mt-auto border-t border-white/10 pt-2 text-xs text-slate-400">
        <span className="font-semibold text-slate-300">Cómo lo atendemos. </span>
        {respuesta}
      </p>
      <button onClick={alternar} aria-expanded={abierta} className="mt-3 self-start text-xs font-medium text-accent hover:underline">
        {abierta ? "Ocultar el análisis ▴" : "Ver el análisis ▾"}
      </button>
    </Tarjeta>
  );
}

// Detalle de una dificultad: las pruebas que hicimos y lo que queda sin resolver.
function Detalle({ titulo, limite, children }: { titulo: string; limite: ReactNode; children: ReactNode }) {
  return (
    <Tarjeta className="mt-3 !p-5">
      <h3 className="font-semibold text-slate-100">{titulo}</h3>
      <div className="mt-3 space-y-2 text-sm text-slate-300">{children}</div>
      <p className="mt-4 border-t border-white/10 pt-2 text-xs text-slate-400">
        <span className="font-semibold text-slate-300">Lo que no resolvemos. </span>
        {limite}
      </p>
    </Tarjeta>
  );
}

// Prueba con su gráfica debajo, en una tarjeta dentro del detalle. Ancha: ocupa las dos columnas, con el texto a la
// izquierda y la gráfica a la derecha.
function PruebaGrafica({ titulo, grafica, ancha, children }: { titulo: string; grafica: ReactNode; ancha?: boolean; children: ReactNode }) {
  if (ancha) {
    return (
      <div className="grid gap-3 rounded-lg border border-white/10 bg-black/20 p-3 md:col-span-2 md:grid-cols-2 md:items-center md:gap-6">
        <p>
          <b className="text-slate-100">{titulo}.</b> {children}
        </p>
        <div>{grafica}</div>
      </div>
    );
  }
  return (
    <div className="flex flex-col rounded-lg border border-white/10 bg-black/20 p-3">
      <p>
        <b className="text-slate-100">{titulo}.</b> {children}
      </p>
      <div className="mt-auto pt-3">{grafica}</div>
    </div>
  );
}

// Paso 1: página de entrada, sin mapa. Plantea el problema, define qué se mide, nombra las dificultades del análisis y
// resume los hallazgos; el mapa empieza en el paso 2.
export default function PaginaContexto() {
  const { meta, metrics, agebPorId } = useData();
  const irA = useApp((s) => s.irA);
  const { nTop, soloRiesgo } = useResumen();
  const [abierta, setAbierta] = useState<number | null>(null);
  const alternar = (i: number) => setAbierta((a) => (a === i ? null : i));

  const c = useMemo(() => {
    const filas = Object.values(metrics);
    const robos = filas.reduce((s, m) => s + meta.anios.reduce((t, a) => t + (m.por_anio[String(a)]?.observados ?? 0), 0), 0);
    const negocios = filas.reduce((s, m) => s + m.anticipado.establecimientos, 0);
    // Ejemplo: la zona más prioritaria que el conteo no ve, con al menos 3 robos al año (para que el ejemplo no sea ruido).
    const ejemplo = [...soloRiesgo]
      .map((id) => ({ id, m: metrics[id].anticipado }))
      .filter(({ m }) => m.observados >= 3)
      .sort((a, b) => a.m.rank_riesgo - b.m.rank_riesgo)[0];
    return {
      robos,
      negocios,
      agebs: filas.length,
      ejemplo,
    };
  }, [meta, metrics, soloRiesgo]);

  const ej = c.ejemplo;
  const zonaEj = ej && agebPorId.get(ej.id);
  const periodo = `${meta.anios[0]}–${meta.anios[meta.anios.length - 1]}`;

  return (
    <div
      className="scroll-fino absolute inset-0 z-20 overflow-y-auto"
      style={{
        paddingTop: 2 * MARGEN + ALTO_ENCABEZADO,
        background: "radial-gradient(1100px 520px at 75% -5%, rgb(56 189 248 / 0.09), transparent), #060a13",
      }}
    >
      <div className="mx-auto max-w-5xl px-6 pb-20 pt-10">
        {/* El problema */}
        <div className="max-w-3xl">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-accent">Robo a negocio en la Ciudad de México · {periodo}</div>
          <h1 className="mt-2 text-3xl font-semibold leading-tight text-slate-50">
            ¿Qué zonas tienen más robo a negocio del que se esperaría por su actividad económica?
          </h1>
          <p className="mt-4 text-base text-slate-300">
            La Secretaría de Desarrollo Económico y la Secretaría de Seguridad Ciudadana necesitan decidir a qué zonas comerciales dirigir prevención y apoyos. La regla más
            simple es ordenar las zonas por robos denunciados. Pero donde hay más negocios hay más robos, así que ese orden señala sobre
            todo a los polos comerciales. Este análisis identifica zonas donde el robo <b className="text-slate-100">supera a lo esperado</b> dada
            su actividad económica.
          </p>
        </div>
        <div className="mt-8 grid max-w-3xl grid-cols-3 gap-6">
          <Cifra valor={entero(c.robos)} texto={`robos a negocio denunciados (FGJ CDMX, ${periodo})`} />
          <Cifra valor={entero(c.negocios)} texto="establecimientos con actividad económica (DENUE, nov. 2023)" />
          <Cifra valor={entero(c.agebs)} texto="zonas analizadas (AGEB urbanas, INEGI)" />
        </div>

        {/* Definiciones */}
        <Bloque etiqueta="Qué medimos" titulo="Definiciones">
          <div className="space-y-3">
            <Tarjeta className="!p-4">
              <h3 className="font-semibold text-slate-100">Robo a negocio</h3>
              <div className="mt-2 grid gap-x-6 gap-y-2 md:grid-cols-2">
                <p className="text-sm text-slate-300">
                  Carpetas de investigación de la Fiscalía (FGJ) por robo a un establecimiento, con violencia o sin ella (alrededor del{" "}
                  {porcentaje(ANALISIS.pctSinViolencia)} es sin violencia), ubicadas en su AGEB por año del hecho.
                </p>
                <div className="md:border-l md:border-white/10 md:pl-6">
                  <p className="text-sm font-medium text-slate-300">Qué se excluye y por qué:</p>
                  <ul className="mt-1 space-y-1 text-sm text-slate-400">
                    <li>
                      <b className="text-slate-300">
                        Robo hormiga (<i>farderos</i>).
                      </b>{" "}
                      La Fiscalía creó estas categorías en 2020 y no las aplica igual en todas las zonas ni en todos los años, por lo que la lectura más
                      limpia se obtiene a través del robo a establecimiento con o sin violencia.
                    </li>
                    <li>
                      <b className="text-slate-300">Robo a oficinas públicas.</b> La Fiscalía lo cuenta como robo a negocio, pero la víctima no es
                      un negocio particular.
                    </li>
                  </ul>
                </div>
              </div>
            </Tarjeta>
            <Tarjeta className="!p-4">
              <h3 className="font-semibold text-slate-100">Actividad económica</h3>
              <p className="mt-2 text-sm text-slate-300">
                Los establecimientos que hay en cada AGEB según el DENUE (nov. 2023), caracterizados a través de 3 dimensiones:
              </p>
              <EjemploActividad
                nota={
                  <p className="mt-3 text-xs text-slate-500">
                    El modelo también considera la población residente y la distancia al Metro o Metrobús, como contexto urbano.
                  </p>
                }
              />
            </Tarjeta>
          </div>
        </Bloque>

        {/* Dificultades */}
        <Bloque etiqueta="Lo que limita a los datos de denuncias" titulo="Dos dificultades del análisis">
          <div className="grid gap-3 md:grid-cols-2">
            <Dificultad
              titulo="Cifra negra: solo vemos lo que se denuncia"
              abierta={abierta === 0}
              alternar={() => alternar(0)}
              problema={
                <>
                  En la CDMX, entre el {pctPequeno(ANALISIS.cifraNegraGrandes)} (empresas grandes) y el {pctPequeno(ANALISIS.cifraNegraMicro)}{" "}
                  (micronegocios) de los delitos contra negocios no llega a una carpeta (ENVE, INEGI). Si unas zonas denuncian más que otras, la
                  lista señalaría dónde se denuncia, no dónde se roba.
                </>
              }
              respuesta={
                <>
                  Sacamos el robo hormiga, que depende de quién denuncia, y probamos si la denuncia cambia con la alcaldía, el tamaño del
                  negocio o el tipo de robo. La lista de prioridades se mantiene.
                </>
              }
            />
            <Dificultad
              titulo="Causalidad: ¿el robo ahuyenta negocios?"
              abierta={abierta === 1}
              alternar={() => alternar(1)}
              problema={
                <>
                  El comercio atrae robo, pero el robo también podría ahuyentar al comercio. Si los negocios se van de donde les roban, la zona
                  aparece con pocos negocios, su robo esperado sale bajo y su exceso se exagera.
                </>
              }
              respuesta={
                <>
                  Contamos los negocios con un censo anterior al robo analizado y seguimos a cada negocio de 2018 a 2023: donde había más
                  robo en 2016–2018 no cerraron más negocios y abrieron un poco más. Reportamos asociaciones, no causas.
                </>
              }
            />
          </div>

          {abierta === 0 && (
            <Detalle
              titulo="Cifra negra: probamos cada fuente de variación que se puede medir"
              limite={
                <>
                  En una zona concreta no podemos separar más robo de más denuncia: no existe un dato de denuncia de negocios por zona (la
                  ENVE solo llega a nivel estatal). Por eso la lista dice dónde mirar primero, no cuánto robo hay.
                </>
              }
            >
              <p>
                Cifra negra homogénea en diferentes zonas de ciudad no genera sesgos. El problema sería que varíe entre ellas, así que
                probamos cada fuente posible de variación:
              </p>
              {/* lab/resultados/decisiones_lab.md (S1); pipeline/outputs/modelos/sesgo_denuncia.*, cifra_negra_alcaldia.md (04e);
                  pipeline/outputs/resultados.md (04c) */}
              <div className="grid gap-3 md:grid-cols-2">
                <PruebaGrafica titulo="Alcaldía" grafica={<GraficaAlcaldia />}>
                  Los hogares denuncian entre el 9% y el 13.5% de sus delitos según la alcaldía, lo que genera diferencias que no son estadísticamente
                  significativas entre ellas (ENVIPE).
                </PruebaGrafica>
                <PruebaGrafica titulo="Tamaño y cantidad de negocios" grafica={<GraficaTamano />}>
                  Según la ENVE, las micro denuncian el 11% de sus delitos y las grandes el 27%. La denuncia esperada de una zona es el
                  promedio según sus tamaños: con 97 micro y 3 grandes, (97 × 11% + 3 × 27%) / 100 ≈ 11.5%. Si la zona duplica sus negocios
                  con la misma mezcla (194 micro y 6 grandes), sigue en 11.5%. Así se ve en los datos: en todas las zonas solo el 2% o 3% de
                  los negocios tiene más de 50 personas, y la denuncia esperada apenas se mueve. Para sesgar el
                  resultado tendría que caer 3.3% por duplicación.
                </PruebaGrafica>
                <PruebaGrafica titulo="Tipo de robo" grafica={<GraficaTipoRobo />} ancha>
                  Con solo robo con violencia, el orden de las zonas por riesgo se parece al de todo el robo
                  (correlación 0.75). La diferencia es sobre todo ruido: el robo con violencia es solo el 40% de los casos.
                </PruebaGrafica>
              </div>
            </Detalle>
          )}

          {abierta === 1 && (
            <Detalle
              titulo="Causalidad: no encontramos que el robo ahuyente negocios"
              limite={
                <>
                  Es una sola ventana de cinco años con la pandemia dentro, solo con robo denunciado, y muestra asociación, no causa.
                </>
              }
            >
              {/* pipeline/outputs/resultados.md (E de 2016); resultados_dinamica.md (D, D2a, D2b);
                  dinamica/rotacion_ritmos.csv y rotacion_quintiles.csv (08b) */}
              <div className="grid gap-3 md:grid-cols-2">
                <PruebaGrafica titulo="Los negocios se cuentan antes del robo" grafica={<GraficaCorte />}>
                  El DENUE se renueva por completo solo con cada Censo Económico; entre censos casi no cambia (sale el 4% de los negocios y
                  entra el 6%). Por eso el 94% de los negocios del DENUE de noviembre de 2023 son los del censo de 2019, anterior al robo
                  analizado. Con los negocios de enero de 2016, anteriores a todo el robo analizado, la relación entre robo y negocios casi no cambia.
                </PruebaGrafica>
                <PruebaGrafica titulo="Cierres y aperturas" grafica={<GraficaSalidas />}>
                  El recambio real aparece al llegar el censo de 2024: en cinco años sale el 35% de los negocios y entran otros tantos (33%),
                  como en la Demografía de los Negocios del INEGI. Cambian los negocios, no cuántos hay: el tamaño comercial de cada zona
                  casi no se mueve (correlación 0.99 entre 2019 y 2024). En las zonas con más robo previo (2016–2018) sale la misma proporción.
                </PruebaGrafica>
                <PruebaGrafica titulo="Aperturas, saldo y rotación" grafica={<GraficaFlujos />}>
                  Con controles, esas zonas pierden la misma proporción de negocios (razón 0.99, IC 0.98–1.01), reciben un poco más de
                  negocios nuevos (1.02) y su saldo no cae. Por eso rotan algo más (74% frente a 65–68% en el resto): por más aperturas,
                  no por más cierres. El robo sigue a la actividad, no la expulsa.
                </PruebaGrafica>
                <PruebaGrafica titulo="Una señal a vigilar" grafica={<GraficaBajasTamano />}>
                  Las bajas que el INEGI registró entre censos (solo el 4% de los negocios) sí suben con el robo, por los negocios medianos
                  (11 a 50 personas). Va en contra del resultado principal; el próximo censo dirá si son cierres reales.
                </PruebaGrafica>
              </div>
            </Detalle>
          )}
        </Bloque>

        {/* Método */}
        <Bloque etiqueta="Cómo lo resolvemos" titulo="Observado contra esperado">
          <div className="grid items-start gap-3 md:grid-cols-2">
            <div className="text-sm text-slate-300">
              <p>
                Nuestro modelo aprende cuántos robos suele tener una zona con cierto número, giro y tamaño de negocios. Para cada
                zona, lo esperado se calcula sin ver sus propios robos. Luego comparamos:
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
                <span className="velo rounded-lg px-2.5 py-1.5 text-slate-100">Robos denunciados</span>
                <span className="text-slate-500">÷</span>
                <span className="velo rounded-lg px-2.5 py-1.5 text-slate-100">Robos esperados por su actividad</span>
                <span className="text-slate-500">→</span>
                <span className="velo rounded-lg px-2.5 py-1.5 text-slate-100">Ajuste por evidencia</span>
                <span className="text-slate-500">→</span>
                <span className="rounded-lg border border-accent/40 bg-accent/10 px-2.5 py-1.5 font-medium text-accent">Riesgo ajustado</span>
              </div>
              <p className="mt-3 text-slate-400">
                Arriba de ×1, la zona tiene más robo del que le corresponde. Pero con pocos robos la división engaña: si una zona espera
                medio robo y tiene 2, sale ×4, y con uno menos bajaría a ×2. Por eso cada zona se combina con lo típico de la ciudad (×1)
                según cuánta evidencia tiene: si espera muchos robos, manda su propio dato; si espera pocos, se acerca a ×1. Cuánto acercarla lo aprendemos
                de cómo varían los robos en toda la ciudad. La lista mira al año siguiente ({meta.anticipado.anio}) y da más peso a los años
                recientes.
              </p>
            </div>
            {ej && zonaEj && (
              <Tarjeta className="!p-4">
                <div className="text-[11px] uppercase tracking-wider text-slate-500">Ejemplo</div>
                <div className="mt-1 font-semibold text-slate-100">
                  {nombrePropio(zonaEj.colonia)} <span className="font-normal text-slate-400">· {zonaEj.alcaldia}</span>
                </div>
                <div className="mt-3 grid grid-cols-4 gap-2 text-center">
                  <div>
                    <div className="text-xl font-semibold tabular-nums text-slate-100">{decimal(ej.m.observados)}</div>
                    <div className="text-[11px] text-slate-400">robos al año (recientes)</div>
                  </div>
                  <div>
                    <div className="text-xl font-semibold tabular-nums text-slate-300">{decimal(ej.m.esperados)}</div>
                    <div className="text-[11px] text-slate-400">esperados dadas sus características</div>
                  </div>
                  <div>
                    <div className="text-xl font-semibold tabular-nums text-slate-300">×{entero(ej.m.observados / ej.m.esperados)}</div>
                    <div className="text-[11px] text-slate-400">riesgo directo</div>
                  </div>
                  <div>
                    <div className="text-xl font-semibold tabular-nums text-orange-300">×{decimal(ej.m.rr)}</div>
                    <div className="text-[11px] text-slate-400">riesgo ajustado</div>
                  </div>
                </div>
                <p className="mt-3 text-xs text-slate-400">
                  Espera solo {decimal(ej.m.esperados)} robos al año: con tan poca evidencia, su ×{entero(ej.m.observados / ej.m.esperados)} se
                  acerca a ×1 y queda en ×{decimal(ej.m.rr)}. Aun así sigue muy arriba.
                </p>
                <p className="mt-2 text-xs text-slate-400">
                  Por conteo ocupa el puesto {entero(ej.m.rank_conteo)}: no entraría entre las {nTop} zonas con más robos. Por riesgo ajustado es
                  de las primeras.{" "}
                  <span style={{ color: COLOR_NUEVO }}>
                    Como ella, {soloRiesgo.size} zonas prioritarias con más robos de lo esperado son invisibles para el conteo.
                  </span>
                </p>
              </Tarjeta>
            )}
          </div>
        </Bloque>

        {/* Hallazgos */}
        <Bloque etiqueta="Resumen ejecutivo" titulo="Hallazgos">
          <p className="mb-3 max-w-3xl text-sm text-slate-400">
            Tres hipótesis, cada una contra una regla más simple que SEDECO podría usar: contar robos, ignorar el giro o priorizar por conteo.
          </p>
          <Hallazgos />
        </Bloque>

        {/* Siguiente paso */}
        <div className="mt-12 flex flex-wrap items-center gap-3">
          <button
            onClick={() => irA("anormal")}
            className="rounded-xl bg-accent px-5 py-3 text-sm font-semibold text-ink-950 shadow-[0_0_24px_rgb(56_189_248/0.35)] transition-transform hover:scale-[1.02]"
          >
            Ver el mapa: del comercio al riesgo →
          </button>
          <button
            onClick={() => irA("consulta")}
            className="rounded-xl border border-accent/50 px-5 py-3 text-sm font-medium text-accent transition-colors hover:bg-accent/10"
          >
            Tengo un negocio: consulta mi zona
          </button>
        </div>

        {/* Cobertura y fuentes */}
        <footer className="mt-14 grid gap-6 border-t border-white/10 pt-6 text-xs text-slate-500 md:grid-cols-2">
          <ul className="space-y-0.5">
            <li className="font-semibold text-slate-400">Fuentes</li>
            {meta.fuentes.map((f) => (
              <li key={f.nombre}>
                <a href={f.url} target="_blank" rel="noreferrer" className="hover:text-slate-300">
                  {f.nombre}
                </a>{" "}
                · {f.corte}
              </li>
            ))}
          </ul>
        </footer>
      </div>
    </div>
  );
}
