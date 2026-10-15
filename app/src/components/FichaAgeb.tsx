import { useMemo, type ReactNode } from "react";
import { useData } from "../data/context";
import { metricaDe } from "../lib/derivados";
import { decimal, decimal2, entero, nombrePropio, pctPequeno, porcentaje } from "../lib/format";
import { useApp } from "../lib/store";
import type { CategoriaRiesgo, CensoFicha, GiroEfectivo, GrupoComposicion } from "../types/data";
import { Dato, InsigniaCategoria, Seccion } from "./ui";

// Ficha de la AGEB (contrato v0.9, ficha.json): contexto descriptivo para leer la zona. Nada de esto entra al modelo
// ni cambia la lista. La comparación con su colonia y su alcaldía usa la lista 2025, así que solo se muestra ahí.

// Nombre corto de cada grupo de giro (el nombre largo de giros.json va en el tooltip). Los dos de efectivo, en ámbar.
const GRUPO: Record<GrupoComposicion, { corto: string; color: string }> = {
  efectivo_barrio: { corto: "Alto efectivo tradicional", color: "#f59e0b" },
  efectivo_formal: { corto: "Alto efectivo formal", color: "#f97316" },
  comercio_menor: { corto: "Otro comercio al por menor", color: "#38bdf8" },
  comercio_mayor: { corto: "Comercio al por mayor", color: "#0ea5e9" },
  manufactura: { corto: "Manufactura", color: "#a78bfa" },
  alojamiento_alim: { corto: "Hospedaje", color: "#f472b6" },
  serv_prof_fin: { corto: "Oficinas y servicios profesionales", color: "#34d399" },
  serv_salud_educ: { corto: "Salud, educación y recreación", color: "#2dd4bf" },
  serv_otros: { corto: "Reparación y servicios personales", color: "#94a3b8" },
  otros: { corto: "Otros (gobierno, construcción, transporte…)", color: "#64748b" },
};
const GIRO_EF: Record<GiroEfectivo, { nombre: string; grupo: GrupoComposicion }> = {
  abarrotes: { nombre: "Abarrotes", grupo: "efectivo_barrio" },
  alim_otros: { nombre: "Carnicería, pollería, frutería…", grupo: "efectivo_barrio" },
  tortillerias: { nombre: "Tortillería", grupo: "efectivo_barrio" },
  licores: { nombre: "Venta de licores", grupo: "efectivo_barrio" },
  restaurantes: { nombre: "Restaurante o taquería", grupo: "efectivo_formal" },
  farmacias: { nombre: "Farmacia", grupo: "efectivo_formal" },
  minisuper: { nombre: "Minisúper", grupo: "efectivo_formal" },
  gasolineras: { nombre: "Gasolinera", grupo: "efectivo_formal" },
  bares: { nombre: "Bar", grupo: "efectivo_formal" },
  empeno_cambio: { nombre: "Casa de empeño o cambio", grupo: "efectivo_formal" },
};

const conSigno = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${pctPequeno(Math.abs(x))}`;
const Nota = ({ children }: { children: ReactNode }) => <p className="mt-2 text-[11px] text-slate-500">{children}</p>;

export default function FichaAgeb({ id, conComparacion }: { id: string; conComparacion: boolean }) {
  const { ficha, agebPorId, alcaldiaPorNombre } = useData();
  const f = ficha.agebs[id];
  const props = agebPorId.get(id);
  const alc = props ? alcaldiaPorNombre.get(props.alcaldia) : undefined;
  if (!f) return null;
  return (
    <>
      {conComparacion && <Comparacion id={id} />}
      <Actividad id={id} />
      {f.dinamica && <Cambio id={id} />}
      <Residentes id={id} />
      {alc && ficha.alcaldias[alc.cve_mun] && <Encuestas cveMun={alc.cve_mun} nombre={alc.alcaldia} />}
    </>
  );
}

// ---------- La AGEB frente a su colonia y su alcaldía (lista 2025) ----------
function Comparacion({ id }: { id: string }) {
  const datos = useData();
  const { metrics, agebPorId, alcaldiaPorNombre, coloniasMetrics, alcaldiasMetrics, bboxPorColonia, bboxPorAlcaldia } = datos;
  const { periodo, seleccionarZona } = useApp();
  const props = agebPorId.get(id);
  const alc = props ? alcaldiaPorNombre.get(props.alcaldia) : undefined;

  // Puesto por riesgo entre las AGEB de su misma alcaldía.
  const enAlcaldia = useMemo(() => {
    if (!props) return undefined;
    const ids = Object.keys(metrics).filter((cv) => agebPorId.get(cv)?.alcaldia === props.alcaldia);
    const propio = metricaDe(metrics[id], periodo).rank_riesgo;
    return { rank: 1 + ids.filter((cv) => metricaDe(metrics[cv], periodo).rank_riesgo < propio).length, n: ids.length };
  }, [metrics, agebPorId, props, id, periodo]);

  const m = metrics[id] ? metricaDe(metrics[id], periodo) : undefined;
  const mc = props && coloniasMetrics[props.cve_col] ? metricaDe(coloniasMetrics[props.cve_col], periodo) : undefined;
  const ma = alc && alcaldiasMetrics[alc.cve_mun] ? metricaDe(alcaldiasMetrics[alc.cve_mun], periodo) : undefined;
  if (!m || !props) return null;

  const filas: { nivel: ReactNode; categoria?: CategoriaRiesgo; rr: number; puesto: string }[] = [
    { nivel: "Esta AGEB", categoria: m.categoria, rr: m.rr, puesto: `#${entero(m.rank_riesgo)} de ${entero(Object.keys(metrics).length)}` },
  ];
  if (enAlcaldia) filas.push({ nivel: `…entre las AGEB de ${props.alcaldia}`, rr: m.rr, puesto: `#${entero(enAlcaldia.rank)} de ${entero(enAlcaldia.n)}` });
  if (mc)
    filas.push({
      nivel: (
        <button onClick={() => seleccionarZona({ nivel: "colonia", id: props.cve_col }, bboxPorColonia.get(props.cve_col))} className="text-left hover:text-accent">
          Colonia {nombrePropio(props.colonia)}
        </button>
      ),
      categoria: mc.categoria,
      rr: mc.rr,
      puesto: `#${entero(mc.rank_riesgo)} de ${entero(Object.keys(coloniasMetrics).length)}`,
    });
  if (ma && alc)
    filas.push({
      nivel: (
        <button onClick={() => seleccionarZona({ nivel: "alcaldia", id: alc.cve_mun }, bboxPorAlcaldia.get(alc.cve_mun))} className="text-left hover:text-accent">
          Alcaldía {alc.alcaldia}
        </button>
      ),
      categoria: ma.categoria,
      rr: ma.rr,
      puesto: `#${entero(ma.rank_riesgo)} de ${entero(Object.keys(alcaldiasMetrics).length)}`,
    });

  return (
    <Seccion titulo="Frente a su colonia y su alcaldía">
      <div className="divide-y divide-white/5 text-xs">
        {filas.map((f, i) => (
          <div key={i} className="flex items-center gap-2 py-1.5">
            <div className="min-w-0 flex-1 truncate text-slate-300">{f.nivel}</div>
            {f.categoria && <InsigniaCategoria categoria={f.categoria} texto={`×${decimal(f.rr)}`} />}
            <div className="w-20 text-right tabular-nums text-slate-400">{f.puesto}</div>
          </div>
        ))}
      </div>
      <Nota>
        Riesgo relativo y puesto por riesgo dentro de cada nivel (1 = más robo del que le corresponde por su actividad). La colonia es la que cubre
        la mayor parte de la AGEB.
      </Nota>
    </Seccion>
  );
}

// ---------- Actividad económica: tamaño, densidad y composición por giro (DENUE 2023-11) ----------
function Actividad({ id }: { id: string }) {
  const { ficha, giros, metrics } = useData();
  const f = ficha.agebs[id];
  const c = ficha.ciudad;
  const E = metrics[id]?.anticipado.establecimientos ?? 0;
  const nombreLargo = new Map(giros.map((g) => [g.id, g.nombre]));
  const grupos = (Object.keys(GRUPO) as GrupoComposicion[]).sort((a, b) => f.composicion[b] - f.composicion[a]);
  const escala = Math.max(...grupos.map((g) => Math.max(f.composicion[g], c.composicion[g])));
  // Lo que más distingue a la zona: el grupo con mayor sobrerrepresentación entre los que pesan al menos 5%.
  const distintivo = grupos
    .filter((g) => f.composicion[g] >= 0.05 && c.composicion[g] > 0)
    .sort((a, b) => f.composicion[b] / c.composicion[b] - f.composicion[a] / c.composicion[a])[0];
  const veces = distintivo ? f.composicion[distintivo] / c.composicion[distintivo] : 0;
  const giros_ef = (Object.keys(f.efectivo) as GiroEfectivo[]).sort((a, b) => (f.efectivo[b] ?? 0) - (f.efectivo[a] ?? 0));

  return (
    <Seccion titulo="Actividad económica">
      <div className="grid grid-cols-2 gap-2">
        <Dato etiqueta="Establecimientos" valor={entero(E)} nota="DENUE nov. 2023" />
        <Dato etiqueta="Densidad" valor={`${entero(f.densidad)} /km²`} nota={`Más que el ${porcentaje(f.pct_densidad)} de las AGEB`} />
        <Dato etiqueta="Personal ocupado" valor={`≈ ${entero(f.personal_ocupado)}`} nota={`≈ ${decimal(E ? f.personal_ocupado / E : 0)} por negocio`} />
        <Dato etiqueta="Negocios de 51 personas o más" valor={pctPequeno(f.pct_grandes)} nota={`Ciudad: ${pctPequeno(c.pct_grandes)}`} />
      </div>

      <div className="mt-3 mb-1.5 flex items-baseline justify-between text-[11px] text-slate-400">
        <span>Composición por giro</span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-px bg-slate-200" /> ciudad
        </span>
      </div>
      <div className="space-y-1.5">
        {grupos.map((g) => (
          <div key={g} title={nombreLargo.get(g) ?? GRUPO[g].corto}>
            <div className="flex justify-between text-[11px]">
              <span className="text-slate-300">{GRUPO[g].corto}</span>
              <span className="tabular-nums text-slate-200">
                {pctPequeno(f.composicion[g])} <span className="text-slate-500">· {pctPequeno(c.composicion[g])}</span>
              </span>
            </div>
            <div className="relative mt-0.5 h-1.5 rounded-full bg-white/10">
              <div className="h-full rounded-full" style={{ width: `${(f.composicion[g] / escala) * 100}%`, background: GRUPO[g].color }} />
              <div className="absolute -top-0.5 h-2.5 w-px bg-slate-200" style={{ left: `${(c.composicion[g] / escala) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      {distintivo && veces >= 1.3 && (
        <p className="mt-2 text-xs text-slate-300">
          Lo que más la distingue: <b className="text-slate-100">{GRUPO[distintivo].corto.toLowerCase()}</b>, {decimal(veces)} veces su peso en la
          ciudad.
        </p>
      )}

      {giros_ef.length > 0 && (
        <>
          <div className="mt-3 mb-1.5 text-[11px] text-slate-400">Giros de alto efectivo (parte de todos los negocios de la AGEB)</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
            {giros_ef.map((g) => (
              <div key={g} className="flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: GRUPO[GIRO_EF[g].grupo].color }} />
                <span className="min-w-0 flex-1 truncate text-slate-300">{GIRO_EF[g].nombre}</span>
                <span className="tabular-nums text-slate-200">{pctPequeno(f.efectivo[g] ?? 0)}</span>
              </div>
            ))}
          </div>
        </>
      )}
      <Nota>
        Es el mismo corte del DENUE que usa el modelo. Personal ocupado estimado con el punto medio del estrato de cada negocio. Ciudad: promedio de
        todos los negocios de las AGEB urbanas.
      </Nota>
    </Seccion>
  );
}

// ---------- Negocios que salieron y entraron (DENUE 2019-11 → 2024-11) ----------
function Cambio({ id }: { id: string }) {
  const { ficha } = useData();
  const d = ficha.agebs[id].dinamica!;
  const c = ficha.ciudad;
  const tasa = (x: number) => (d.antes ? x / d.antes : 0);
  return (
    <Seccion titulo="Negocios que abren y cierran (2018 → 2023)">
      <div className="grid grid-cols-3 gap-2">
        <Dato etiqueta="Cambio neto" valor={d.antes ? conSigno(d.despues / d.antes - 1) : "—"} nota={`${entero(d.antes)} → ${entero(d.despues)}`} />
        <Dato etiqueta="Salieron" valor={entero(d.salidas)} nota={`${porcentaje(tasa(d.salidas))} (ciudad ${porcentaje(c.dinamica.salidas / c.dinamica.antes)})`} />
        <Dato etiqueta="Entraron" valor={entero(d.entradas)} nota={`${porcentaje(tasa(d.entradas))} (ciudad ${porcentaje(c.dinamica.entradas / c.dinamica.antes)})`} />
      </div>
      <Nota>
        DENUE de nov. 2019 y nov. 2024 (Censos Económicos 2019 y 2024), comparando cada negocio por su clave. Cambio mediano por AGEB en la ciudad:{" "}
        {conSigno(c.cambio_mediano)}. No encontramos evidencia de que las zonas con más robo del esperado pierdan más negocios.
      </Nota>
    </Seccion>
  );
}

// ---------- Quién vive aquí (Censo 2020) ----------
const CENSO: { k: keyof CensoFicha; t: string; f: (x: number) => string }[] = [
  { k: "escolaridad", t: "Escolaridad promedio (15 años o más)", f: (x) => `${decimal(x)} años` },
  { k: "pct_sin_basica", t: "Sin secundaria terminada (15 años o más)", f: porcentaje },
  { k: "pct_sin_salud", t: "Sin afiliación a servicios de salud", f: porcentaje },
  { k: "pct_sin_internet", t: "Viviendas sin internet", f: porcentaje },
];

function Residentes({ id }: { id: string }) {
  const { ficha, agebPorId } = useData();
  const f = ficha.agebs[id].censo;
  const c = ficha.ciudad.censo;
  return (
    <Seccion titulo="Quién vive aquí (Censo 2020)">
      <div className="text-xs">
        <div className="flex justify-between border-b border-white/5 pb-1 text-[11px] text-slate-500">
          <span>Población</span>
          <span className="tabular-nums text-slate-200">{entero(agebPorId.get(id)?.poblacion_2020 ?? 0)} personas</span>
        </div>
        <div className="flex justify-end gap-3 pt-1.5 text-[10px] uppercase tracking-wider text-slate-500">
          <span className="w-16 text-right">AGEB</span>
          <span className="w-16 text-right">Ciudad</span>
        </div>
        {CENSO.map(({ k, t, f: fmt }) => {
          const v = f[k];
          const ref = c[k];
          return (
            <div key={k} className="flex items-center gap-3 py-1">
              <span className="min-w-0 flex-1 text-slate-300">{t}</span>
              <span className="w-16 text-right tabular-nums text-slate-100" title={v === null ? "Dato reservado por INEGI (confidencialidad)" : undefined}>
                {v === null ? "—" : fmt(v)}
              </span>
              <span className="w-16 text-right tabular-nums text-slate-500">{ref === null ? "—" : fmt(ref)}</span>
            </div>
          );
        })}
      </div>
      <Nota>Describe a quienes viven en la AGEB, no a los dueños ni a los empleados de los negocios. Es contexto para priorizar; no explica el robo.</Nota>
    </Seccion>
  );
}

// ---------- Su alcaldía en las encuestas (ENVIPE y ENSU) ----------
const ENSU: { k: "inseg_ciudad" | "inseg_calle" | "inseg_mercado"; t: string }[] = [
  { k: "inseg_ciudad", t: "Considera inseguro vivir en su ciudad" },
  { k: "inseg_calle", t: "Se siente inseguro en las calles que usa" },
  { k: "inseg_mercado", t: "Se siente inseguro en los mercados" },
];

function Encuestas({ cveMun, nombre }: { cveMun: string; nombre: string }) {
  const { ficha } = useData();
  const a = ficha.alcaldias[cveMun];
  const c = ficha.ciudad;
  const ic = (v: number, ee: number) => `IC 95%: ${porcentaje(Math.max(0, v - 1.96 * ee))}–${porcentaje(v + 1.96 * ee)}`;
  return (
    <Seccion titulo={`${nombre} en las encuestas`}>
      <div className="grid grid-cols-2 gap-2">
        <Dato
          etiqueta="Delitos que los hogares denuncian"
          valor={pctPequeno(a.envipe.denuncia)}
          nota={<span title={ic(a.envipe.denuncia, a.envipe.ee)}>CDMX: {pctPequeno(c.envipe_denuncia)} · ENVIPE</span>}
        />
        <Dato
          etiqueta="Hogares con víctima de robo en el semestre"
          valor={porcentaje(a.ensu.hog_vict_robo.valor)}
          nota={`#${a.ensu.hog_vict_robo.rank} de 16 alcaldías · ENSU`}
        />
      </div>
      <p className="mt-2 text-[11px] text-slate-500">
        En la ENVIPE (delitos de 2020–2024), las alcaldías denuncian casi lo mismo (no se rechaza que sean iguales, p = {decimal2(c.envipe_p_iguales)}).
        La cifra negra es enorme en toda la ciudad, pero no cambia el orden de la lista.
      </p>

      <div className="mt-3 mb-1 flex justify-between text-[11px] text-slate-400">
        <span>Percepción de quienes viven aquí (ENSU {c.anio_ensu})</span>
        <span>puesto de 16</span>
      </div>
      <div className="text-xs">
        {ENSU.map(({ k, t }) => {
          const x = a.ensu[k];
          return (
            <div key={k} className="flex items-center gap-2 py-1" title={ic(x.valor, x.ee)}>
              <span className="min-w-0 flex-1 text-slate-300">{t}</span>
              <span className="w-12 text-right tabular-nums text-slate-100">{porcentaje(x.valor)}</span>
              <span className="w-14 text-right tabular-nums text-slate-500">#{x.rank}</span>
            </div>
          );
        })}
      </div>
      <Nota>
        Promedio de los trimestres de {c.anio_ensu}, población de 18 años o más (1 = la alcaldía con el valor más alto). Mide a los residentes de la
        alcaldía, dondequiera que les pase. No entra al modelo.
      </Nota>
    </Seccion>
  );
}
