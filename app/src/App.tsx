import { useEffect, useState, type CSSProperties } from "react";
import { AvisoCambio, Leyenda, SelectorNivel } from "./components/ControlesMapa";
import Encabezado from "./components/Encabezado";
import MapaAgeb from "./components/MapaAgeb";
import PanelAgeb from "./components/PanelAgeb";
import PanelAnormal from "./components/PanelAnormal";
import PanelConsulta from "./components/PanelConsulta";
import PanelLateral from "./components/PanelLateral";
import PanelPrioridades from "./components/PanelPrioridades";
import PanelValidacion from "./components/PanelValidacion";
import PanelZona from "./components/PanelZona";
import PaginaContexto from "./components/PaginaContexto";
import { cargarDatos, type AppData } from "./data";
import { DataContext } from "./data/context";
import { useApp, useMargenesMapa } from "./lib/store";

function Panel() {
  const { vista, agebSel, zonaSel } = useApp();
  if (agebSel && vista !== "consulta") return <PanelAgeb id={agebSel} />;
  if (zonaSel && vista !== "consulta") return <PanelZona zona={zonaSel} />;
  switch (vista) {
    case "contexto":
      return null;
    case "anormal":
      return <PanelAnormal />;
    case "validar":
      return <PanelValidacion />;
    case "priorizar":
      return <PanelPrioridades />;
    case "consulta":
      return <PanelConsulta />;
  }
}

function Principal() {
  const { vista, agebSel, zonaSel } = useApp();
  const m = useMargenesMapa();
  // "¿Habría funcionado?" es un recorrido: el panel entero se desplaza y la tabla queda al final.
  const tabla = vista === "priorizar" && !agebSel && !zonaSel;
  // Lo que tapan el encabezado y el panel: lo usan los controles de MapLibre (index.css).
  const panelAbierto = useApp((s) => s.panel.abierto);
  // Con el panel oculto, los botones de zoom bajan para dejar sitio a "Mostrar panel".
  const variables = { "--m-sup": `${m.top + (panelAbierto ? 0 : 46)}px`, "--m-der": `${m.right}px` } as CSSProperties;
  return (
    <div className="relative h-full overflow-hidden bg-ink-950" style={variables}>
      <div className="absolute inset-0">
        <MapaAgeb />
      </div>

      <Encabezado />

      {/* Paso 1: página de contexto sin mapa. El mapa sigue montado debajo para que ya esté cargado al pasar al paso 2. */}
      {vista === "contexto" ? (
        <PaginaContexto />
      ) : (
        <>
          {/* Controles del mapa: se acomodan en el espacio que deja libre el panel (los botones de zoom ocupan ~50 px). */}
          <div
            className="pointer-events-none absolute left-3 z-10 flex flex-col items-start gap-2 transition-[right] duration-300 ease-out"
            style={{ top: m.top, right: m.right + 56 }}
          >
            <div className="flex flex-wrap items-start gap-2 [&>*]:pointer-events-auto">
              <SelectorNivel />
            </div>
            <div className="[&>*]:pointer-events-auto">
              <AvisoCambio />
            </div>
          </div>
          <div className="absolute bottom-3 left-3 z-10">
            <Leyenda />
          </div>

          <PanelLateral llave={`${vista}·${agebSel ?? ""}·${zonaSel ? `${zonaSel.nivel}:${zonaSel.id}` : ""}`} tabla={tabla}>
            <Panel />
          </PanelLateral>
        </>
      )}
    </div>
  );
}

export default function App() {
  const [datos, setDatos] = useState<AppData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    cargarDatos().then(setDatos, (e: Error) => setError(e.message));
  }, []);

  if (error)
    return (
      <div className="flex h-full items-center justify-center p-8 text-center">
        <div>
          <div className="text-lg font-semibold text-rose-300">No se pudieron cargar los datos</div>
          <div className="mt-2 text-sm text-slate-400">{error}</div>
        </div>
      </div>
    );
  if (!datos) return <div className="flex h-full items-center justify-center text-sm text-slate-400">Cargando datos…</div>;

  return (
    <DataContext.Provider value={datos}>
      <Principal />
    </DataContext.Provider>
  );
}
