import { useEffect, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { ALTO_ENCABEZADO, MARGEN, PANEL } from "../lib/config";
import { useAnchoPanel, useApp } from "../lib/store";

const PASO_TECLADO = 40;

function Chevron({ hacia }: { hacia: "izq" | "der" }) {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden>
      <path d={hacia === "der" ? "M6 3l5 5-5 5" : "M10 3L5 8l5 5"} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Panel flotante a la derecha del mapa. Su borde izquierdo se arrastra para cambiar el ancho
// (doble clic: ancho por defecto) y se puede ocultar para ver el mapa completo.
export default function PanelLateral({ children, llave, tabla }: { children: ReactNode; llave: string; tabla: boolean }) {
  const ancho = useAnchoPanel();
  const abierto = useApp((s) => s.panel.abierto);
  const setAncho = useApp((s) => s.setAnchoPanel);
  const setAbierto = useApp((s) => s.setPanelAbierto);

  // Si la ventana se achica, el panel también, para no dejar al mapa sin espacio.
  useEffect(() => {
    const ajustar = () => setAncho(useApp.getState().panel.ancho);
    window.addEventListener("resize", ajustar);
    return () => window.removeEventListener("resize", ajustar);
  }, [setAncho]);

  const arrastrar = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const x0 = e.clientX;
    const a0 = ancho;
    const raiz = document.documentElement;
    raiz.dataset.arrastrando = "";
    const mover = (ev: globalThis.PointerEvent) => setAncho(a0 + x0 - ev.clientX);
    const soltar = () => {
      delete raiz.dataset.arrastrando;
      window.removeEventListener("pointermove", mover);
      window.removeEventListener("pointerup", soltar);
      window.removeEventListener("pointercancel", soltar);
    };
    window.addEventListener("pointermove", mover);
    window.addEventListener("pointerup", soltar);
    window.addEventListener("pointercancel", soltar);
  };

  const teclado = (e: KeyboardEvent<HTMLDivElement>) => {
    const cambios: Record<string, number> = {
      ArrowLeft: ancho + PASO_TECLADO,
      ArrowRight: ancho - PASO_TECLADO,
      Home: PANEL.max,
      End: PANEL.min,
    };
    if (e.key in cambios) {
      e.preventDefault();
      setAncho(cambios[e.key]);
    }
  };

  const top = 2 * MARGEN + ALTO_ENCABEZADO;

  return (
    <>
      <aside
        aria-label="Panel de información"
        inert={!abierto}
        className={`vidrio-panel absolute bottom-3 right-3 z-20 flex flex-col rounded-2xl transition-[translate,opacity] duration-300 ease-out ${
          abierto ? "" : "pointer-events-none translate-x-[calc(100%+24px)] opacity-0"
        }`}
        style={{ top, width: ancho }}
      >
        {/* Borde arrastrable: toda la altura, con un asa visible al centro. */}
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Ancho del panel: arrastra o usa las flechas"
          aria-valuenow={ancho}
          aria-valuemin={PANEL.min}
          aria-valuemax={PANEL.max}
          tabIndex={0}
          title="Arrastra para cambiar el ancho · doble clic para restablecer"
          onPointerDown={arrastrar}
          onDoubleClick={() => setAncho(PANEL.inicial)}
          onKeyDown={teclado}
          className="group absolute inset-y-4 -left-2 z-10 flex w-4 cursor-col-resize items-center justify-center focus:outline-none"
        >
          <span className="h-10 w-1 rounded-full bg-slate-400/30 transition-all duration-200 group-hover:h-24 group-hover:bg-accent group-focus-visible:h-24 group-focus-visible:bg-accent group-active:bg-accent" />
        </div>

        <button
          onClick={() => setAbierto(false)}
          title="Ocultar panel"
          aria-label="Ocultar panel"
          className="vidrio absolute -left-3.5 top-3 z-20 flex h-7 w-7 items-center justify-center rounded-full text-slate-300 transition-colors hover:text-white"
        >
          <Chevron hacia="der" />
        </button>

        <div className={`scroll-fino min-h-0 flex-1 p-4 ${tabla ? "overflow-hidden" : "overflow-y-auto"}`}>
          <div key={llave} className={`animate-entrar ${tabla ? "h-full" : ""}`}>
            {children}
          </div>
        </div>
      </aside>

      <button
        onClick={() => setAbierto(true)}
        aria-label="Mostrar panel"
        className={`vidrio absolute right-3 z-20 flex items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-medium text-slate-200 transition-[translate,opacity] duration-300 ease-out hover:text-white ${
          abierto ? "pointer-events-none translate-x-4 opacity-0" : ""
        }`}
        style={{ top }}
        tabIndex={abierto ? -1 : 0}
      >
        <Chevron hacia="izq" />
        Mostrar panel
      </button>
    </>
  );
}
