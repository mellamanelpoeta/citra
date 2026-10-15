import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";

// Recorrido por scroll del panel: la sección a la vista decide qué pinta el mapa. Lo usan "¿Qué es anormal?" y
// "¿Habría funcionado?".

// `holgado`: más aire entre pasos y entre bloques (lo usa "¿Habría funcionado?", que va en cifras y tarjetas).
export function Paso({
  n,
  titulo,
  refPaso,
  holgado = false,
  children,
}: {
  n: number;
  titulo: string;
  refPaso: (el: HTMLElement | null) => void;
  holgado?: boolean;
  children: ReactNode;
}) {
  return (
    <section ref={refPaso} className={`scroll-mt-14 ${holgado ? "pt-9" : "pt-5"}`}>
      <h3 className="flex items-center gap-2 text-base font-semibold text-slate-50">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-accent/10 text-xs font-bold text-accent ring-1 ring-accent/30">{n}</span>
        {titulo}
      </h3>
      <div className={`text-sm text-slate-300 ${holgado ? "mt-3 space-y-4" : "mt-2 space-y-2"}`}>{children}</div>
    </section>
  );
}

// `inicial`: paso en que se abre el panel (al volver del detalle de una zona se monta de nuevo). `alCambiar` se llama
// cada vez que cambia el paso activo.
export function useRecorrido(inicial: number, alCambiar: (i: number) => void) {
  const pasos = useRef<(HTMLElement | null)[]>([]);
  const barra = useRef<HTMLElement>(null);
  const [activo, setActivo] = useState(inicial);
  const actual = useRef(inicial);
  const aviso = useRef(alCambiar);
  useEffect(() => {
    aviso.current = alCambiar;
  });

  const activar = (i: number) => {
    if (i === actual.current) return;
    actual.current = i;
    setActivo(i);
    aviso.current(i);
  };

  useEffect(() => {
    if (inicial > 0) pasos.current[inicial]?.scrollIntoView({ block: "start" });
    // Solo al montar.
  }, []);

  // El paso activo es el último cuya cabecera ya llegó a la barra de pasos (con un margen). Un corte más bajo (p. ej. un
  // tercio del panel) activaría el paso siguiente al saltar a uno corto con la barra.
  useEffect(() => {
    // El panel desplazable es el contenedor de PanelLateral.
    const raiz = pasos.current[0]?.closest<HTMLElement>(".scroll-fino");
    if (!raiz) return;
    const alDesplazar = () => {
      const corte = (barra.current?.getBoundingClientRect().bottom ?? raiz.getBoundingClientRect().top) + 48;
      let i = 0;
      pasos.current.forEach((el, k) => {
        if (el && el.getBoundingClientRect().top <= corte) i = k;
      });
      activar(i);
    };
    raiz.addEventListener("scroll", alDesplazar, { passive: true });
    return () => raiz.removeEventListener("scroll", alDesplazar);
  }, []);

  const ir = (i: number) => {
    activar(i);
    pasos.current[i]?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const refPaso = (i: number) => (el: HTMLElement | null) => {
    pasos.current[i] = el;
  };
  return { activo, ir, refPaso, barra };
}

// Atajo a cada paso (útil en la demo); se queda arriba mientras se baja.
export function BarraPasos({ barra, etiquetas, activo, ir }: { barra: RefObject<HTMLElement | null>; etiquetas: string[]; activo: number; ir: (i: number) => void }) {
  return (
    <nav ref={barra} className="sticky -top-4 z-10 -mx-4 mt-3 flex gap-1 bg-ink-900/95 px-4 py-2" aria-label="Pasos del recorrido">
      {etiquetas.map((t, i) => (
        <button
          key={t}
          onClick={() => ir(i)}
          aria-current={activo === i ? "step" : undefined}
          className={`flex-1 whitespace-nowrap rounded-lg px-1.5 py-1.5 text-xs transition-colors ${
            activo === i ? "bg-accent font-semibold text-ink-950" : "velo text-slate-300 hover:bg-white/10"
          }`}
        >
          {i + 1}. {t}
        </button>
      ))}
    </nav>
  );
}
