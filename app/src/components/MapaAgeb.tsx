import type {
  ExpressionSpecification,
  FillLayerSpecification,
  FilterSpecification,
  Map as MapLibreMap,
  StyleSpecification,
} from "maplibre-gl";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Map, {
  Layer,
  NavigationControl,
  ScaleControl,
  Source,
  type MapLayerMouseEvent,
} from "react-map-gl/maplibre";
import { useData } from "../data/context";
import type { MetricaZona, ZonaMetrics } from "../types/data";
import {
  camaraPara,
  conMargenExtra,
  crearRestriccion,
  restringir,
  suavizar,
  zoomParaAjustar,
  type Lienzo,
  type Margenes,
} from "../lib/camara";
import { BASEMAP_URL, CDMX_BBOX, LIMITE_MAPA, MASCARA_URL, ZOOM_MAX } from "../lib/config";
import { mapLib } from "../lib/maplibre";
import { metricaDe, percentiles, useResumen } from "../lib/derivados";
import {
  COLOR_ACIERTO,
  COLOR_FALLO,
  COLOR_NUEVO,
  COLOR_RURAL,
  COLOR_SIN_DATOS,
  MAPA_CATEGORIA,
  RAYA_RURAL,
  decimal,
  entero,
  nombrePropio,
  pctPequeno,
} from "../lib/format";
import { PASOS_PRUEBA, useApp, useMargenesMapa, type Modo, type Nivel } from "../lib/store";

// Respaldo sin internet: fondo liso. Así la demo no depende de que cargue el basemap.
const ESTILO_SIN_RED: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [{ id: "fondo", type: "background", paint: { "background-color": "#0a1020" } }],
};

// Zoom al hacer clic en una zona: siempre el mismo, para que la distancia no dependa de cuánto se había
// explorado antes. Desde listas y buscador se encuadra la zona completa, hasta ZOOM_MOSTRAR.
const ZOOM_CLIC = 13;
const ZOOM_MOSTRAR = 14.5;

const lienzoDe = (mapa: MapLibreMap, m: Margenes): Lienzo => {
  const c = mapa.getContainer();
  return { ancho: c.clientWidth, alto: c.clientHeight, m };
};
const arrastrandoPanel = () => document.documentElement.dataset.arrastrando !== undefined;

const CAT_NUM = { por_debajo: 0, dentro: 1, por_encima: 2 } as const;
// Una fuente por nivel, más las AGEB rurales (fuera del análisis). Cada una con su clave de unión.
type Fuente = Nivel | "rural";
const FUENTE_DE_CAPA: Record<string, Fuente> = {
  "ageb-relleno": "ageb",
  "ageb-sin-colonia": "ageb",
  "colonia-relleno": "colonia",
  "alcaldia-relleno": "alcaldia",
  "rural-relleno": "rural",
};
const CAPAS_INTERACTIVAS = Object.keys(FUENTE_DE_CAPA);
const RAYADO = "rayado-rural";
// La zona bajo el cursor: la capa de más arriba, salvo el fondo "fuera del catálogo de colonias", que solo cuenta
// donde no hay colonia.
const zonaDe = (e: MapLayerMouseEvent) => e.features?.find((f) => f.layer.id !== "ageb-sin-colonia") ?? e.features?.[0];

// Patrón diagonal de las AGEB rurales, dibujado en un canvas.
function imagenRayado(): ImageData {
  const n = 8;
  const c = document.createElement("canvas");
  c.width = c.height = n;
  const ctx = c.getContext("2d")!;
  ctx.strokeStyle = RAYA_RURAL;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (const d of [-n, 0, n]) {
    ctx.moveTo(d, n);
    ctx.lineTo(d + n, 0);
  }
  ctx.stroke();
  return ctx.getImageData(0, 0, n, n);
}

// Estado de las zonas agregadas (alcaldía, colonia): color y selección. Sin contornos de top: el top 10% y la
// comparación con el conteo son de la hipótesis C, que se evalúa por AGEB.
function estadosZona(ids: string[], m: ZonaMetrics, periodo: string, sel: string | null) {
  const n = Object.keys(m).length;
  const qe = percentiles(Object.keys(m), (id) => metricaDe(m[id], periodo).establecimientos);
  return ids.map((id) => {
    const z: MetricaZona | undefined = m[id] ? metricaDe(m[id], periodo) : undefined;
    return {
      id,
      estado: {
        sin: !z,
        q: z && z.observados > 0 ? 1 - (z.rank_conteo - 1) / n : 0,
        qe: qe.get(id) ?? 0,
        cat: z ? CAT_NUM[z.categoria] : 1,
        lrr: z ? Math.log2(Math.max(z.rr, 0.01)) : 0,
        sel: id === sel,
      },
    };
  });
}
const fs = (k: string): ExpressionSpecification => ["feature-state", k];
const fsNum = (k: string): ExpressionSpecification => ["to-number", ["coalesce", fs(k), 0]];
const fsBool = (k: string): ExpressionSpecification => ["boolean", ["coalesce", fs(k), false]];

function colorRelleno(modo: Modo, escala: "categoria" | "rr", validar: boolean): ExpressionSpecification {
  let color: ExpressionSpecification;
  if (validar) {
    // Prueba con el año anterior: solo se pintan las zonas de la lista elegida, según lo que pasó ese año.
    color = ["match", fsNum("val"), 2, COLOR_ACIERTO, 1, COLOR_FALLO, COLOR_SIN_DATOS];
  } else if (modo === "comercio") {
    // Percentil del número de establecimientos: violeta, para no confundirlo con el "calor" del robo.
    color = ["interpolate", ["linear"], fsNum("qe"), 0, "#1a2540", 0.5, "#2b2f5c", 0.75, "#4c3a8f", 0.9, "#8b5cf6", 1, "#c4b5fd"];
  } else if (modo === "conteo") {
    // Percentil del conteo crudo: el "calor" se concentra en el decil superior (el top que se contornea).
    color = ["interpolate", ["linear"], fsNum("q"), 0, "#1a2540", 0.5, "#2c3352", 0.75, "#6b3420", 0.9, "#ea580c", 1, "#ef4444"];
  } else if (escala === "categoria") {
    // Solo "por encima" va saturado: es la categoría que importa para decidir.
    color = ["match", fsNum("cat"), 0, MAPA_CATEGORIA.por_debajo, 2, MAPA_CATEGORIA.por_encima, MAPA_CATEGORIA.dentro];
  } else {
    // Divergente en log2(RR): azul = menos de lo esperado, rojo = más.
    color = ["interpolate", ["linear"], fsNum("lrr"), -1.5, "#1d4ed8", -0.5, "#3b82f6", 0, "#475569", 0.5, "#f97316", 1.5, "#dc2626"];
  }
  return ["case", fsBool("sin"), COLOR_SIN_DATOS, color];
}

export default function MapaAgeb() {
  const datos = useData();
  const {
    geo,
    metrics,
    agebPorId,
    bboxPorAgeb,
    coloniaPorId,
    alcaldiasGeo,
    alcaldiasMetrics,
    alcaldiaPorId,
    bboxPorAlcaldia,
    coloniasGeo,
    coloniasMetrics,
    coloniaGeoPorId,
    bboxPorColonia,
    rural,
    meta,
    validacion,
    validacionPorId,
  } = datos;
  const { modo, escala, periodo, resaltarCambios, pasoPrueba, nivel, agebSel, zonaSel, coloniaSel, vista, encuadre } = useApp();
  const seleccionarAgeb = useApp((s) => s.seleccionarAgeb);
  const seleccionarColonia = useApp((s) => s.seleccionarColonia);
  const seleccionarZona = useApp((s) => s.seleccionarZona);
  const resumen = useResumen();

  const [estilo, setEstilo] = useState<string | StyleSpecification | null>(null);
  // Instancia que ya emitió "load" (con StrictMode en desarrollo se crean dos).
  const [mapa, setMapa] = useState<MapLibreMap | null>(null);
  const listo = mapa !== null;
  const [antesDe, setAntesDe] = useState<string | undefined>();
  const [hover, setHover] = useState<{ fuente: Fuente; id: string; x: number; y: number } | null>(null);

  // Lo que tapan el encabezado y el panel. La restricción lee un ref para no recrearse en cada arrastre;
  // se actualiza al terminar la animación de ajuste (ver abajo).
  const margenes = useMargenesMapa();
  const margenesRestriccion = useRef(margenes);
  const margenesUltimos = useRef(margenes);
  // Mientras el mapa se reacomoda al nuevo panel, la restricción se suspende: con el límite viejo el
  // destino podría no ser alcanzable (p. ej. al ocultar el panel, recentrar la ciudad).
  const reacomodando = useRef(false);
  useEffect(() => {
    margenesUltimos.current = margenes;
  });

  // Vista inicial: la CDMX completa en el área que deja libre el panel.
  const [vistaInicial] = useState(() => {
    const { center, zoom } = camaraPara(CDMX_BBOX, { ancho: window.innerWidth, alto: window.innerHeight, m: margenes });
    return { longitude: center[0], latitude: center[1], zoom };
  });

  // Verifica que el basemap responda; si no, usa el estilo sin red.
  useEffect(() => {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 5000);
    fetch(BASEMAP_URL, { signal: ctrl.signal })
      .then((r) => (r.ok ? setEstilo(BASEMAP_URL) : setEstilo(ESTILO_SIN_RED)))
      .catch(() => setEstilo(ESTILO_SIN_RED))
      .finally(() => clearTimeout(t));
    return () => ctrl.abort();
  }, []);

  // El patrón de las AGEB rurales se registra antes de montar las capas (que esperan a `mapa`).
  const alCargar = useCallback((e: { target: MapLibreMap }) => {
    if (!e.target.hasImage(RAYADO)) e.target.addImage(RAYADO, imagenRayado());
    setMapa(e.target);
  }, []);

  // react-map-gl puede reemplazar el estilo después de "load" (estilo vacío → basemap). Cada vez que
  // cambia el estilo se ajustan las etiquetas y se recalcula dónde insertar los polígonos.
  useEffect(() => {
    if (!mapa) return;
    const alCambiarEstilo = () => {
      const simbolos = mapa.getStyle().layers.filter((l) => l.type === "symbol" && !l.id.startsWith("ageb-"));
      // Etiquetas con el nombre local (español) en lugar del nombre en inglés del basemap.
      for (const l of simbolos) {
        if (JSON.stringify(mapa.getLayoutProperty(l.id, "text-field") ?? "").includes("name_en"))
          mapa.setLayoutProperty(l.id, "text-field", ["coalesce", ["get", "name"], ["get", "name_en"]]);
      }
      // Un estilo nuevo llega sin las imágenes agregadas a mano.
      if (!mapa.hasImage(RAYADO)) mapa.addImage(RAYADO, imagenRayado());
      // Los polígonos van debajo de las etiquetas del basemap para no tapar nombres de calles.
      setAntesDe(simbolos[0]?.id);
    };
    alCambiarEstilo();
    mapa.on("styledata", alCambiarEstilo);
    return () => {
      mapa.off("styledata", alCambiarEstilo);
    };
  }, [mapa]);

  // Límite de la vista: no se puede alejar ni desplazar más allá de la CDMX.
  useEffect(() => {
    if (!mapa) return;
    let vigente = true;
    void mapLib.then(({ LngLat }) => {
      if (!vigente) return;
      const restriccion = crearRestriccion(LngLat, LIMITE_MAPA, () => lienzoDe(mapa, margenesRestriccion.current), ZOOM_MAX);
      mapa.setTransformConstrain((centro, zoom) => (reacomodando.current ? { center: centro, zoom } : restriccion(centro, zoom)));
      mapa.setMinZoom(zoomParaAjustar(LIMITE_MAPA, lienzoDe(mapa, margenesRestriccion.current)));
    });
    const alRedimensionar = () => mapa.setMinZoom(zoomParaAjustar(LIMITE_MAPA, lienzoDe(mapa, margenesRestriccion.current)));
    mapa.on("resize", alRedimensionar);
    return () => {
      vigente = false;
      mapa.off("resize", alRedimensionar);
    };
  }, [mapa]);

  // Al abrir, cerrar o ensanchar el panel cambia el área útil. Si la vista actual ya no cabe, se lleva
  // al límite nuevo con una animación corta (al arrastrar, sin animación: sigue al cursor).
  useEffect(() => {
    if (!mapa) return;
    const aplicar = () => {
      margenesRestriccion.current = margenes;
      mapa.setMinZoom(zoomParaAjustar(LIMITE_MAPA, lienzoDe(mapa, margenes)));
    };
    const c = mapa.getCenter();
    const z = mapa.getZoom();
    const destino = restringir(LIMITE_MAPA, lienzoDe(mapa, margenes), c.lng, c.lat, z, ZOOM_MAX);
    const cambia = Math.abs(destino.zoom - z) > 1e-6 || Math.abs(destino.center[0] - c.lng) > 1e-9 || Math.abs(destino.center[1] - c.lat) > 1e-9;
    if (!cambia || arrastrandoPanel()) {
      aplicar();
      if (cambia) mapa.jumpTo(destino);
      return;
    }
    const terminar = () => {
      reacomodando.current = false;
      aplicar();
    };
    reacomodando.current = true;
    mapa.easeTo({ ...destino, duration: 300, easing: suavizar });
    mapa.once("moveend", terminar);
    return () => {
      mapa.off("moveend", terminar);
      reacomodando.current = false;
    };
  }, [mapa, margenes]);


  // Estado de cada zona (color y resaltados), por fuente: se actualiza sin volver a subir la geometría.
  const consulta = vista === "consulta";
  const validar = vista === "validar";
  const estados = useMemo(() => {
    const n = Object.keys(metrics).length;
    const qe = percentiles(Object.keys(metrics), (id) => metricaDe(metrics[id], periodo).establecimientos);
    const top = { comercio: resumen.topComercio, conteo: resumen.topConteo, riesgo: resumen.topRiesgo }[modo];
    const ageb = geo.features.map((f) => {
      const id = f.properties.cvegeo;
      const m = metrics[id] ? metricaDe(metrics[id], periodo) : undefined;
      const enTop = !validar && top.has(id);
      const soloAqui = modo === "conteo" ? resumen.soloConteo.has(id) : resumen.soloRiesgo.has(id);
      const z = validar ? validacionPorId.get(id) : undefined;
      const enLista = z && (modo === "conteo" ? z.en_lista_conteo : z.en_lista_riesgo);
      // En la prueba no hay contorno del top: solo el cian de las zonas nuevas, cuando el recorrido llega a la periferia.
      const nuevaPrueba = validar && PASOS_PRUEBA[pasoPrueba].nuevas && modo === "riesgo" && z?.en_lista_riesgo && !z.en_lista_conteo;
      return {
        id,
        estado: {
          sin: !m,
          q: m && m.observados > 0 ? 1 - (m.rank_conteo - 1) / n : 0,
          qe: qe.get(id) ?? 0,
          cat: m ? CAT_NUM[m.categoria] : 1,
          lrr: m ? Math.log2(Math.max(m.rr, 0.01)) : 0,
          top: nuevaPrueba ? 2 : enTop ? (soloAqui ? 2 : 1) : 0,
          sale: !validar && modo === "riesgo" && resumen.soloConteo.has(id),
          val: enLista ? (z.mas_de_lo_esperado ? 2 : 1) : 0,
          sel: id === agebSel,
        },
      };
    });
    const colSel = consulta ? coloniaSel : zonaSel?.nivel === "colonia" ? zonaSel.id : null;
    const alcSel = zonaSel?.nivel === "alcaldia" ? zonaSel.id : null;
    return {
      ageb,
      colonia: estadosZona(coloniasGeo.features.map((f) => f.properties.cve_col), coloniasMetrics, periodo, colSel),
      alcaldia: estadosZona(alcaldiasGeo.features.map((f) => f.properties.cve_mun), alcaldiasMetrics, periodo, alcSel),
    };
  }, [geo, metrics, modo, periodo, resumen, agebSel, zonaSel, coloniaSel, consulta, validar, pasoPrueba, validacionPorId, coloniasGeo, coloniasMetrics, alcaldiasGeo, alcaldiasMetrics]);

  useEffect(() => {
    if (!mapa) return;
    const aplicar = (fuente: keyof typeof estados) => {
      if (!mapa.getSource(fuente)) return;
      for (const { id, estado } of estados[fuente]) mapa.setFeatureState({ source: fuente, id }, estado);
    };
    for (const f of Object.keys(estados) as (keyof typeof estados)[]) aplicar(f);
    // Si una fuente se crea o se vuelve a crear (cambio de estilo), el feature-state se pierde: reaplicar.
    const alCambiarFuente = (e: { sourceId?: string; sourceDataType?: string }) => {
      if (e.sourceId && e.sourceId in estados && e.sourceDataType === "metadata") aplicar(e.sourceId as keyof typeof estados);
    };
    mapa.on("sourcedata", alCambiarFuente);
    return () => {
      mapa.off("sourcedata", alCambiarFuente);
    };
  }, [mapa, estados]);

  useEffect(() => {
    const map = mapa;
    if (!map || !hover) return;
    const objetivo = { source: hover.fuente, id: hover.id };
    map.setFeatureState(objetivo, { hover: true });
    return () => {
      if (map.getSource(objetivo.source)) map.setFeatureState(objetivo, { hover: false });
    };
  }, [mapa, hover?.fuente, hover?.id]);

  // Encuadre de una zona. "acercar" (clic en el mapa): siempre a ZOOM_CLIC, sin importar el zoom previo.
  // "mostrar" (listas, buscador): vuelo hasta ver la zona completa.
  useEffect(() => {
    if (!encuadre || !mapa) return;
    const l = lienzoDe(mapa, margenesUltimos.current);
    if (encuadre.tipo === "acercar") {
      mapa.easeTo({ ...camaraPara(encuadre.bbox, l, ZOOM_CLIC), duration: 900, easing: suavizar });
    } else {
      const ajuste = zoomParaAjustar(encuadre.bbox, conMargenExtra(l, 60));
      mapa.flyTo({ ...camaraPara(encuadre.bbox, l, Math.min(ajuste, ZOOM_MOSTRAR)), speed: 1.3, curve: 1.4, maxDuration: 2200 });
    }
  }, [encuadre, mapa]);

  // Al cambiar de nivel la zona bajo el cursor es otra: se descarta el tooltip hasta que el ratón se mueva.
  useEffect(() => setHover(null), [nivel]);

  const alMover = useCallback((e: MapLayerMouseEvent) => {
    const f = zonaDe(e);
    const fuente = f ? FUENTE_DE_CAPA[f.layer.id] : undefined;
    setHover(f && fuente && f.id !== undefined ? { fuente, id: String(f.id), x: e.point.x, y: e.point.y } : null);
  }, []);

  const alClic = useCallback(
    (e: MapLayerMouseEvent) => {
      const f = zonaDe(e);
      const fuente = f ? FUENTE_DE_CAPA[f.layer.id] : undefined;
      if (!f || f.id === undefined || !fuente || fuente === "rural") return;
      const id = String(f.id);
      if (fuente === "alcaldia") {
        // Una alcaldía no cabe a ZOOM_CLIC: se encuadra completa.
        seleccionarZona({ nivel: "alcaldia", id }, bboxPorAlcaldia.get(id), "mostrar");
      } else if (fuente === "colonia") {
        if (consulta) {
          const col = coloniaPorId.get(id);
          if (col) seleccionarColonia(col.cve_col, col.bbox, "acercar");
        } else {
          seleccionarZona({ nivel: "colonia", id }, bboxPorColonia.get(id), "acercar");
        }
      } else if (consulta) {
        const col = coloniaPorId.get(agebPorId.get(id)?.cve_col ?? "");
        if (col) seleccionarColonia(col.cve_col, col.bbox, "acercar");
      } else {
        seleccionarAgeb(id, bboxPorAgeb.get(id), "acercar");
      }
    },
    [consulta, agebPorId, bboxPorAgeb, bboxPorAlcaldia, bboxPorColonia, coloniaPorId, seleccionarAgeb, seleccionarColonia, seleccionarZona],
  );

  const relleno = useMemo(() => colorRelleno(modo, escala, validar), [modo, escala, validar]);
  // En "Consulta mi zona" solo se pintan las categorías (lenguaje llano): sin contornos de top ni de cambios.
  const destacar = resaltarCambios && modo === "riesgo" && !consulta && !validar && nivel === "ageb";
  // Contorno blanco del top 10%: no en "¿Qué es anormal?"; ahí solo queda el cian.
  const contornoTop = !consulta && vista !== "anormal";
  // Las propiedades que dependen de feature-state siempre son expresiones: si alternan entre constante y expresión,
  // MapLibre falla ("this.expression.evaluate is not a function") cuando llega un setFeatureState durante el cambio.
  const esNuevo: ExpressionSpecification = ["all", destacar || validar, ["==", fsNum("top"), 2]];
  const todas: FilterSpecification = ["has", "cvegeo"];
  const visible = (n: Nivel) => ({ visibility: nivel === n ? ("visible" as const) : ("none" as const) });
  const pintura: FillLayerSpecification["paint"] = {
    "fill-color": relleno,
    "fill-opacity": ["case", fsBool("hover"), 0.95, fsBool("sin"), 0.45, 0.8],
  };
  // Van sin beforeId: quedan encima de las etiquetas del basemap, como el resto de los resaltados. Arreglo y no
  // Fragment: <Source> solo le pasa `source` a sus hijos directos.
  const seleccion = (fuente: Nivel) => [
    // Halo difuso bajo el contorno de la zona seleccionada
    <Layer
      key="halo"
      id={`${fuente}-halo`}
      type="line"
      layout={visible(fuente)}
      paint={{ "line-color": "#facc15", "line-width": 10, "line-blur": 8, "line-opacity": ["case", fsBool("sel"), 0.55, 0] }}
    />,
    <Layer
      key="seleccion"
      id={`${fuente}-seleccion`}
      type="line"
      layout={visible(fuente)}
      paint={{
        "line-color": "#facc15",
        "line-width": ["case", fsBool("sel"), 3.5, fsBool("hover"), 2, 0],
        "line-opacity": ["case", fsBool("sel"), 1, fsBool("hover"), 0.8, 0],
      }}
    />,
  ];

  if (!estilo) return <div className="h-full w-full bg-ink-900" />;

  return (
    <div className="relative h-full w-full">
      <Map
        mapLib={mapLib}
        initialViewState={vistaInicial}
        maxZoom={ZOOM_MAX}
        renderWorldCopies={false}
        mapStyle={estilo}
        onLoad={alCargar}
        interactiveLayerIds={CAPAS_INTERACTIVAS}
        onMouseMove={alMover}
        onMouseLeave={() => setHover(null)}
        onClick={alClic}
        cursor={hover && hover.fuente !== "rural" ? "pointer" : "grab"}
        dragRotate={false}
        touchPitch={false}
        attributionControl={{ compact: true }}
      >
        <NavigationControl position="top-right" showCompass={false} />
        <ScaleControl position="bottom-right" unit="metric" />
        {/* Rellenos por nivel (solo se ve el activo); todos debajo de las etiquetas del basemap. */}
        {listo && (
          <Source id="alcaldia" type="geojson" data={alcaldiasGeo} promoteId="cve_mun">
            <Layer id="alcaldia-relleno" type="fill" beforeId={antesDe} layout={visible("alcaldia")} paint={pintura} />
            {seleccion("alcaldia")}
          </Source>
        )}
        {listo && (
          <Source id="colonia" type="geojson" data={coloniasGeo} promoteId="cve_col">
            <Layer id="colonia-relleno" type="fill" beforeId={antesDe} layout={visible("colonia")} paint={pintura} />
            <Layer
              id="colonia-borde"
              type="line"
              beforeId={antesDe}
              layout={visible("colonia")}
              paint={{ "line-color": "#0a1020", "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.3, 14, 1], "line-opacity": 0.8 }}
            />
            {seleccion("colonia")}
          </Source>
        )}
        {listo && (
          <Source id="ageb" type="geojson" data={geo} promoteId="cvegeo">
            <Layer id="ageb-relleno" type="fill" beforeId={antesDe} filter={todas} layout={visible("ageb")} paint={pintura} />
            {/* En el nivel colonia, debajo de las colonias: lo que se ve son los huecos del catálogo de colonias. */}
            <Layer
              id="ageb-sin-colonia"
              type="fill"
              beforeId="colonia-relleno"
              layout={visible("colonia")}
              paint={{ "fill-color": COLOR_SIN_DATOS, "fill-opacity": ["case", fsBool("hover"), 0.7, 0.45] }}
            />
            <Layer
              id="ageb-borde"
              type="line"
              beforeId={antesDe}
              layout={visible("ageb")}
              paint={{ "line-color": "#0a1020", "line-width": ["interpolate", ["linear"], ["zoom"], 10, 0.2, 14, 0.8], "line-opacity": 0.7 }}
            />
            {/* Zonas que eran top por conteo y salen al ajustar por actividad económica */}
            <Layer
              id="ageb-sale"
              type="line"
              beforeId={antesDe}
              layout={visible("ageb")}
              paint={{
                "line-color": "#e2e8f0",
                "line-width": 1.2,
                "line-dasharray": [1.5, 1.5],
                "line-opacity": ["case", ["all", destacar, fsBool("sale")], 0.75, 0],
              }}
            />
            {/* Top 10% del modo activo (fuera de "¿Qué es anormal?"); en riesgo, cian = invisibles por conteo */}
            <Layer
              id="ageb-top"
              type="line"
              beforeId={antesDe}
              layout={visible("ageb")}
              paint={{
                "line-color": ["case", esNuevo, COLOR_NUEVO, "#f8fafc"],
                "line-width": ["case", esNuevo, 2, 1.2],
                "line-opacity": ["case", esNuevo, 0.95, ["all", contornoTop, [">", fsNum("top"), 0]], 0.95, 0],
              }}
            />
            {seleccion("ageb")}
          </Source>
        )}
        {/* AGEB rurales: fuera del análisis en todos los niveles. Tapan el relleno de la alcaldía o colonia que las contiene. */}
        {listo && (
          <Source id="rural" type="geojson" data={rural} promoteId="cvegeo">
            <Layer id="rural-relleno" type="fill" beforeId={antesDe} paint={{ "fill-color": COLOR_RURAL, "fill-opacity": 0.92 }} />
            <Layer id="rural-rayado" type="fill" beforeId={antesDe} paint={{ "fill-pattern": RAYADO, "fill-opacity": 0.45 }} />
          </Source>
        )}
        {/* Límites de alcaldía: referencia en todos los niveles, más marcados en el nivel alcaldía. */}
        {listo && (
          <Source id="alcaldia-limite" type="geojson" data={alcaldiasGeo}>
            <Layer
              id="alcaldia-limite"
              type="line"
              beforeId={antesDe}
              paint={{
                "line-color": "#cbd5e1",
                "line-width": nivel === "alcaldia" ? 1.6 : 0.9,
                "line-opacity": nivel === "alcaldia" ? 0.7 : 0.35,
              }}
            />
          </Source>
        )}
        {/* Fuera de la CDMX: velo oscuro y contorno tenue, para que el ojo se quede en la ciudad. */}
        {listo && (
          <Source id="cdmx-fuera" type="geojson" data={MASCARA_URL}>
            <Layer id="cdmx-fuera-velo" type="fill" paint={{ "fill-color": "#03060d", "fill-opacity": 0.62 }} />
            <Layer id="cdmx-borde-halo" type="line" paint={{ "line-color": "#38bdf8", "line-width": 7, "line-blur": 7, "line-opacity": 0.16 }} />
            <Layer id="cdmx-borde" type="line" paint={{ "line-color": "#7dd3fc", "line-width": 1, "line-opacity": 0.45 }} />
          </Source>
        )}
      </Map>

      {hover && (
        <div
          className="vidrio pointer-events-none absolute z-10 max-w-64 rounded-lg px-3 py-2 text-xs"
          style={{ left: hover.x + 14, top: hover.y + 14 }}
        >
          {tooltip(hover.fuente, hover.id)}
        </div>
      )}
    </div>
  );

  function tooltip(fuente: Fuente, id: string) {
    // En la lista del año siguiente: robos por año recientes y los que le corresponden (con un decimal si son pocos); en
    // un año observado, lo que pasó.
    const cifra = (x: number) => (x < 10 ? decimal(x) : entero(x));
    const cifras = (m: { observados: number; esperados: number; establecimientos: number } | undefined) =>
      consulta ? null : m ? (
        <div className="mt-1 text-slate-300">
          {modo === "comercio"
            ? `${entero(m.establecimientos)} establecimientos (DENUE 2023)`
            : periodo === "anticipado"
            ? `${fuente === "ageb" ? "" : "≈ "}${cifra(m.observados)} robos por año · le corresponden ${cifra(m.esperados)}`
            : `${fuente === "colonia" ? `≈ ${entero(m.observados)}` : entero(m.observados)} robos · ${cifra(m.esperados)} esperados`}
        </div>
      ) : (
        <div className="mt-1 text-slate-400">Sin actividad económica registrada</div>
      );
    if (fuente === "rural") {
      const r = rural.features.find((f) => f.properties.cvegeo === id)?.properties;
      return (
        <>
          <div className="font-semibold text-slate-100">Zona rural · fuera del análisis</div>
          <div className="text-slate-400">
            {r?.alcaldia} · {r ? decimal(r.area_km2) : ""} km²
          </div>
          <div className="mt-1 text-slate-400">
            Suelo de conservación y cultivo: aquí ocurre el {pctPequeno(meta.cobertura.pct_robos_rurales)} del robo a negocio.
          </div>
        </>
      );
    }
    if (fuente === "alcaldia") {
      const a = alcaldiaPorId.get(id);
      return (
        <>
          <div className="font-semibold text-slate-100">{a?.alcaldia}</div>
          <div className="text-slate-400">Alcaldía · {a?.agebs_urbanas} AGEB urbanas</div>
          {cifras(alcaldiasMetrics[id] ? metricaDe(alcaldiasMetrics[id], periodo) : undefined)}
        </>
      );
    }
    if (fuente === "colonia") {
      const c = coloniaGeoPorId.get(id);
      return (
        <>
          <div className="font-semibold text-slate-100">{c ? nombrePropio(c.nombre) : id}</div>
          <div className="text-slate-400">{c?.alcaldia}</div>
          {cifras(coloniasMetrics[id] ? metricaDe(coloniasMetrics[id], periodo) : undefined)}
        </>
      );
    }
    const p = agebPorId.get(id);
    if (nivel === "colonia")
      return (
        <>
          <div className="font-semibold text-slate-100">Fuera del catálogo de colonias</div>
          <div className="text-slate-400">
            {p?.alcaldia} · AGEB {id}
          </div>
          <div className="mt-1 text-slate-400">
            Ninguna colonia del catálogo cubre esta parte. Sus robos cuentan en los niveles AGEB y alcaldía.
          </div>
        </>
      );
    const z = validar ? validacionPorId.get(id) : undefined;
    return (
      <>
        <div className="font-semibold text-slate-100">{p ? nombrePropio(p.colonia) : id}</div>
        <div className="text-slate-400">{p?.alcaldia}</div>
        {validar ? (
          z && (modo === "conteo" ? z.en_lista_conteo : z.en_lista_riesgo) ? (
            <div className="mt-1 text-slate-300">
              #{modo === "conteo" ? z.rank_conteo : z.rank_riesgo} de la lista · en {validacion.anio}: {entero(z.observados)} robos,{" "}
              {decimal(z.esperados)} esperados
            </div>
          ) : (
            <div className="mt-1 text-slate-400">Fuera de la lista por {modo === "conteo" ? "conteo" : "riesgo"}</div>
          )
        ) : (
          cifras(metrics[id] ? metricaDe(metrics[id], periodo) : undefined)
        )}
      </>
    );
  }
}
