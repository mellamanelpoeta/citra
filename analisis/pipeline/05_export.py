"""05 · Export de los 12 archivos del contrato (docs/6.Contrato-Datos.md v0.7) a app/public/data/.

Entradas: data/processed/modelos/ (salidas de 04_modelos.py y de 04g_anticipacion.py), panel, intermedios de AGEB,
catálogo de colonias, Marco Geoestadístico (AGEB rurales) y, si existe, la denuncia relativa por alcaldía de 04d.
Cada archivo se valida contra el esquema (pydantic, espejo de app/src/types/data.ts) antes de escribirse.

Uso: .venv/bin/python analisis/pipeline/05_export.py
"""
import json
import zipfile
from datetime import datetime, timezone
from typing import Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pydantic import BaseModel, Field, RootModel, field_validator
from sklearn.neighbors import NearestNeighbors

from common import CANDIDATOS_EFECTIVO, CORTE_FIJO, EFECTIVO, INTERIM, RAW, ROOT, SECTORES, asigna_ageb, lee_mg, lee_panel
from modelado import MODELOS, contraccion

OUT = ROOT / "app/public/data"
OUT.mkdir(parents=True, exist_ok=True)
UMBRALES = {"por_encima": 0.9, "por_debajo": 0.1}
TOP_Q = 0.10
AVISO_Q = 0.25  # aviso "señal dominada por robo sin violencia": puesto con solo violencia fuera de este top
SIMPLIFICA_M = 12  # tolerancia de simplificación (m); coverage_simplify conserva las fronteras compartidas


# ============================================================ esquema (espejo del contrato)
Cat = Literal["por_debajo", "dentro", "por_encima"]
Prob = Field(ge=0, le=1)
CVEGEO = r"^\d{9}[\dA-Z]{4}$"
CVEGEO_RURAL = r"^\d{5}[\dA-Z]{4}$"


class Veredicto(BaseModel):
    titulo: str
    enunciado: str
    estadistico: str
    resultado: Literal["confirmada", "rechazada", "no_concluyente"]
    lectura: str
    evidencia: list[str]


class Fuente(BaseModel):
    nombre: str
    corte: str
    url: str


class Cobertura(BaseModel):
    agebs_urbanas: int = Field(ge=1)
    agebs_rurales: int = Field(ge=0)
    pct_area_urbana: float = Prob
    pct_robos_rurales: float = Prob
    pct_establecimientos_rurales: float = Prob
    pct_area_urbana_sin_colonia: float = Prob
    pct_robos_sin_colonia: float = Prob


class Anticipacion(BaseModel):
    anio: int
    datos: str
    vida_media_anios: float = Field(gt=0)


class Meta(BaseModel):
    generado_en: str
    version_contrato: Literal["0.8"]
    anios: list[int] = Field(min_length=1)
    anticipado: Anticipacion
    variable_objetivo: str
    modelo_esperado: Literal["M1", "M2"]
    umbrales_categoria: dict[Literal["por_encima", "por_debajo"], float]
    hipotesis: dict[Literal["A", "B", "C"], Veredicto]
    fuentes: list[Fuente]
    cobertura: Cobertura


class AgebProps(BaseModel):
    cvegeo: str = Field(pattern=CVEGEO)
    alcaldia: str
    colonia: str
    cve_col: str = Field(min_length=1)
    poblacion_2020: int = Field(ge=0)


class Geometria(BaseModel):
    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list


class AlcaldiaProps(BaseModel):
    cve_mun: str = Field(pattern=r"^\d{3}$")
    alcaldia: str
    agebs_urbanas: int = Field(ge=0)
    agebs_rurales: int = Field(ge=0)
    pct_area_urbana: float = Prob
    prioritarias: int = Field(ge=0)
    denuncia_relativa: Optional[float] = Field(default=None, ge=0)
    denuncia_relativa_ic: Optional[tuple[float, float]] = None


class ColoniaProps(BaseModel):
    cve_col: str = Field(min_length=1)
    nombre: str
    alcaldia: str


class RuralProps(BaseModel):
    cvegeo: str = Field(pattern=CVEGEO_RURAL)
    alcaldia: str
    area_km2: float = Field(gt=0)


def coleccion(props):
    """Modelos pydantic de una FeatureCollection con propiedades `props`."""
    feat = type(f"Feature{props.__name__}", (BaseModel,), {"__annotations__": {
        "type": Literal["Feature"], "geometry": Geometria, "properties": props}})
    return type(f"GeoJSON{props.__name__}", (BaseModel,), {"__annotations__": {
        "type": Literal["FeatureCollection"], "features": list[feat]}})


AgebGeoJSON = coleccion(AgebProps)
AlcaldiasGeoJSON = coleccion(AlcaldiaProps)
ColoniasGeoJSON = coleccion(ColoniaProps)
RuralGeoJSON = coleccion(RuralProps)


class MetricaAgeb(BaseModel):
    observados: int = Field(ge=0)
    esperados: float = Field(ge=0)
    rr: float = Field(ge=0)
    p_exceso: float = Prob
    categoria: Cat
    establecimientos: int = Field(ge=0)
    pct_efectivo: float = Prob
    delito_dominante: Optional[str] = None
    rank_conteo: int = Field(ge=1)
    rank_riesgo: int = Field(ge=1)


class MetricaAgebAnticipada(BaseModel):
    observados: float = Field(ge=0)
    esperados: float = Field(ge=0)
    rr: float = Field(ge=0)
    p_exceso: float = Prob
    categoria: Cat
    establecimientos: int = Field(ge=0)
    pct_efectivo: float = Prob
    delito_dominante: Optional[str] = None
    rank_conteo: int = Field(ge=1)
    rank_riesgo: int = Field(ge=1)


class AgebEntrada(BaseModel):
    anticipado: MetricaAgebAnticipada
    por_anio: dict[str, MetricaAgeb]

    @field_validator("por_anio")
    @classmethod
    def _anios(cls, v):
        assert all(k.isdigit() and len(k) == 4 for k in v), "las llaves de por_anio deben ser años"
        return v


class AgebMetrics(RootModel[dict[str, AgebEntrada]]):
    pass


class MetricaZona(BaseModel):
    observados: float = Field(ge=0)
    esperados: float = Field(ge=0)
    rr: float = Field(ge=0)
    p_exceso: float = Prob
    categoria: Cat
    establecimientos: int = Field(ge=0)
    rank_conteo: int = Field(ge=1)
    rank_riesgo: int = Field(ge=1)


class ZonaEntrada(BaseModel):
    anticipado: MetricaZona
    por_anio: dict[str, MetricaZona]


class ZonaMetrics(RootModel[dict[str, ZonaEntrada]]):
    pass


class Prioridad(BaseModel):
    cvegeo: str = Field(pattern=CVEGEO)
    colonia: str
    cve_col: str = Field(min_length=1)
    alcaldia: str
    rank_riesgo: int = Field(ge=1)
    rank_conteo: int = Field(ge=1)
    p_exceso: float = Prob
    rr: float = Field(ge=0)
    esperados: float = Field(ge=0)
    persistencia: int = Field(ge=0)
    visible_por_conteo: bool
    rank_riesgo_violencia: int = Field(ge=1)
    senal_sin_violencia: bool


class ColoniaIndex(BaseModel):
    cve_col: str = Field(min_length=1)
    nombre: str
    alcaldia: str
    agebs: list[str]
    centroide: tuple[float, float]
    bbox: tuple[float, float, float, float]
    categoria: Cat
    rr: float = Field(ge=0)
    similares: list[str]


class ResultadoLista(BaseModel):
    n: int = Field(ge=0)
    pct_mas_de_lo_esperado: float = Prob
    pct_por_encima: float = Prob
    robos: int = Field(ge=0)
    esperados: float = Field(ge=0)
    pct_exceso_ciudad: float = Prob


class PuntoSerie(BaseModel):
    anio: int
    observados: int = Field(ge=0)
    esperados: float = Field(ge=0)


class ZonaValidacion(BaseModel):
    cvegeo: str = Field(pattern=CVEGEO)
    colonia: str
    cve_col: str = Field(min_length=1)
    alcaldia: str
    rank_riesgo: int = Field(ge=1)
    rank_conteo: int = Field(ge=1)
    en_lista_riesgo: bool
    en_lista_conteo: bool
    rr: float = Field(ge=0)
    p_exceso: float = Prob
    categoria: Cat
    serie: list[PuntoSerie] = Field(min_length=1)
    esperados: float = Field(ge=0)
    observados: int = Field(ge=0)
    mas_de_lo_esperado: bool


class Vulnerabilidad(BaseModel):
    n: int = Field(ge=1)
    poblacion: int = Field(ge=0)
    escolaridad: float = Field(ge=0)
    pct_bajo_mediana_escolaridad: float = Prob
    pct_sin_basica: float = Prob
    pct_sin_salud: float = Prob
    pct_sin_internet: float = Prob


GrupoVal = Literal["lista_riesgo", "lista_conteo", "no_visibles", "ciudad"]


class Validacion(BaseModel):
    anio: int
    datos: str
    resumen: dict[GrupoVal, ResultadoLista]
    vulnerabilidad: dict[GrupoVal, Vulnerabilidad]
    zonas: list[ZonaValidacion]


class GrupoGiro(BaseModel):
    id: str
    nombre: str
    scian: list[str]
    irr: Optional[float]
    mensaje: str


def escribe(nombre, obj, modelo, lista=False):
    """Valida con pydantic (cada elemento si `lista`) y escribe JSON compacto. Devuelve el tamaño en KB."""
    for x in (obj if lista else [obj]):
        modelo.model_validate(x)
    ruta = OUT / nombre
    ruta.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    kb = ruta.stat().st_size / 1024
    print(f"{nombre:22s} {kb:8.0f} KB")
    return kb


def r3(x):
    return float(round(float(x), 3))


def puesto(o):
    """Puesto único por conteo (1 = más robos), con el mismo orden que `modelado.top` usa en la hipótesis C. Con rangos
    compartidos, el top 10% por conteo de la app tendría más de 10% de AGEB (251 en el periodo) y no cuadraría con
    `visible_por_conteo`; los empates se rompen en el orden del ranking."""
    return pd.Series(np.arange(1, len(o) + 1), index=o.sort_values(ascending=False).index).reindex(o.index)


# El catálogo de colonias trae 'ó' mal codificada como 'ë' en 2 nombres ("Secciën", "Ampliaciën"); no hay 'ë' legítimas.
def limpia_colonia(s):
    return s.replace("ë", "ó")


def categoria(p_exc):
    return np.where(p_exc >= UMBRALES["por_encima"], "por_encima", np.where(p_exc <= UMBRALES["por_debajo"], "por_debajo", "dentro"))


# ============================================================ insumos
resumen = json.load(open(MODELOS / "resumen.json"))
anual = pd.read_parquet(MODELOS / "riesgo_anual.parquet")
# Lista anticipada para el año siguiente y prueba un año atrás (04g, contrato v0.5): O y Ê de cada año pesan
# w = 0.5^((último año − año) / vida media). W = Σw es igual para todas las AGEB (todas tienen los mismos años).
ant = pd.read_parquet(MODELOS / "anticipado.parquet").set_index("cvegeo")
val = pd.read_parquet(MODELOS / "validacion_2024.parquet").set_index("cvegeo")
W = float(ant.w.iloc[0])
assert np.allclose(ant.w, W), "pesos distintos entre AGEB"
VIDA_MEDIA = 1.5  # = 04g_anticipacion.VIDA_MEDIA
coef = pd.read_csv(MODELOS / "coef_m1.csv")
ANIOS = resumen["anios"]
at = pd.read_parquet(INTERIM / "ageb_atributos.parquet").set_index("cvegeo")
at["colonia"] = at.colonia.map(limpia_colonia)
pan = lee_panel()
fijo = pan[pan.anio == CORTE_FIJO].set_index("cvegeo")
E = fijo.E_fijo
pct_ef = (fijo.ps_efectivo_barrio_fijo + fijo.ps_efectivo_formal_fijo).fillna(0)

# Delito dominante dentro de la variable principal (V2b): con o sin violencia
d = pd.read_parquet(INTERIM / "delitos_ageb.parquet", columns=["anio_hecho", "grupo", "objetivo", "farderos", "cvegeo", "tipo_asig"])
d = d[(d.tipo_asig == "URBANA") & (d.objetivo == "negocio") & (d.farderos == 0) & (d.grupo != "negocio_sv_dentro")
      & d.anio_hecho.isin(ANIOS)]
d["tipo"] = np.where(d.grupo.str.startswith("negocio_cv"), "Robo a negocio con violencia", "Robo a negocio sin violencia")
dom_anio = d.groupby(["cvegeo", "anio_hecho"]).tipo.agg(lambda s: s.value_counts().idxmax())
dom_per = d.groupby("cvegeo").tipo.agg(lambda s: s.value_counts().idxmax())

# ============================================================ ageb_metrics.json
ant["categoria"] = categoria(ant.p_exc)
ant["rank_conteo"] = puesto(ant.O_w)
ant["rank_riesgo"] = ant.score.rank(ascending=False, method="first").astype(int)
anual["categoria"] = categoria(anual.p_exc)
anual["score"] = anual.p_exc + 1e-6 * anual.rr
anual["rank_conteo"] = anual.groupby("anio").O.transform(puesto)
anual["rank_riesgo"] = anual.groupby("anio").score.rank(ascending=False, method="first").astype(int)


def metrica(cv, r, dom):
    m = {"observados": int(r.O), "esperados": r3(r.Ehat), "rr": r3(r.rr), "p_exceso": r3(r.p_exc),
         "categoria": r.categoria, "establecimientos": int(E[cv]), "pct_efectivo": r3(pct_ef[cv]),
         "rank_conteo": int(r.rank_conteo), "rank_riesgo": int(r.rank_riesgo)}
    if r.O > 0 and dom is not None:
        m["delito_dominante"] = dom
    return m


def metrica_ant(cv, r, dom):
    """Lista del año siguiente: `observados` es el ritmo reciente (robos por año, ponderado) y `esperados` lo que le
    corresponde en un año por su actividad. Desde v0.6 no se exporta un "anticipado" aparte: era rr × esperados."""
    m = {"observados": r3(r.O_w / W), "esperados": r3(r.E_next), "rr": r3(r.rr),
         "p_exceso": r3(r.p_exc), "categoria": r.categoria, "establecimientos": int(E[cv]), "pct_efectivo": r3(pct_ef[cv]),
         "rank_conteo": int(r.rank_conteo), "rank_riesgo": int(r.rank_riesgo)}
    if r.O_w > 0 and dom is not None:
        m["delito_dominante"] = dom
    return m


metrics = {}
for cv, r in ant.iterrows():
    metrics[cv] = {"anticipado": metrica_ant(cv, r, dom_per.get(cv)), "por_anio": {}}
for r in anual.itertuples(index=False):
    metrics[r.cvegeo]["por_anio"][str(r.anio)] = metrica(r.cvegeo, r, dom_anio.get((r.cvegeo, r.anio)))
assert len(metrics) == (E > 0).sum(), "AGEB con E > 0 sin métricas"

# ============================================================ prioridades.json
# Lista del año siguiente (v0.5). "Visible por conteo" compara con el top por conteo con la misma ponderación por año:
# las dos listas usan la misma información y solo difieren en el ajuste por actividad económica.
n_top = int(round(len(ant) * TOP_Q))
pers = anual[anual.categoria == "por_encima"].groupby("cvegeo").size()
top_c = set(ant.index[ant.rank_conteo <= n_top])
# Aviso (2026-10-11): ranking residual con solo robo con violencia (04g, misma ponderación).
# Supuesto: si la zona sale del top 25% sin el robo sin violencia, su prioridad puede reflejar denuncia o un
# establecimiento atípico. No la saca de la lista: pide revisarla caso por caso.
cv_rank = (pd.read_parquet(MODELOS / "anticipado_cv.parquet").set_index("cvegeo").score
           .rank(ascending=False, method="first").astype(int))
assert set(cv_rank.index) == set(ant.index), "ranking con violencia desalineado"
n_aviso = int(round(len(ant) * AVISO_Q))
prior = []
for cv, r in ant.sort_values("rank_riesgo").head(n_top).iterrows():
    prior.append({"cvegeo": cv, "colonia": at.colonia[cv], "cve_col": at.cve_col[cv], "alcaldia": at.alcaldia[cv],
                  "rank_riesgo": int(r.rank_riesgo), "rank_conteo": int(r.rank_conteo), "p_exceso": r3(r.p_exc),
                  "rr": r3(r.rr), "esperados": r3(r.E_next),
                  "persistencia": int(pers.get(cv, 0)), "visible_por_conteo": cv in top_c,
                  "rank_riesgo_violencia": int(cv_rank[cv]), "senal_sin_violencia": bool(cv_rank[cv] > n_aviso)})

# ============================================================ validacion.json
# "Si lo hubiéramos hecho en 2024": la misma receta con datos hasta 2023 (M2 reentrenado sin 2024, 04g) y lo que pasó.
# Éxito = más robos de los esperados por su actividad en 2024 (métrica de 04c). Desde v0.6 se exportan todas las AGEB y
# todo lo que la app muestra en la prueba sale del modelo que no vio 2024: `rr`, `p_exceso` y `categoria` con 2020–2023
# (la señal) y `serie` con O y Ê de esos años; `esperados` y `observados` son 2024 (el resultado).
T_VAL = int(ANIOS[-1])
val["categoria"] = categoria(val.p_exc)
bt = pd.read_parquet(MODELOS / "validacion_2024_anual.parquet")
bt = bt[bt.anio < T_VAL]
serie_bt = {cv: [{"anio": int(a), "observados": int(o), "esperados": r3(e)} for a, o, e in zip(g.anio, g.y, g.Ehat)]
            for cv, g in bt.sort_values("anio").groupby("cvegeo")}
val["mas"] = val.O_sig > val.E_sig
exc_val = (val.O_sig - val.E_sig).clip(lower=0)


def resultado(L):
    x = val[L]
    return {"n": int(len(x)), "pct_mas_de_lo_esperado": r3(x.mas.mean()), "pct_por_encima": r3((x.p_exc_sig >= 0.9).mean()),
            "robos": int(x.O_sig.sum()), "esperados": r3(x.E_sig.sum()),
            "pct_exceso_ciudad": r3(exc_val[L].sum() / exc_val.sum())}


# Quién vive en las zonas de cada lista (Censo 2020, RESAGEBURB, filas de total por AGEB). v0.7. Supuesto: describen a
# los residentes de la AGEB, no a los dueños ni empleados de los negocios; sirven como contexto de la zona, no como causa
# del robo. Indicadores tomados del índice de marginación urbana de CONAPO (educación básica, salud, internet) más la
# escolaridad promedio. Se agregan como cociente de sumas (ponderados por población); "*" (confidencialidad) cuenta como
# faltante y la AGEB sale solo de ese indicador.
CENSO_V = ["POBTOT", "P_15YMAS", "GRAPROES", "P15YM_SE", "P15PRI_IN", "P15PRI_CO", "P15SEC_IN", "PSINDER", "VIVPAR_HAB", "VPH_INTER"]
zc = zipfile.ZipFile(RAW / "censo2020/RESAGEBURB_09_2020_csv.zip")
nc = [n for n in zc.namelist() if n.lower().endswith(".csv") and "resageburb" in n.lower()][0]
cz = pd.read_csv(zc.open(nc), dtype=str, usecols=["ENTIDAD", "MUN", "LOC", "AGEB", "MZA", *CENSO_V], encoding="utf-8-sig")
cz = cz[(cz.MZA == "000") & (cz.AGEB != "0000")]
cz.index = cz.ENTIDAD.str.zfill(2) + cz.MUN.str.zfill(3) + cz.LOC.str.zfill(4) + cz.AGEB
cz = cz[CENSO_V].apply(pd.to_numeric, errors="coerce").reindex(val.index)
assert cz.POBTOT.notna().all(), "AGEB de la prueba sin censo"
cz["SIN_BASICA"] = cz[["P15YM_SE", "P15PRI_IN", "P15PRI_CO", "P15SEC_IN"]].sum(axis=1, min_count=4)
mediana_esc = cz.GRAPROES.median()


def vulnerabilidad(L):
    x = cz[L]

    def cociente(num, den):
        ok = x[num].notna() & x[den].notna()
        return x.loc[ok, num].sum() / x.loc[ok, den].sum()

    ok = x.GRAPROES.notna() & x.P_15YMAS.notna()
    return {"n": int(len(x)), "poblacion": int(x.POBTOT.sum()), "escolaridad": round(float(np.average(x.GRAPROES[ok], weights=x.P_15YMAS[ok])), 2),
            "pct_bajo_mediana_escolaridad": r3((x.GRAPROES.dropna() < mediana_esc).mean()),
            "pct_sin_basica": r3(cociente("SIN_BASICA", "P_15YMAS")), "pct_sin_salud": r3(cociente("PSINDER", "POBTOT")),
            "pct_sin_internet": r3(1 - cociente("VPH_INTER", "VIVPAR_HAB"))}


grupos_val = {"lista_riesgo": val.en_riesgo, "lista_conteo": val.en_conteo, "no_visibles": val.en_riesgo & ~val.en_conteo,
              "ciudad": val.index == val.index}
validacion = {
    "anio": T_VAL, "datos": f"{ANIOS[0]}–{T_VAL - 1}",
    "resumen": {k: resultado(L) for k, L in grupos_val.items()},
    "vulnerabilidad": {k: vulnerabilidad(L) for k, L in grupos_val.items()},
    "zonas": [{"cvegeo": cv, "colonia": at.colonia[cv], "cve_col": at.cve_col[cv], "alcaldia": at.alcaldia[cv],
               "rank_riesgo": int(r.rank_riesgo), "rank_conteo": int(r.rank_conteo), "en_lista_riesgo": bool(r.en_riesgo),
               "en_lista_conteo": bool(r.en_conteo), "rr": r3(r.rr), "p_exceso": r3(r.p_exc), "categoria": r.categoria,
               "serie": serie_bt[cv], "esperados": r3(r.E_sig), "observados": int(r.O_sig), "mas_de_lo_esperado": bool(r.mas)}
              for cv, r in val.sort_values("rank_riesgo").iterrows()],
}
print("prueba {}: {:.0%} de la lista por riesgo con más robos de lo esperado (conteo {:.0%}, ciudad {:.0%})".format(
    T_VAL, *(validacion["resumen"][k]["pct_mas_de_lo_esperado"] for k in ["lista_riesgo", "lista_conteo", "ciudad"])))
print(f"aviso sin violencia: {sum(x['senal_sin_violencia'] for x in prior)} de {len(prior)} prioritarias (puesto > {n_aviso})")

# ============================================================ ageb.geojson y fuera_analisis.geojson
def coleccion_gj(geoms, props):
    """FeatureCollection en WGS84 (5 decimales) a partir de geometrías en el CRS del Marco y una lista de propiedades."""
    g = gpd.GeoSeries(geoms, crs=CRS).to_crs("EPSG:4326")
    feats = []
    for geom, pr in zip(shapely.set_precision(g.values, 1e-5), props):
        gj = shapely.geometry.mapping(geom)
        feats.append({"type": "Feature", "properties": pr,
                      "geometry": {"type": gj["type"], "coordinates": json.loads(json.dumps(gj["coordinates"]))}})
    return {"type": "FeatureCollection", "features": feats}


# Supuesto (cobertura): el análisis usa solo las AGEB URBANAS. Las 22 rurales son suelo de conservación, bosque y
# cultivo (2 a 163 km², frente a 0.23 km² de mediana urbana): concentran ~0.05% del robo a negocio y ~0.3% de los
# establecimientos (ver `cobertura` en meta.json), el censo no publica su población por AGEB y no son zonas comparables
# con una urbana. Se exportan solo para dibujarlas como "fuera del análisis". Urbanas y rurales cubren el 100% de la
# CDMX y se simplifican juntas para que compartan fronteras; las alcaldías se arman uniendo sus AGEB.
geo = gpd.read_parquet(INTERIM / "ageb_geom.parquet")
CRS = geo.crs
rur = lee_mg("09ar")[["CVEGEO", "CVE_MUN", "geometry"]].to_crs(CRS)
simp = shapely.coverage_simplify(np.concatenate([geo.geometry.values, rur.geometry.values]), tolerance=SIMPLIFICA_M)
geo["geometry"], rur["geometry"] = simp[:len(geo)], simp[len(geo):]
geo = geo.join(at[["cve_mun", "alcaldia", "colonia", "cve_col", "POBTOT"]], on="cvegeo")
geojson = coleccion_gj(geo.geometry.values, [
    {"cvegeo": r.cvegeo, "alcaldia": r.alcaldia, "colonia": r.colonia, "cve_col": r.cve_col, "poblacion_2020": int(r.POBTOT)}
    for r in geo.itertuples(index=False)])
nom_mun = at.groupby("cve_mun").alcaldia.first()
rur["area_km2"] = rur.area / 1e6
rural_gj = coleccion_gj(rur.geometry.values, [
    {"cvegeo": r.CVEGEO, "alcaldia": nom_mun[r.CVE_MUN], "area_km2": r3(r.area_km2)} for r in rur.itertuples(index=False)])

# ============================================================ colonias.json
# Supuesto (interpolación areal): los delitos, el esperado y los establecimientos de una AGEB se reparten entre
# las colonias que la cubren en proporción al área. O y Ê de la colonia se suman y se recalcula la contracción
# con (a, b) propios del nivel colonia (no se promedian rr de AGEB).
col = gpd.read_file(RAW / "colonias/catalogo/poligonos_colonias_cdmx.shp")[["cve_col", "colonia", "alc", "geometry"]]
col["colonia"] = col.colonia.map(limpia_colonia)
col["geometry"] = col.geometry.make_valid()
ag = gpd.read_parquet(INTERIM / "ageb_geom.parquet")
col = col.to_crs(ag.crs)
ov = gpd.overlay(ag, col[["cve_col", "geometry"]], how="intersection", keep_geom_type=True)
ov["a"] = ov.area
ov["w"] = ov.a / ov.cvegeo.map(ag.set_index("cvegeo").area)            # fracción de la AGEB en la colonia
ov["w_col"] = ov.a / ov.cve_col.map(col.set_index("cve_col").area)     # fracción de la colonia en la AGEB
# AGEB "de" la colonia (para el resaltado en el mapa): cubren > 5% de la AGEB o > 25% de la colonia
ov_l = ov[(ov.w > 0.05) | (ov.w_col > 0.25)]
comp_cols = [c for c in fijo.columns if c.startswith("ps_") and c.endswith("_fijo")]
# Lista del año siguiente (v0.5): se reparten O y Ê ponderados por año y el esperado del año siguiente.
x = ov.join(ant[["O_w", "E_w", "E_next"]], on="cvegeo", how="inner")
x = x.join(fijo[["E_fijo", *comp_cols]], on="cvegeo")
x["Ow"], x["Ew"], x["En"], x["Nw"] = x.O_w * x.w, x.E_w * x.w, x.E_next * x.w, x.E_fijo * x.w
for c in comp_cols:
    x[c] = x[c] * x.Nw
agg = x.groupby("cve_col")[["Ow", "Ew", "En", "Nw", *comp_cols]].sum()
agg = agg[(agg.Ew > 0) & (agg.Nw > 0)]
s = contraccion(agg.Ow, agg.Ew)
agg["rr"], agg["p_exc"] = s["rr"], s["p_exc"]
# Colonias similares: 5 vecinos más cercanos en log(establecimientos) y composición, estandarizados
F = pd.concat([np.log(agg.Nw).rename("logN"), agg[comp_cols].div(agg.Nw, axis=0)], axis=1)
F = (F - F.mean()) / F.std(ddof=0)
nn = NearestNeighbors(n_neighbors=6).fit(F.values)
_, idx = nn.kneighbors(F.values)
ids = F.index.to_numpy()
col_ll = col.set_index("cve_col").to_crs("EPSG:4326")
cen = col.set_index("cve_col").geometry.representative_point().to_crs("EPSG:4326")
agebs_de = ov_l.groupby("cve_col").cvegeo.agg(lambda s: sorted(set(s)))
colonias = []
for i, cc in enumerate(ids):
    b = col_ll.geometry[cc].bounds
    colonias.append({"cve_col": cc, "nombre": col_ll.colonia[cc], "alcaldia": col_ll.alc[cc],
                     "agebs": agebs_de.get(cc, []),
                     "centroide": [round(cen[cc].x, 5), round(cen[cc].y, 5)],
                     "bbox": [round(v, 5) for v in b], "categoria": str(categoria(agg.p_exc[cc])), "rr": r3(agg.rr[cc]),
                     "similares": [ids[j] for j in idx[i] if ids[j] != cc][:5]})
print(f"colonias con métricas: {len(colonias)} de {len(col)} (sin AGEB con establecimientos: {len(col) - len(colonias)})")


# ============================================================ niveles alcaldía y colonia (v0.4)
# Supuesto: el riesgo de una zona mayor se calcula sumando O y Ê de sus AGEB (no promediando rr) y recalculando la
# contracción con (a, b) propios del nivel, igual que colonias.json. En alcaldía la suma es exacta (las AGEB anidan en
# su alcaldía); en colonia se usa la misma interpolación areal. Con Ê de M2, que no incluye la alcaldía, el rr de una
# alcaldía mezcla robo y diferencias de denuncia entre alcaldías (04d): por eso se acompaña de la denuncia relativa.
def niveles(sig, anu, N, decimales):
    """sig: zona × (O_w, E_w, E_next) para el año siguiente; anu: (anio, zona) × (O, Ehat); N: establecimientos por zona.
    Devuelve {zona: {anticipado, por_anio}} con contracción y puestos propios de cada año y del año anticipado."""
    def tabla(g):
        s = contraccion(g.O, g.Ehat)
        g = g.assign(rr=s["rr"], p_exc=s["p_exc"])
        g["categoria"] = categoria(g.p_exc)
        g["rank_conteo"] = puesto(g.O)
        g["rank_riesgo"] = (g.p_exc + 1e-6 * g.rr).rank(ascending=False, method="first").astype(int)
        return {z: {"observados": round(float(r.O), decimales) if decimales else int(round(r.O)), "esperados": r3(r.Ehat),
                    "rr": r3(r.rr), "p_exceso": r3(r.p_exc), "categoria": r.categoria,
                    "establecimientos": int(round(N[z])), "rank_conteo": int(r.rank_conteo), "rank_riesgo": int(r.rank_riesgo)}
                for z, r in g.iterrows()}
    t = tabla(sig.rename(columns={"O_w": "O", "E_w": "Ehat"}))
    out = {z: {"anticipado": {**m, "observados": r3(sig.O_w[z] / W), "esperados": r3(sig.E_next[z])}, "por_anio": {}}
           for z, m in t.items()}
    for a, g in anu.groupby(level=0):
        for z, m in tabla(g.droplevel(0)).items():
            out[z]["por_anio"][str(a)] = m
    return out


mun_de = at.cve_mun
alc_metrics = niveles(
    ant.groupby(mun_de.reindex(ant.index))[["O_w", "E_w", "E_next"]].sum(),
    anual.assign(z=anual.cvegeo.map(mun_de)).groupby(["anio", "z"])[["O", "Ehat"]].sum(),
    E[E > 0].groupby(mun_de).sum(), 0)
xa = ov[ov.cve_col.isin(agg.index)][["cvegeo", "cve_col", "w"]].merge(anual[["cvegeo", "anio", "O", "Ehat"]], on="cvegeo")
xa["O"], xa["Ehat"] = xa.O * xa.w, xa.Ehat * xa.w
col_metrics = niveles(agg.rename(columns={"Ow": "O_w", "Ew": "E_w", "En": "E_next"})[["O_w", "E_w", "E_next"]],
                      xa.groupby(["anio", "cve_col"])[["O", "Ehat"]].sum(), agg.Nw, 1)
assert all(abs(col_metrics[c["cve_col"]]["anticipado"]["rr"] - c["rr"]) < 1e-9 for c in colonias), "colonias.json desalineado"
print("alcaldías por encima (anticipado):", sum(m["anticipado"]["categoria"] == "por_encima" for m in alc_metrics.values()),
      "de", len(alc_metrics), "| colonias por encima:", sum(m["anticipado"]["categoria"] == "por_encima" for m in col_metrics.values()))

# Denuncia relativa por alcaldía (04d): ya no se exporta (2026-10-14). 04d concluyó que el
# indicador de la ENSU mide población flotante, no denuncia, y la ENVIPE (04e) muestra denuncia casi pareja entre
# alcaldías: mostrarlo en la app como "aquí se denuncia más/menos" contradecía el análisis.
r_alc = {}

# Geometría de alcaldías: unión de sus AGEB urbanas y rurales ya simplificadas (fronteras idénticas a las del mapa de AGEB).
partes = pd.concat([pd.DataFrame({"m": geo.cve_mun, "g": geo.geometry.values}),
                    pd.DataFrame({"m": rur.CVE_MUN, "g": rur.geometry.values})])
alc_geom = partes.groupby("m").g.agg(lambda gs: shapely.union_all(gs.values))  # hay traslapes mínimos urbana-rural
a_urb = gpd.GeoSeries(geo.geometry.values, crs=CRS).area.groupby(geo.cve_mun.values).sum()
n_prior = pd.Series([at.cve_mun[p["cvegeo"]] for p in prior]).value_counts()
alcaldias_gj = coleccion_gj(alc_geom.values, [
    {"cve_mun": m, "alcaldia": nom_mun[m], "agebs_urbanas": int((geo.cve_mun == m).sum()),
     "agebs_rurales": int((rur.CVE_MUN == m).sum()), "pct_area_urbana": r3(a_urb[m] / alc_geom[m].area),
     "prioritarias": int(n_prior.get(m, 0)),
     **({"denuncia_relativa": r_alc[nom_mun[m]][0], "denuncia_relativa_ic": r_alc[nom_mun[m]][1]} if r_alc else {})}
    for m in alc_geom.index])

# Geometría de colonias: el catálogo completo (las que no tienen métricas se pintan como sin actividad económica).
col_s = shapely.coverage_simplify(col.geometry.values, tolerance=SIMPLIFICA_M)
colonias_gj = coleccion_gj(col_s, [{"cve_col": r.cve_col, "nombre": r.colonia, "alcaldia": r.alc} for r in col.itertuples(index=False)])

# Cobertura del análisis (supuesto de la sección ageb.geojson): qué se deja fuera al usar solo AGEB urbanas.
v2b = pd.read_parquet(INTERIM / "delitos_ageb.parquet", columns=["anio_hecho", "grupo", "objetivo", "farderos", "tipo_asig"])
v2b = v2b[(v2b.objetivo == "negocio") & (v2b.farderos == 0) & (v2b.grupo != "negocio_sv_dentro") & v2b.anio_hecho.isin(ANIOS)
          & v2b.tipo_asig.isin(["URBANA", "RURAL"])]
den = pd.read_parquet(RAW / f"denue/parquet/denue_09_{CORTE_FIJO}_11.parquet", columns=["latitud", "longitud"])
den = den.apply(pd.to_numeric, errors="coerce").dropna()
den_ageb = asigna_ageb(den, lee_mg("09a")[["CVEGEO", "geometry"]], lee_mg("09ar")[["CVEGEO", "geometry"]])
den_ageb = pd.Series(np.asarray(den_ageb))
den_ageb = den_ageb[den_ageb != "SIN_ASIGNAR"]
cobertura = {"agebs_urbanas": len(geo), "agebs_rurales": len(rur),
             "pct_area_urbana": r3(a_urb.sum() / alc_geom.map(lambda g: g.area).sum()),
             "pct_robos_rurales": float(round((v2b.tipo_asig == "RURAL").mean(), 5)),
             "pct_establecimientos_rurales": float(round((den_ageb == "RURAL").mean(), 5))}
# Huecos del catálogo de colonias dentro de la zona urbana (sobre todo pueblos del sur): la interpolación areal no
# reparte esa parte de la AGEB a ninguna colonia, así que en el nivel colonia esos robos no aparecen (sí en AGEB y alcaldía).
cub = ov.groupby("cvegeo").w.sum().clip(upper=1).reindex(ag.cvegeo).fillna(0).values
a_ag = ag.area.values
O_ag = anual.groupby("cvegeo").O.sum().reindex(ag.cvegeo).fillna(0).values
cobertura["pct_area_urbana_sin_colonia"] = r3(((1 - cub) * a_ag).sum() / a_ag.sum())
cobertura["pct_robos_sin_colonia"] = r3(((1 - cub) * O_ag).sum() / O_ag.sum())
print("cobertura:", cobertura)

# ============================================================ giros.json
# Grupos = partición de M1. IRR = razón de tasas por +10 puntos porcentuales del grupo en la zona, a costa de la
# categoría de referencia (servicios profesionales y financieros). Mensaje según el IC 95% (bootstrap de bloques).
GIROS = [
    ("efectivo_barrio", "Abarrotes, tortillería, carnicería, frutería o venta de licores"),
    ("efectivo_formal", "Minisúper, farmacia, gasolinera, restaurante, bar o casa de empeño"),
    ("comercio_menor", "Otro comercio al por menor (ropa, papelería, ferretería…)"),
    ("comercio_mayor", "Comercio al por mayor"),
    ("alojamiento_alim", "Hotel u hospedaje"),
    ("manufactura", "Taller de fabricación o manufactura"),
    ("serv_otros", "Taller de reparación, estética u otro servicio personal"),
    ("serv_salud_educ", "Consultorio, escuela, gimnasio o centro recreativo"),
    ("serv_prof_fin", "Oficina, servicio profesional, financiero o inmobiliario"),
    ("otros", "Otro giro"),
]
CONSEJO = {
    "efectivo_barrio": "Retirar el efectivo de la caja varias veces al día y aceptar pagos con tarjeta o transferencia reduce lo que se puede perder.",
    "efectivo_formal": "Reducir el efectivo en caja, cuidar los horarios de apertura y cierre y usar pagos digitales ayuda a bajar el riesgo.",
}
# Análisis posterior 04c (opcional): IRR por tipo de robo y validación 2020–2023 → 2024. Si no existe, se omiten los textos.
RC = ROOT / "analisis/pipeline/outputs/modelos/robustez_C.csv"
rc = pd.read_csv(RC).set_index("metrica") if RC.exists() else None


def irr_cv(gid):
    """IRR (+10 pp) del grupo en robo con violencia y su IC 95%, de 04c; None si no está."""
    m = f"IRR +10 pp {gid} · con violencia"
    if rc is None or m not in rc.index:
        return None
    lo, hi = (float(x) for x in rc.nota[m].split("[")[1].rstrip("]").split(","))
    return float(rc.valor[m]), lo, hi


scian = {g: [p for c in cs for p in CANDIDATOS_EFECTIVO[c]] for g, cs in EFECTIVO.items()}
scian |= {g: list(ps) for g, ps in SECTORES.items()}
scian["otros"] = ["11", "21", "22", "23", "48", "49", "93"]
cf = coef.set_index("variable")
giros = []
for gid, nombre in GIROS:
    v = f"s_{gid}"
    if v in cf.index:
        irr, lo, hi = cf.irr10_coef[v], cf.irr10_ic_lo[v], cf.irr10_ic_hi[v]
        if lo > 1:
            msg = (f"En las zonas donde hay más negocios como el tuyo se denuncian más robos que en otras zonas con el mismo "
                   f"número de negocios (alrededor de {round((irr - 1) * 100)}% más por cada 10 de cada 100 negocios de este tipo).")
        elif hi < 1:
            msg = ("En las zonas donde hay más negocios como el tuyo se denuncian menos robos que en otras con el mismo número de "
                   "negocios. Ojo: los negocios pequeños denuncian mucho menos, así que esto puede reflejar menos denuncias y no "
                   "necesariamente menos robos.")
        else:
            msg = "No encontramos una diferencia clara en los robos denunciados entre zonas con más o menos negocios de este tipo."
        cv = irr_cv(gid)
        if cv and hi < 1 and cv[1] <= 1 <= cv[2]:
            msg = ("En las zonas donde hay más negocios como el tuyo se denuncian menos robos sin violencia que en otras con el "
                   "mismo número de negocios, pero en los robos con violencia no hay diferencia. "
                   "Puede ser que en estos negocios se denuncie menos, no que les roben menos. Denunciar ayuda a que "
                   "los problemas de tu zona se vean.")
        elif cv and lo > 1 and cv[1] > 1:
            msg += f" La diferencia es mayor en los robos con violencia (alrededor de {round((cv[0] - 1) * 100)}% más)."
        msg = f"{msg} {CONSEJO.get(gid, '')}".strip()
        irr_v = r3(irr)
    else:
        irr_v = None
        msg = "Es el grupo de referencia: los demás giros se comparan con él."
    giros.append({"id": gid, "nombre": nombre, "scian": scian[gid], "irr": irr_v, "mensaje": msg})

# ============================================================ meta.json
P = resumen["principal"]
# β de cabecera de A: el condicional de M1 (decisión del 2026-10-14). El registrado sin controles (P["beta"]) se reporta en el paper.
BA = coef.set_index("variable").loc["logE"]
# Lectura de A en malla de área constante (T7 partes 1b–1c, decidida el 2026-10-14): colchón de densidad, con el nodo
# estimado (parte 1c), no fijo en el p75. Cae cerca de la mediana, pero su ubicación es imprecisa: no se cita un umbral.
colchon = json.load(open(ROOT / "app/src/data/colchon.json"))["m1"]
A_, B_, C_ = P["A"], P["B"], P["C"]
# Mismo criterio que el contorno cian de la app: fuera del top por conteo y "por encima de lo esperado"
# (2026-10-14). Las demás no visibles por conteo están en la lista con un exceso menos claro.
inv = [p for p in prior if not p["visible_por_conteo"] and p["p_exceso"] >= UMBRALES["por_encima"]]
n_inv = len(inv)
CENTRO = ["Cuauhtémoc", "Benito Juárez", "Miguel Hidalgo"]  # = ALCALDIAS_CENTRO de la app
assert set(CENTRO) <= set(at.alcaldia), "alcaldías centrales con otro nombre"
inv_centro = np.mean([p["alcaldia"] in CENTRO for p in inv])
rv, vv = validacion["resumen"], validacion["vulnerabilidad"]
meta = {
    "generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "version_contrato": "0.8",
    "anios": ANIOS,
    "anticipado": {"anio": T_VAL + 1, "datos": f"{ANIOS[0]}–{ANIOS[-1]}", "vida_media_anios": VIDA_MEDIA},
    "variable_objetivo": "Robo a negocio con y sin violencia, sin farderos ni robo sin violencia dentro de autoservicios y tiendas de conveniencia",
    "modelo_esperado": resumen["modelo_esperado"],
    "umbrales_categoria": UMBRALES,
    # Resumen ejecutivo (2026-10-14): una lectura y tres evidencias por hipótesis. La lista 2025 y
    # la prueba 2024 se citan en viñetas separadas para no mezclar los cortes.
    "hipotesis": {
        "A": {"titulo": "¿Cuánto?",
              "enunciado": ("El robo a negocio crece menos que proporcionalmente con el número de establecimientos (β < 1), "
                            "a igual giro, tamaño y entorno urbano."),
              "estadistico": f"β = {BA.coef:.2f} [{BA.ic_lo:.2f}, {BA.ic_hi:.2f}]",
              "resultado": A_,
              **({"lectura": ("El robo crece con el comercio, pero cada vez más despacio: en las zonas con más negocios aparece "
                              "un colchón de densidad."),
                  "evidencia": [
                      f"Al duplicar los negocios, el robo sube ~{(2 ** BA.coef - 1) * 100:.0f}%, no el doble: contar robos "
                      "exagera el riesgo de los polos comerciales.",
                      f"En la mitad de la ciudad con más comercio, duplicar los negocios multiplica el robo por "
                      f"×{2 ** colchon['beta_alto']:.1f}; en la otra mitad, por ×{2 ** colchon['beta_bajo']:.1f}.",
                      "La explicación probable, que no probamos: más gente en la calle, más vigilancia y más inversión en "
                      "seguridad. Es una señal a favor de que una economía local activa acompaña menos riesgo para cada "
                      "negocio, no una prueba causal.",
                  ]} if A_ == "confirmada" else
                 {"lectura": "El robo crece al menos al mismo ritmo que los negocios: la densidad comercial no protege.",
                  "evidencia": []})},
        "B": {"titulo": "¿Cuáles?",
              "enunciado": "La composición de giros (sobre todo los de alto efectivo) explica el robo mejor que el número de negocios.",
              "estadistico": f"Mejora fuera de muestra = {P['dD'] * 100:.1f}% (en {P['folds_mejora']} de 5 bloques)",
              "resultado": B_,
              **({"lectura": ("Qué negocios hay, no solo cuántos: el giro y el tamaño de los establecimientos mejoran la "
                              "estimación del robo esperado."),
                  "evidencia": [
                      f"Con giro, tamaño y entorno, la predicción en zonas que el modelo no vio mejora {P['dD'] * 100:.1f}% "
                      f"frente a solo contar negocios (en {P['folds_mejora']} de 5 bloques de la ciudad).",
                      "Los giros de alto efectivo formal (minisúpers, farmacias, gasolineras, restaurantes) tienen más robo, "
                      "con y sin violencia: ahí va la prevención.",
                      "Los de alto efectivo tradicional (abarrotes, tortillerías) tienen el mismo robo con violencia, pero "
                      "denuncian menos: ahí conviene promover la denuncia.",
                  ]} if B_ == "confirmada" else
                 {"lectura": "El tipo de negocio no agrega información suficiente: focalizar por zona.", "evidencia": []})},
        "C": {"titulo": "¿Dónde?",
              "enunciado": "Priorizar por riesgo ajustado señala zonas distintas que priorizar por conteo, y esas zonas persisten en el tiempo.",
              "estadistico": f"Coincidencia con el top por conteo (Jaccard) = {P['jaccard']:.2f}; persistencia (ρ) = {P['rho']:.2f}",
              "resultado": C_,
              **({"lectura": ("No dejar fuera a la periferia: el riesgo ajustado encuentra zonas con más robo del esperado que "
                              "el conteo no ve, y con menos recursos para absorberlo."),
                  "evidencia": [
                      f"{n_inv} de las {len(prior)} zonas prioritarias para {T_VAL + 1} no están entre las de más robos; solo "
                      f"el {inv_centro:.0%} está en {', '.join(CENTRO[:-1])} o {CENTRO[-1]}.",
                      f"Habría funcionado: con datos hasta {T_VAL - 1}, el {rv['lista_riesgo']['pct_mas_de_lo_esperado']:.0%} "
                      f"de la lista tuvo más robos de los esperados en {T_VAL}, contra el "
                      f"{rv['lista_conteo']['pct_mas_de_lo_esperado']:.0%} de la lista por conteo y el "
                      f"{rv['ciudad']['pct_mas_de_lo_esperado']:.0%} de la ciudad.",
                      f"En esa prueba, las zonas que el conteo no veía tienen menos recursos (Censo 2020): el "
                      f"{vv['no_visibles']['pct_bajo_mediana_escolaridad']:.0%} está bajo la escolaridad mediana, contra el "
                      f"{vv['lista_conteo']['pct_bajo_mediana_escolaridad']:.0%} de la lista por conteo. Ahí un mismo robo "
                      "pesa más.",
                  ]} if C_ == "confirmada" else
                 {"lectura": "El riesgo ajustado no cambia o no es estable: usarlo con cautela.", "evidencia": []})},
    },
    # Todas las fuentes que usa el análisis, no solo las que alimentan el mapa: el corte dice qué parte se usa y para qué.
    "fuentes": [
        {"nombre": "Carpetas de investigación FGJ CDMX (API, año del hecho)",
         "corte": "Hechos 2016–2024 (consulta con inicio hasta ene-2025): 2020–2024 en el análisis y la lista; 2016–2018 como "
                  "robo previo en la prueba de causalidad; 2019 en una prueba de sensibilidad",
         "url": "https://datos.cdmx.gob.mx/dataset/carpetas-de-investigacion-fgj-de-la-ciudad-de-mexico"},
        {"nombre": "DENUE (INEGI)",
         "corte": "Noviembre 2023 en el análisis y la lista; enero 2016 y noviembre 2018, 2019 y 2024 en la prueba de causalidad; "
                  "los 19 cortes de 2016 a 2026 para comprobar que casi no cambia entre censos",
         "url": "https://www.inegi.org.mx/app/mapa/denue/"},
        {"nombre": "Censos Económicos 2019 y 2024 (INEGI)", "corte": "Para comprobar que el DENUE mide el cambio de negocios entre censos",
         "url": "https://www.inegi.org.mx/programas/ce/2024/"},
        {"nombre": "Demografía de los Negocios (EDN) 2023 (INEGI)", "corte": "Mayo 2019 – mayo 2023: cierres y aperturas de referencia",
         "url": "https://www.inegi.org.mx/programas/edn/"},
        {"nombre": "ENVE 2024 (INEGI)", "corte": "Delitos de 2023: cifra negra por tamaño de negocio en la CDMX",
         "url": "https://www.inegi.org.mx/programas/enve/2024/"},
        {"nombre": "ENVIPE 2021–2025 (INEGI)", "corte": "Delitos de 2020–2024: denuncia por alcaldía (edición 2026 como sensibilidad)",
         "url": "https://www.inegi.org.mx/programas/envipe/"},
        {"nombre": "ENSU (INEGI)", "corte": "Sep. 2019 – dic. 2025: se probó como medida de denuncia por alcaldía y se descartó "
                                           "(refleja población flotante)",
         "url": "https://www.inegi.org.mx/programas/ensu/"},
        {"nombre": "Marco Geoestadístico 2020 (INEGI)", "corte": "2020: AGEB urbanas y rurales, manzanas", "url": "https://www.inegi.org.mx/temas/mg/"},
        {"nombre": "Censo de Población y Vivienda 2020 (INEGI)", "corte": "2020: población, escolaridad y otros rasgos por AGEB urbana",
         "url": "https://www.inegi.org.mx/programas/ccpv/2020/"},
        {"nombre": "Catálogo de colonias CDMX (Datos Abiertos)", "corte": "Febrero 2023",
         "url": "https://datos.cdmx.gob.mx/dataset/02c6ce99-dbd8-47d8-aee1-ae885a12bb2f"},
        {"nombre": "Estaciones de Metro y Metrobús (Datos Abiertos CDMX)",
         "corte": "Con año de apertura: distancia a la estación más cercana en cada año, 2016–2024", "url": "https://datos.cdmx.gob.mx/"},
    ],
    "cobertura": cobertura,
}

# ============================================================ escribir
print("== export →", OUT)
kb = {"meta.json": escribe("meta.json", meta, Meta)}
kb["ageb.geojson"] = escribe("ageb.geojson", geojson, AgebGeoJSON)
kb["ageb_metrics.json"] = escribe("ageb_metrics.json", metrics, AgebMetrics)
kb["prioridades.json"] = escribe("prioridades.json", prior, Prioridad, lista=True)
kb["colonias.json"] = escribe("colonias.json", colonias, ColoniaIndex, lista=True)
kb["giros.json"] = escribe("giros.json", giros, GrupoGiro, lista=True)
kb["alcaldias.geojson"] = escribe("alcaldias.geojson", alcaldias_gj, AlcaldiasGeoJSON)
kb["alcaldias_metrics.json"] = escribe("alcaldias_metrics.json", alc_metrics, ZonaMetrics)
kb["colonias.geojson"] = escribe("colonias.geojson", colonias_gj, ColoniasGeoJSON)
kb["colonias_metrics.json"] = escribe("colonias_metrics.json", col_metrics, ZonaMetrics)
kb["fuera_analisis.geojson"] = escribe("fuera_analisis.geojson", rural_gj, RuralGeoJSON)
kb["validacion.json"] = escribe("validacion.json", validacion, Validacion)
LIM = {"validacion.json": 1536, "meta.json": 10, "ageb.geojson": 3 * 1024, "ageb_metrics.json": 4 * 1024, "prioridades.json": 100,
       "colonias.json": 1024, "giros.json": 10, "alcaldias.geojson": 300, "alcaldias_metrics.json": 50,
       "colonias.geojson": 1.5 * 1024, "colonias_metrics.json": 2 * 1024, "fuera_analisis.geojson": 300}
excede = {k: round(v) for k, v in kb.items() if v > LIM[k]}
print("tamaños dentro del objetivo" if not excede else f"EXCEDEN el objetivo del contrato: {excede}")
