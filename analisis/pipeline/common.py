"""Rutas, constantes y utilidades compartidas del pipeline (01_datos, 02_features, 03_eda)."""
from pathlib import Path
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/_samples"
INTERIM = ROOT / "data/interim"
PROCESSED = ROOT / "data/processed"
CONFIG = ROOT / "analisis/pipeline/config"
OUT_EDA = ROOT / "analisis/pipeline/outputs/eda"
MG = RAW / "mg2020/conjunto_de_datos"
for p in (INTERIM, PROCESSED, OUT_EDA):
    p.mkdir(parents=True, exist_ok=True)

ANIOS = list(range(2019, 2025))
# Corte DENUE fijo para la variante S2(b): marco de Censos Económicos 2019, último corte antes del cambio de marco 2024-11.
CORTE_FIJO = 2023

# Estratos de personal ocupado del DENUE. Supuesto: el punto medio del estrato aproxima el tamaño
# (para "251 y más" se usa 300; solo se usa como variable descriptiva, no como exposición).
PER_OCU = {
    "0 a 5 personas": 3, "6 a 10 personas": 8, "11 a 30 personas": 20, "31 a 50 personas": 40,
    "51 a 100 personas": 75, "101 a 250 personas": 175, "251 y más personas": 300,
}
GRANDES = {"51 a 100 personas", "101 a 250 personas", "251 y más personas"}

# Giros candidatos a "intensivos en efectivo" (prefijos SCIAN 2018, estables en los cortes 2019–2024).
# Son mutuamente excluyentes; la lista final es una unión de candidatos (EFECTIVO, abajo).
CANDIDATOS_EFECTIVO = {
    "abarrotes": ("461110",),
    "alim_otros": ("4611",),            # resto de 4611: carnicerías, pollerías, fruterías, lácteos, dulces…
    "minisuper": ("462112",),
    "licores": ("461211", "461212", "461220"),
    "farmacias": ("46411",),            # incluye naturistas (464113), como el 46411 del diseño
    "gasolineras": ("46841",),          # gasolina, gas LP y otros combustibles al por menor
    "restaurantes": ("7225", "7223"),   # restaurantes, taquerías, loncherías, cafeterías y servicios de comedor
    "bares": ("7224",),
    "empeno_cambio": ("522452", "523121", "523122"),
    "tortillerias": ("311830",),
}
# Lista tentativa del diseño (§3): 4611 + 462112 + 46411 + 46841 + 722. Se conserva solo como referencia.
LISTA_TENTATIVA = ["abarrotes", "alim_otros", "minisuper", "farmacias", "gasolineras", "restaurantes", "bares"]
# Lista FINAL (decidida 2026-10-10 en analisis/lab/, variante (b)): dos grupos definidos a priori por tipo de negocio.
# El EDA mostró que la lista tentativa mezcla señales opuestas (analisis/pipeline/03_eda.ipynb §4) y en el lab (b)
# superó a (a) y a (c) en devianza fuera de muestra (analisis/lab/resultados/decisiones_lab.md).
EFECTIVO = {
    "efectivo_barrio": ["abarrotes", "alim_otros", "licores", "tortillerias"],       # "tradicional" en textos: proximidad, micro y familiar
    "efectivo_formal": ["minisuper", "farmacias", "gasolineras", "restaurantes", "bares", "empeno_cambio"],  # ticket alto, cadena, caja
}

# Partición mutuamente excluyente del tejido económico (composición S de M1). "efectivo" va primero:
# los demás sectores son el resto de su prefijo.
SECTORES = {
    "comercio_menor": ("46",),
    "comercio_mayor": ("43",),
    "manufactura": ("31", "32", "33"),
    "alojamiento_alim": ("72",),
    "serv_prof_fin": ("51", "52", "53", "54", "55", "56"),
    "serv_salud_educ": ("61", "62", "71"),
    "serv_otros": ("81",),
}  # lo que no cae en ninguno (93 gobierno, 11, 21, 22, 23, 48-49) → "otros"


def clasifica(codigo, grupos, resto="otros"):
    """Asigna cada SCIAN al primer grupo cuyo prefijo coincide (prefijos más largos primero)."""
    pref = sorted(((p, g) for g, ps in grupos.items() for p in ps), key=lambda t: -len(t[0]))
    out = pd.Series(resto, index=codigo.index, dtype=object)
    asignado = pd.Series(False, index=codigo.index)
    for p, g in pref:
        m = ~asignado & codigo.str.startswith(p)
        out[m] = g
        asignado |= m
    return out


def norm(s):
    """Mayúsculas, sin acentos, espacios colapsados."""
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


def lee_mg(capa):
    return gpd.read_file(MG / f"{capa}.shp")


def puntos(df, lat="latitud", lon="longitud", crs=None):
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
    return g.to_crs(crs) if crs is not None else g


def asigna_ageb(df, urb, rur):
    """CVEGEO urbana de cada punto, o 'RURAL' / 'SIN_ASIGNAR'. `df` debe traer latitud/longitud válidas.

    Supuesto: un punto pertenece a la AGEB cuyo polígono lo contiene (predicado `within`); los puntos
    exactamente en la frontera entre dos AGEB se asignan a la primera coincidencia (casos despreciables).
    """
    g = puntos(df, crs=urb.crs)
    j = gpd.sjoin(g[["geometry"]], urb[["CVEGEO", "geometry"]], how="left", predicate="within")
    out = j[~j.index.duplicated()]["CVEGEO"].copy()
    falt = out.isna()
    if falt.any():
        jr = gpd.sjoin(g.loc[falt, ["geometry"]], rur[["CVEGEO", "geometry"]], how="left", predicate="within")
        jr = jr[~jr.index.duplicated()]
        out.loc[falt] = np.where(jr["CVEGEO"].notna(), "RURAL", "SIN_ASIGNAR")
    return out.values


def lee_panel():
    return pd.read_parquet(PROCESSED / "panel_ageb_anio.parquet")
