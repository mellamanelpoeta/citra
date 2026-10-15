"""01 · Carga, homologación y unión espacial.

Entradas (data/raw/_samples/): parquet de la API FGJ, cortes DENUE de noviembre 2019–2024, MG2020,
RESAGEBURB 2020, catálogo de colonias CDMX, SHP de estaciones de Metro y Metrobús.

Salidas:
- data/interim/delitos_ageb.parquet      un registro por carpeta (delitos de interés), con grupo homologado y AGEB
- data/interim/denue_ageb.parquet        un registro por establecimiento × corte, con SCIAN, estrato y AGEB
- data/interim/ageb_atributos.parquet    atributos estáticos por AGEB (alcaldía, colonia dominante, población, área)
- data/interim/ageb_dist_estacion.parquet  distancia a la estación de Metro/Metrobús abierta, por AGEB × año
- analisis/pipeline/outputs/homologacion_fgj_final.csv  tabla de homologación con conteos por año del hecho (evidencia D2)

Uso: .venv/bin/python analisis/pipeline/01_datos.py
"""
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd

from common import ANIOS, CONFIG, INTERIM, RAW, ROOT, asigna_ageb, lee_mg, norm

urb = lee_mg("09a")[["CVEGEO", "CVE_MUN", "geometry"]]
rur = lee_mg("09ar")[["CVEGEO", "geometry"]]
ent = lee_mg("09ent")
CRS = urb.crs  # LCC ITRF2008 en metros: distancias y áreas se calculan aquí


def coord_valida(df):
    """Supuesto: coordenada válida = no nula, distinta de 0 y dentro del polígono estatal 09ent.
    Fuera de eso no hay forma de ubicar el hecho; se pierde <0.4% de robo a negocio y ≤3% de vía pública."""
    lat, lon = df.latitud, df.longitud
    ok = lat.notna() & lon.notna() & (lat != 0) & (lon != 0)
    ok &= lat.between(19.0, 19.65) & lon.between(-99.40, -98.90)
    g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon[ok], lat[ok]), index=df.index[ok], crs="EPSG:4326").to_crs(CRS)
    out = pd.Series(False, index=df.index)
    out.loc[ok] = g.within(ent.geometry.iloc[0]).values
    return out


# ============================================================ FGJ
print("== FGJ")
api = pd.read_parquet(RAW / "fgj/fgj_delitos_objetivo_2016_2025.parquet")
# Supuesto: la unidad temporal es el AÑO DEL HECHO. El rezago de denuncia es despreciable (98% se denuncia
# el mismo año) y la API llega hasta fecha_inicio 2025-01-31, así que 2024 está completo al ~99%.
api["fecha_hecho"] = pd.to_datetime(api.fecha_hecho.astype(str).str[:10], errors="coerce", format="%Y-%m-%d")
api["anio_hecho"] = api.fecha_hecho.dt.year
for c in ["latitud", "longitud"]:
    api[c] = pd.to_numeric(api[c], errors="coerce")
api["delito_norm"] = api.delito.map(norm)

homol = pd.read_csv(CONFIG / "homologacion_fgj.csv")
falt = set(api.delito_norm.dropna()) - set(homol.delito_norm)
assert not falt, f"Delitos sin homologar en config/homologacion_fgj.csv: {falt}"
api = api.merge(homol[["delito_norm", "grupo", "objetivo", "farderos"]], on="delito_norm", how="left")

# tabla final de homologación con conteos (año del hecho 2019–2024, antes de filtrar coordenadas)
cnt = (api[api.anio_hecho.isin(ANIOS)].pivot_table(index="delito_norm", columns="anio_hecho", values="_id",
                                                  aggfunc="size", fill_value=0))
cnt.columns = [f"n_{c}" for c in cnt.columns]
fin = homol.merge(cnt, left_on="delito_norm", right_index=True, how="left")
fin["fuente"] = np.where(fin.n_2019.notna(), "API", "solo CSV (no está en el parquet de la API)")
(ROOT / "analisis/pipeline/outputs").mkdir(exist_ok=True)
fin.to_csv(ROOT / "analisis/pipeline/outputs/homologacion_fgj_final.csv", index=False)

d = api[api.anio_hecho.isin(ANIOS) & (api.objetivo != "excluir")].copy()
d["coord_ok"] = coord_valida(d)
d["cvegeo"] = "SIN_COORD"
d.loc[d.coord_ok, "cvegeo"] = asigna_ageb(d[d.coord_ok], urb, rur)
d["tipo_asig"] = np.where(d.cvegeo.isin(["SIN_COORD", "RURAL", "SIN_ASIGNAR"]), d.cvegeo, "URBANA")
print(pd.crosstab(d.objetivo, d.tipo_asig, normalize="index").mul(100).round(2))
# latitud/longitud se conservan para la repetición en malla H3 (MAUP, T2b)
cols = ["_id", "fecha_hecho", "anio_hecho", "delito_norm", "grupo", "objetivo", "farderos", "cvegeo", "tipo_asig", "latitud", "longitud"]
d[cols].to_parquet(INTERIM / "delitos_ageb.parquet", index=False)
print("delitos de interés 2019–2024:", len(d), "| en AGEB urbana:", int((d.tipo_asig == "URBANA").sum()))

# ============================================================ DENUE
print("== DENUE")
# Supuesto: se usa el corte de NOVIEMBRE de cada año (el más cercano al cierre del año del hecho). El DENUE
# está congelado entre censos económicos (99% de ids persisten 2020–2023), así que E es casi transversal.
den = []
for y in ANIOS:
    e = pd.read_parquet(RAW / f"denue/parquet/denue_09_{y}_11.parquet", columns=["codigo_act", "per_ocu", "latitud", "longitud"])
    e["latitud"] = pd.to_numeric(e.latitud, errors="coerce")
    e["longitud"] = pd.to_numeric(e.longitud, errors="coerce")
    e = e.dropna(subset=["latitud", "longitud"])  # 100% no nulas en todos los cortes; por seguridad
    e["cvegeo"] = asigna_ageb(e, urb, rur)
    e["corte"] = y
    print(y, len(e), "% en AGEB urbana:", round((e.cvegeo.str.len() == 13).mean() * 100, 2))
    den.append(e[e.cvegeo.str.len() == 13][["corte", "cvegeo", "codigo_act", "per_ocu", "latitud", "longitud"]])
den = pd.concat(den, ignore_index=True)
den["codigo_act"] = den.codigo_act.astype(str).str.zfill(6)
den["corte"] = den.corte.astype("int16")
den.to_parquet(INTERIM / "denue_ageb.parquet", index=False)

# ============================================================ Atributos de AGEB
print("== Atributos AGEB")
mun = lee_mg("09mun")[["CVE_MUN", "NOMGEO"]].rename(columns={"NOMGEO": "alcaldia"})
at = urb.merge(mun, on="CVE_MUN")
at["area_km2"] = at.area / 1e6
cen = at.geometry.centroid
at["x"], at["y"] = cen.x, cen.y
cen_ll = cen.to_crs("EPSG:4326")
at["lon"], at["lat"] = cen_ll.x, cen_ll.y

# Población 2020 (RESAGEBURB, filas de total por AGEB: MZA == "000").
z = zipfile.ZipFile(RAW / "censo2020/RESAGEBURB_09_2020_csv.zip")
name = [n for n in z.namelist() if n.lower().endswith(".csv") and "resageburb" in n.lower()][0]
r = pd.read_csv(z.open(name), dtype=str, usecols=["ENTIDAD", "MUN", "LOC", "AGEB", "MZA", "POBTOT"], encoding="utf-8-sig")
r = r[(r.MZA == "000") & (r.AGEB != "0000")]
r["CVEGEO"] = r.ENTIDAD.str.zfill(2) + r.MUN.str.zfill(3) + r.LOC.str.zfill(4) + r.AGEB
at = at.merge(r[["CVEGEO", "POBTOT"]].assign(POBTOT=lambda x: pd.to_numeric(x.POBTOT)), on="CVEGEO", how="left")
assert at.POBTOT.notna().all(), "AGEB sin población"

# Colonia dominante por área (catálogo CDMX, el mismo que usa la FGJ en colonia_catalogo).
# Supuesto: la colonia con mayor área de intersección etiqueta a la AGEB. Es solo para búsqueda y
# lectura en la app; los modelos no usan la colonia.
col = gpd.read_file(RAW / "colonias/catalogo/poligonos_colonias_cdmx.shp")[["cve_col", "colonia", "alc", "geometry"]]
col["geometry"] = col.geometry.make_valid()
col = col.to_crs(CRS)
ov = gpd.overlay(at[["CVEGEO", "geometry"]], col, how="intersection", keep_geom_type=True)
ov["a"] = ov.area
ov = ov.sort_values("a", ascending=False)
tot = ov.groupby("CVEGEO").a.sum()
dom = ov.drop_duplicates("CVEGEO").set_index("CVEGEO")
dom["pct_area_colonia"] = (dom.a / at.set_index("CVEGEO").area.reindex(dom.index)).clip(upper=1)
at = at.merge(dom[["cve_col", "colonia", "pct_area_colonia"]], left_on="CVEGEO", right_index=True, how="left")
# Las AGEB que no intersectan ninguna colonia del catálogo (huecos del catálogo) toman la más cercana.
sin = at.cve_col.isna()
print("AGEB sin colonia que las intersecte:", int(sin.sum()), "→ se asigna la colonia más cercana")
nn = gpd.sjoin_nearest(at.loc[sin, ["CVEGEO", "geometry"]], col[["cve_col", "colonia", "geometry"]])
nn = nn[~nn.index.duplicated()]
at.loc[sin, ["cve_col", "colonia"]] = nn[["cve_col", "colonia"]].values
at.loc[sin, "pct_area_colonia"] = 0.0
print("AGEB sin colonia:", int(at.cve_col.isna().sum()),
      "| mediana % del área cubierta por la colonia dominante:", round(at.pct_area_colonia.median() * 100, 1))
# relación AGEB ↔ colonias (todas las que intersectan con >5% del área de la AGEB), para colonias.json en T2b
ov["pct_ageb"] = ov.a / ov.CVEGEO.map(at.set_index("CVEGEO").area)
ov[ov.pct_ageb > 0.05][["CVEGEO", "cve_col", "pct_ageb"]].rename(columns={"CVEGEO": "cvegeo"}).to_parquet(
    INTERIM / "ageb_colonia.parquet", index=False)

at = at.rename(columns={"CVEGEO": "cvegeo", "CVE_MUN": "cve_mun"})
at.drop(columns="geometry").to_parquet(INTERIM / "ageb_atributos.parquet", index=False)
at[["cvegeo", "geometry"]].to_parquet(INTERIM / "ageb_geom.parquet", index=False)

# ============================================================ Estaciones
print("== Estaciones")
mt = gpd.read_file(RAW / "movilidad/metro/stcmetro_shp/STC_Metro_estaciones_utm14n.shp")
mb = gpd.read_file(RAW / "movilidad/metrobus/mb_shp/Metrobus_estaciones.shp")
# Supuestos:
# - Una estación está abierta en el año t si su año de apertura (AÑO) es ≤ t.
# - Metrobús: 20 estaciones sin AÑO toman el año mínimo de su línea (todas son de líneas anteriores a 2019).
#   Se excluyen las 17 de "Servicio temporal".
# - Línea 12 del Metro: cerrada en 2021 y 2022 (colapso del 2021-05-03; reapertura parcial en enero de 2023).
#   Otros cierres temporales (p. ej., renovación de la L1) se ignoran: son parciales y el control es grueso.
mb = mb[mb.TIPO.fillna("") != "Servicio temporal"].copy()
mb["AÑO"] = mb["AÑO"].fillna(mb.groupby("LINEA")["AÑO"].transform("min"))
est = pd.concat([mt[["LINEA", "AÑO", "geometry"]].assign(sistema="metro"),
                 mb[["LINEA", "AÑO", "geometry"]].assign(sistema="metrobus")], ignore_index=True)
est = gpd.GeoDataFrame(est, crs=mt.crs).to_crs(CRS)
est["AÑO"] = est["AÑO"].astype(int)
pts = gpd.GeoDataFrame(at[["cvegeo"]], geometry=gpd.points_from_xy(at.x, at.y), crs=CRS)
rows = []
for y in ANIOS:
    abiertas = est[(est["AÑO"] <= y) & ~((est.sistema == "metro") & (est.LINEA == "12") & (y in (2021, 2022)))]
    j = gpd.sjoin_nearest(pts, abiertas[["geometry"]], distance_col="dist_m")
    j = j[~j.index.duplicated()]
    rows.append(pd.DataFrame({"cvegeo": j.cvegeo.values, "anio": y, "dist_estacion_m": j.dist_m.values, "n_estaciones": len(abiertas)}))
dist = pd.concat(rows, ignore_index=True)
print(dist.groupby("anio").agg(n_estaciones=("n_estaciones", "first"), mediana_m=("dist_estacion_m", "median")))
dist.drop(columns="n_estaciones").to_parquet(INTERIM / "ageb_dist_estacion.parquet", index=False)
print("listo")
