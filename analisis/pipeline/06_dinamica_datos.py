"""06 · Datos de T6 (inseguridad y dinámica empresarial) → data/processed/panel_dinamica.parquet

Protocolo: analisis/lab/dinamica/README.md (registrado antes de correr). Ticket: docs/tickets/T6-dinamica-empresarial.md.

Una fila por AGEB urbana (2,431) con:
- DENUE 2018-11, 2019-11, 2023-11 y 2024-11: N, personal ocupado estimado, composición (lista b), p_grandes; N por
  grupo de giros, por estrato y de 6+ empleados en 2019-11 y 2024-11.
- Robo por AGEB y año del hecho 2016–2023: V1 (negocio agregado), con violencia, sin violencia y V3 (vía pública).
- Distancia a estación 2016–2023, rasgos del Censo 2020 y % de superficie no residencial (manzanas).
- Valor agregado y producción bruta imputados (2018 con CE 2019 y DENUE 2019-11; 2023 con CE 2024 y DENUE 2024-11),
  a precios de 2023. Solo descriptivos.

Otras salidas: data/interim/ce_alc_sector_estrato.parquet (CE armonizados), data/interim/denue_alcaldia_sector.parquet
(DENUE completo por alcaldía × sector, para la prueba M), analisis/pipeline/outputs/dinamica/imputacion_*.csv.

Uso: .venv/bin/python analisis/pipeline/06_dinamica_datos.py   (~2 min la primera vez; cachea la asignación espacial)
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from common import (CONFIG, EFECTIVO, CANDIDATOS_EFECTIVO, GRANDES, INTERIM, PER_OCU, PROCESSED, RAW, ROOT, SECTORES,
                    asigna_ageb, clasifica, lee_mg, norm)

OUT = ROOT / "analisis/pipeline/outputs/dinamica"
OUT.mkdir(parents=True, exist_ok=True)
CORTES = [2018, 2019, 2023, 2024]          # noviembre de cada año
ANIOS_DEL = list(range(2016, 2024))
# Inflación promedio anual del INPC 2019–2023 (INEGI). Lleva pesos de 2018 a pesos de 2023. Solo descriptivo.
INPC_2018_2023 = float(np.prod([1.0364, 1.0340, 1.0569, 1.0790, 1.0553]))
# Estratos DENUE → estratos del Censo Económico (ID_ESTRATO 1–4). La agrupación es exacta.
ESTRATO_CE = {"0 a 5 personas": 1, "6 a 10 personas": 1, "11 a 30 personas": 2, "31 a 50 personas": 2,
              "51 a 100 personas": 3, "101 a 250 personas": 3, "251 y más personas": 4}
TAMANO = {1: "t0_10", 2: "t11_50", 3: "t51_mas", 4: "t51_mas"}

urb = lee_mg("09a")[["CVEGEO", "CVE_MUN", "geometry"]]
rur = lee_mg("09ar")[["CVEGEO", "geometry"]]
CRS = urb.crs
at = pd.read_parquet(INTERIM / "ageb_atributos.parquet")


def sector_ce(codigo):
    """Sector SCIAN a 2 dígitos con las agrupaciones del Censo Económico (31–33 y 48–49)."""
    s = codigo.str[:2]
    return s.replace({"31": "31-33", "32": "31-33", "33": "31-33", "48": "48-49", "49": "48-49"})


# ============================================================ DENUE por establecimiento
print("== DENUE")
cache = INTERIM / "_eda/denue2018_11_ageb.parquet"
if not cache.exists():
    e = pd.read_parquet(RAW / "denue/parquet/denue_09_2018_11.parquet", columns=["codigo_act", "per_ocu", "latitud", "longitud"])
    e[["latitud", "longitud"]] = e[["latitud", "longitud"]].apply(pd.to_numeric, errors="coerce")
    e = e.dropna(subset=["latitud", "longitud"])
    e["cvegeo"] = asigna_ageb(e, urb, rur)
    e = e[e.cvegeo.str.len() == 13].assign(corte=2018)
    e["codigo_act"] = e.codigo_act.astype(str).str.zfill(6)
    e[["corte", "cvegeo", "codigo_act", "per_ocu"]].to_parquet(cache, index=False)
den = pd.concat([pd.read_parquet(cache),
                 pd.read_parquet(INTERIM / "denue_ageb.parquet", columns=["corte", "cvegeo", "codigo_act", "per_ocu"])
                 .query("corte in [2019, 2023, 2024]")], ignore_index=True)
den["corte"] = den.corte.astype(int)
den["cand"] = clasifica(den.codigo_act, CANDIDATOS_EFECTIVO, resto="")
den["grupo"] = den.cand.map({c: g for g, cs in EFECTIVO.items() for c in cs}).fillna(clasifica(den.codigo_act, SECTORES))
den["po_est"] = den.per_ocu.map(PER_OCU)
den["estrato_ce"] = den.per_ocu.map(ESTRATO_CE)
assert den.po_est.notna().all() and den.estrato_ce.notna().all(), den.per_ocu[den.po_est.isna()].unique()
den["tamano"] = den.estrato_ce.map(TAMANO)
den["grande"] = den.per_ocu.isin(GRANDES)
den["seis_mas"] = den.per_ocu != "0 a 5 personas"
print(den.groupby("corte").size())

g = den.groupby(["cvegeo", "corte"])
base = pd.DataFrame({"N": g.size(), "po_est": g.po_est.sum(), "p_grandes": g.grande.mean(), "N6": g.seis_mas.sum()})
comp = pd.crosstab([den.cvegeo, den.corte], den.grupo)
base = base.join(comp.div(comp.sum(axis=1), axis=0).add_prefix("s_"))
base = base.join(pd.crosstab([den.cvegeo, den.corte], den.grupo).add_prefix("n_"))
base = base.join(pd.crosstab([den.cvegeo, den.corte], den.tamano).add_prefix("n_"))
wide = base.unstack("corte")
wide.columns = [f"{v}_{c}" for v, c in wide.columns]
wide = wide.reindex(at.cvegeo)
for c in wide.columns:
    if c.split("_")[0] in ("N", "N6", "po", "n"):
        wide[c] = wide[c].fillna(0)

# ============================================================ Robo 2016–2023
print("== Robo 2016–2023")
cache = INTERIM / "_eda/robo_2016_2018_ageb.parquet"
if not cache.exists():
    api = pd.read_parquet(RAW / "fgj/fgj_delitos_objetivo_2016_2025.parquet", columns=["delito", "fecha_hecho", "latitud", "longitud"])
    api["anio_hecho"] = pd.to_datetime(api.fecha_hecho.astype(str).str[:10], errors="coerce", format="%Y-%m-%d").dt.year
    api = api[api.anio_hecho.between(2016, 2018)].copy()
    api["delito_norm"] = api.delito.map(norm)
    homol = pd.read_csv(CONFIG / "homologacion_fgj.csv")
    api = api.merge(homol[["delito_norm", "grupo", "objetivo", "farderos"]], on="delito_norm", how="left")
    api = api[api.objetivo.isin(["negocio", "via_publica"])]
    api[["latitud", "longitud"]] = api[["latitud", "longitud"]].apply(pd.to_numeric, errors="coerce")
    # Mismo filtro de coordenadas que 01_datos.py (caja de la CDMX; lo que cae fuera del estado queda SIN_ASIGNAR)
    api = api[api.latitud.between(19.0, 19.65) & api.longitud.between(-99.40, -98.90) & (api.latitud != 0)]
    api["cvegeo"] = asigna_ageb(api, urb, rur)
    api = api[api.cvegeo.str.len() == 13]
    api[["anio_hecho", "cvegeo", "grupo", "objetivo"]].to_parquet(cache, index=False)
d = pd.concat([pd.read_parquet(cache),
               pd.read_parquet(INTERIM / "delitos_ageb.parquet", columns=["anio_hecho", "cvegeo", "grupo", "objetivo", "tipo_asig"])
               .query("tipo_asig == 'URBANA' and anio_hecho <= 2023").drop(columns="tipo_asig")], ignore_index=True)
neg = d.objetivo == "negocio"
cv = d.grupo.fillna("").str.startswith("negocio_cv")
# Supuesto 3: el robo a negocio agregado (V1) y sus totales con/sin violencia son comparables antes y después de la
# reclasificación de 2020, que solo movió subtipos (factibilidad §3.1).
TIPOS = {"v1": neg, "cv": neg & cv, "sv": neg & ~cv, "v3": d.objetivo == "via_publica"}
rob = pd.concat({t: d[m].groupby(["cvegeo", "anio_hecho"]).size() for t, m in TIPOS.items()}, axis=1).fillna(0)
rob = rob.unstack("anio_hecho").fillna(0)
rob.columns = [f"y_{t}_{a}" for t, a in rob.columns]
rob = rob.reindex(at.cvegeo).fillna(0).astype(int)
print(pd.DataFrame({t: [rob[f"y_{t}_{a}"].sum() for a in ANIOS_DEL] for t in TIPOS}, index=ANIOS_DEL))

# ============================================================ Distancia a estación 2016–2023 (regla de 01_datos.py)
print("== Estaciones")
mt = gpd.read_file(RAW / "movilidad/metro/stcmetro_shp/STC_Metro_estaciones_utm14n.shp")
mb = gpd.read_file(RAW / "movilidad/metrobus/mb_shp/Metrobus_estaciones.shp")
mb = mb[mb.TIPO.fillna("") != "Servicio temporal"].copy()
mb["AÑO"] = mb["AÑO"].fillna(mb.groupby("LINEA")["AÑO"].transform("min"))
est = pd.concat([mt[["LINEA", "AÑO", "geometry"]].assign(sistema="metro"),
                 mb[["LINEA", "AÑO", "geometry"]].assign(sistema="metrobus")], ignore_index=True)
est = gpd.GeoDataFrame(est, crs=mt.crs).to_crs(CRS)
est["AÑO"] = est["AÑO"].astype(int)
pts = gpd.GeoDataFrame(at[["cvegeo"]], geometry=gpd.points_from_xy(at.x, at.y), crs=CRS)
dist = {}
for y in ANIOS_DEL:
    abiertas = est[(est["AÑO"] <= y) & ~((est.sistema == "metro") & (est.LINEA == "12") & (y in (2021, 2022)))]
    j = gpd.sjoin_nearest(pts, abiertas[["geometry"]], distance_col="dist_m")
    j = j[~j.index.duplicated()]
    dist[f"dist_km_{y}"] = pd.Series(j.dist_m.values / 1000, index=j.cvegeo.values)
dist = pd.DataFrame(dist).reindex(at.cvegeo)
chk = pd.read_parquet(INTERIM / "ageb_dist_estacion.parquet").query("anio == 2019").set_index("cvegeo").dist_estacion_m / 1000
assert np.allclose(dist.dist_km_2019, chk.reindex(dist.index)), "la distancia 2019 no reproduce la de 01_datos.py"

# ============================================================ Censo 2020 (RESAGEBURB) y superficie no residencial
print("== Censo 2020")
COLS = ["ENTIDAD", "MUN", "LOC", "AGEB", "MZA", "POBTOT", "GRAPROES", "PEA", "P_12YMAS", "VPH_INTER", "TVIVPARHAB",
        "VIVPAR_DES", "VIVTOT"]
r = pd.read_csv(RAW / "censo2020/resageburb/RESAGEBURB_09CSV20.csv", dtype=str, usecols=COLS, encoding="utf-8-sig")
r["cvegeo"] = r.ENTIDAD.str.zfill(2) + r.MUN.str.zfill(3) + r.LOC.str.zfill(4) + r.AGEB
num = r[COLS[5:]].apply(pd.to_numeric, errors="coerce")  # '*' y 'N/D' (confidencialidad) → NaN
r = pd.concat([r[["cvegeo", "MZA", "AGEB"]], num], axis=1)
ra = r[(r.MZA == "000") & (r.AGEB != "0000")].set_index("cvegeo")
censo = pd.DataFrame({
    "escolaridad": ra.GRAPROES,
    "pct_pea": ra.PEA / ra.P_12YMAS,
    "pct_internet": ra.VPH_INTER / ra.TVIVPARHAB,
    "pct_deshabitadas": ra.VIVPAR_DES / ra.VIVTOT,
}).reindex(at.cvegeo)
# Supuesto: una manzana sin población residente (POBTOT = 0 o ausente del censo) es de uso no residencial
# (comercio, oficinas, equipamiento). % de la superficie de manzanas de la AGEB que no es residencial.
mz = lee_mg("09m")[["CVEGEO", "geometry"]]
mz["a"] = mz.area
mz["cvegeo"] = mz.CVEGEO.str[:13]
pob_mz = r[~r.MZA.isin(["000"])].assign(cvemz=lambda x: x.cvegeo + x.MZA).set_index("cvemz").POBTOT
mz["residencial"] = mz.CVEGEO.map(pob_mz).fillna(0) > 0
nr = mz.groupby("cvegeo").apply(lambda x: x.a[~x.residencial].sum() / x.a.sum(), include_groups=False)
censo["pct_no_residencial"] = nr.reindex(at.cvegeo)
print(censo.describe().round(3).T)

# ============================================================ Censos Económicos armonizados
print("== Censos Económicos")
CE = RAW / "censos_economicos"
VARS = ["UE", "A111A", "A131A", "H001A"]


def lee_ce(anio):
    if anio == 2019:
        c = pd.read_csv(CE / "ce2019_cdmx_csv/conjunto_de_datos/ce2019_cdmx.csv", dtype=str, encoding="utf-8-sig",
                        index_col=False, usecols=["MUNICIPIO", "CODIGO", "ID_ESTRATO", *VARS]).rename(columns={"MUNICIPIO": "mun"})
    else:  # 2024: entidad y municipio cambian de nombre (E03/E04)
        c = pd.read_csv(CE / "conjunto_de_datos_ce_cdmx_2024_csv/conjunto_de_datos/tr_ce_cdmx_2024.csv", dtype=str,
                        encoding="utf-8-sig", index_col=False, usecols=["E04", "CODIGO", "ID_ESTRATO", *VARS]).rename(columns={"E04": "mun"})
    c = c.apply(lambda s: s.str.strip())
    c["mun"] = c.mun.replace("", np.nan).fillna("CDMX")
    c["estrato"] = c.ID_ESTRATO.replace("", np.nan).fillna("total")
    c["CODIGO"] = c.CODIGO.replace({"": "todos", "TOTAL DE SECTOR": "todos"}).fillna("todos")  # total de la alcaldía
    c = c[c.CODIGO.isin(["todos", "11", "21", "22", "23", "31-33", "43", "46", "48-49", "51", "52", "53", "54", "55", "56", "61",
                         "62", "71", "72", "81"])]
    c[VARS] = c[VARS].apply(pd.to_numeric, errors="coerce")
    return c.rename(columns={"CODIGO": "sector"})[["mun", "sector", "estrato", *VARS]].assign(anio=anio)


ce = pd.concat([lee_ce(2019), lee_ce(2024)], ignore_index=True)
ce.to_parquet(INTERIM / "ce_alc_sector_estrato.parquet", index=False)
print(ce[(ce.mun == "CDMX") & (ce.estrato == "total") & (ce.sector != "todos")].pivot(index="sector", columns="anio", values="UE"))
# Subsectores (3 dígitos) presentes en cada CE: el DENUE se recorta a ellos en la prueba M
sub = {}
for anio, f, col in [(2019, "ce2019_cdmx_csv/conjunto_de_datos/ce2019_cdmx.csv", "CODIGO"),
                     (2024, "conjunto_de_datos_ce_cdmx_2024_csv/conjunto_de_datos/tr_ce_cdmx_2024.csv", "CODIGO")]:
    cod = pd.read_csv(CE / f, dtype=str, encoding="utf-8-sig", index_col=False, usecols=[col])[col].str.strip()
    sub[anio] = set(cod[cod.str.len() == 3])

# DENUE completo (todas las coordenadas, por la clave de municipio del propio DENUE) × sector, para la prueba M
filas = []
for corte, anio_ce in [(2019, 2019), (2024, 2024)]:
    e = pd.read_parquet(RAW / f"denue/parquet/denue_09_{corte}_11.parquet", columns=["codigo_act", "cve_mun"])
    e["codigo_act"] = e.codigo_act.astype(str).str.zfill(6)
    e["cve_mun"] = e.cve_mun.astype(str).str.zfill(3)
    n0 = len(e)
    e = e[~e.codigo_act.str.startswith("93") & e.codigo_act.str[:3].isin(sub[anio_ce])]
    print(f"DENUE {corte}-11: {n0:,} → {len(e):,} tras quitar SCIAN 93 y subsectores ausentes del CE {anio_ce}")
    filas.append(e.assign(sector=sector_ce(e.codigo_act)).groupby(["cve_mun", "sector"]).size().rename("N")
                 .reset_index().assign(corte=corte))
pd.concat(filas).to_parquet(INTERIM / "denue_alcaldia_sector.parquet", index=False)

# ============================================================ Imputación monetaria (descriptiva)
print("== Imputación monetaria")
at_mun = at.set_index("cvegeo").cve_mun


def medias(anio):
    """Valor medio por UE en cada celda; NaN si está suprimida (estrato 99 o valores en blanco)."""
    c = ce[ce.anio == anio]
    m = c.assign(pb=c.A111A / c.UE, va=c.A131A / c.UE)
    return m.set_index(["mun", "sector", "estrato"])[["pb", "va"]]


imp, cob, val = {}, [], []
for corte, anio_ce, anio_act in [(2019, 2019, 2018), (2024, 2024, 2023)]:
    m = medias(anio_ce)
    e = den[den.corte == corte].copy()
    e["mun"] = e.cvegeo.map(at_mun)
    e["sector"] = sector_ce(e.codigo_act)
    e["estrato"] = e.estrato_ce.astype(int).astype(str)
    e = e[~e.codigo_act.str.startswith("93")]
    nivel = pd.Series("sin valor", index=e.index)
    for c in ["pb", "va"]:
        e[c] = np.nan
    # Cascada del supuesto 4: alcaldía × sector × estrato → alcaldía × sector → CDMX × sector × estrato
    pasos = [("alcaldía × sector × estrato", lambda x: x),
             ("alcaldía × sector", lambda x: x.assign(estrato="total")),
             ("CDMX × sector × estrato", lambda x: x.assign(mun="CDMX"))]
    for nom, f in pasos:
        falta = e.pb.isna()
        k = f(e.loc[falta, ["mun", "sector", "estrato"]])
        v = m.reindex(pd.MultiIndex.from_frame(k[["mun", "sector", "estrato"]]))
        e.loc[falta, "pb"], e.loc[falta, "va"] = v.pb.values, v.va.values
        nivel[falta & e.pb.notna()] = nom
    f = INPC_2018_2023 if anio_act == 2018 else 1.0
    cob.append(nivel.value_counts(normalize=True).rename(anio_act))
    # Calibración (desviación del supuesto 4, registrada tras la validación): la imputación directa duplica el total
    # del CE porque el DENUE tiene más establecimientos de 251+ personas que el censo (2,174 vs 1,856 UE en 2019;
    # 1,905 vs 1,348 en 2024) y ese estrato tiene medias enormes. Se reparte el total del CE de cada alcaldía × sector
    # entre los establecimientos del DENUE, en proporción a su valor imputado. Si el total está suprimido, queda la
    # imputación directa.
    tot = ce[(ce.anio == anio_ce) & (ce.estrato == "total")].set_index(["mun", "sector"])[["A111A", "A131A"]]
    alc = tot.xs("todos", level="sector")
    tot = tot[tot.index.get_level_values("sector") != "todos"]
    s_cel = e.groupby(["mun", "sector"])[["pb", "va"]].transform("sum")
    k = pd.MultiIndex.from_frame(e[["mun", "sector"]])
    for c, cc in [("pb", "A111A"), ("va", "A131A")]:
        fac = tot[cc].reindex(k).values / s_cel[c].values
        e[f"{c}_directa"] = e[c]
        sup = ~np.isfinite(fac)
        e[c] = np.where(sup, e[c], e[c] * fac)
        # Segunda etapa: los sectores suprimidos de cada alcaldía se reparten el resto (total de la alcaldía menos los
        # sectores publicados), también en proporción a su valor imputado. El resto negativo se trunca en 0.
        resto = (alc[cc] - tot[cc].groupby(level="mun").sum()).clip(lower=0)
        s_sup = e[c].where(sup, 0).groupby(e.mun).transform("sum")
        fac2 = e.mun.map(resto).values / s_sup.values
        e[c] = np.where(sup & np.isfinite(fac2), e[c] * fac2, e[c])
    e[["pb", "va", "pb_directa", "va_directa"]] *= f  # millones de pesos de 2023
    imp[anio_act] = e.groupby("cvegeo")[["pb", "va"]].sum()
    # Validación: suma por alcaldía frente al total de la alcaldía en el CE
    t = alc[alc.index != "CDMX"] * f
    s = e.groupby("mun")[["pb", "va", "pb_directa", "va_directa"]].sum()
    val.append(pd.DataFrame({"anio": anio_act, "va_ce": t.A131A, "va_directa": s.va_directa, "va_calibrado": s.va,
                             "pb_ce": t.A111A, "pb_directa": s.pb_directa, "pb_calibrada": s.pb,
                             "N_denue_urbano": e.groupby("mun").size(),
                             "UE_ce": ce[(ce.anio == anio_ce) & (ce.estrato == "total") & (ce.sector == "todos")].set_index("mun").UE}))
cob = pd.concat(cob, axis=1).fillna(0)
val = pd.concat(val).rename_axis("cve_mun").reset_index()
val["razon_va_directa"] = val.va_directa / val.va_ce
val["razon_va_calibrado"] = val.va_calibrado / val.va_ce
val["razon_N"] = val.N_denue_urbano / val.UE_ce
cob.to_csv(OUT / "imputacion_cobertura.csv")
val.to_csv(OUT / "imputacion_validacion.csv", index=False)
print("Cobertura (% de establecimientos):\n", (cob * 100).round(1))
print(val.groupby("anio")[["razon_va_directa", "razon_va_calibrado", "razon_N"]].median().round(3))
mon = pd.DataFrame({"pb_2018": imp[2018].pb, "va_2018": imp[2018].va, "pb_2023": imp[2023].pb, "va_2023": imp[2023].va})
mon = mon.reindex(at.cvegeo).fillna(0)

# ============================================================ Panel
p = at[["cvegeo", "cve_mun", "alcaldia", "colonia", "POBTOT", "area_km2", "x", "y", "lon", "lat"]].rename(columns={"POBTOT": "pobtot"})
p = p.set_index("cvegeo").join([wide, rob, dist, censo, mon]).reset_index()
p["log_pob"] = np.log1p(p.pobtot)
p["no_residencial"] = (p.pobtot == 0).astype(int)
assert len(p) == len(at) and p.cvegeo.is_unique
p.to_parquet(PROCESSED / "panel_dinamica.parquet", index=False)
print(p.shape, "| AGEB con N_2018, N_2019 > 0:", int(((p.N_2018 > 0) & (p.N_2019 > 0)).sum()))
print("Totales N:", {c: int(p[f"N_{c}"].sum()) for c in CORTES})
print("listo →", PROCESSED / "panel_dinamica.parquet")
