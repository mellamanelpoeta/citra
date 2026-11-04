"""02 · Variables por AGEB × año → data/processed/panel_ageb_anio.parquet

Una fila por AGEB urbana (2,431) × año del hecho (2019–2024). Las variables del DENUE vienen en dos
variantes (experimento S2): sufijo `_nov` (corte de noviembre del mismo año) y `_fijo` (corte 2023-11
para todos los años). Diccionario de columnas: analisis/pipeline/README.md.

Uso: .venv/bin/python analisis/pipeline/02_features.py
"""
import numpy as np
import pandas as pd

from common import (ANIOS, CANDIDATOS_EFECTIVO, CORTE_FIJO, EFECTIVO, GRANDES, INTERIM, PER_OCU, PROCESSED,
                    SECTORES, clasifica)

at = pd.read_parquet(INTERIM / "ageb_atributos.parquet")
idx = pd.MultiIndex.from_product([at.cvegeo, ANIOS], names=["cvegeo", "anio"])

# ============================================================ variables objetivo
d = pd.read_parquet(INTERIM / "delitos_ageb.parquet")
d = d[d.tipo_asig == "URBANA"]  # restringir a AGEB urbanas cuesta <0.1% de los delitos (rural + sin asignar)
neg = d.objetivo == "negocio"
far = d.farderos == 1
DEFS = {
    "y_v1_negocio": neg,                                  # V1: robo a negocio agregado (todos los subtipos)
    "y_v2_sin_farderos": neg & ~far,                      # V2: sin farderos (solo comparable desde 2020)
    "y_v2b": neg & ~far & (d.grupo != "negocio_sv_dentro"),  # V2b: V2 sin "S/V dentro de negocios" (variante del EDA)
    "y_v3_via_publica": d.objetivo == "via_publica",      # V3: transeúnte en vía pública + repartidor + cuentahabiente
    "y_farderos": neg & far,
    "y_negocio_sv_dentro": d.grupo == "negocio_sv_dentro",  # "S/V dentro de negocios" (aparece en 2020), dentro de V1 y V2
    "y_negocio_cv": d.grupo.str.startswith("negocio_cv"),
    "y_ampliado": d.objetivo == "ampliado",               # transeúnte dentro de negocio/restaurante/parques y mercados
}
y = pd.DataFrame({k: d[m].groupby(["cvegeo", "anio_hecho"]).size().rename_axis(["cvegeo", "anio"]).reindex(idx, fill_value=0)
                  for k, m in DEFS.items()})

# ============================================================ DENUE por corte
e = pd.read_parquet(INTERIM / "denue_ageb.parquet")
e["cand"] = clasifica(e.codigo_act, CANDIDATOS_EFECTIVO, resto="")
grupo_ef = e.cand.map({c: g for g, cs in EFECTIVO.items() for c in cs})
e["sector"] = grupo_ef.fillna(clasifica(e.codigo_act, SECTORES))
e["grande"] = e.per_ocu.isin(GRANDES)
e["po_est"] = e.per_ocu.map(PER_OCU)
assert e.po_est.notna().all(), e.per_ocu[e.po_est.isna()].unique()
# Autoservicio grande: supermercados/minisúpers (SCIAN 4621xx) con ≥51 personas. Proxy de las tiendas que
# concentran farderos y denuncian sistemáticamente (ENVE: las grandes tienen menor cifra oculta).
e["autoserv_grande"] = e.codigo_act.str.startswith("4621") & e.grande
g = e.groupby(["corte", "cvegeo"])
f = pd.DataFrame({
    "E": g.size(),
    "po_est": g.po_est.sum(),
    "n_grandes": g.grande.sum(),
    "n_autoserv_grande": g.autoserv_grande.sum(),
})
f["p_grandes"] = f.n_grandes / f.E
for c in CANDIDATOS_EFECTIVO:
    f[f"pc_{c}"] = e[e.cand == c].groupby(["corte", "cvegeo"]).size().reindex(f.index, fill_value=0) / f.E
for s in [*EFECTIVO, *SECTORES, "otros"]:
    f[f"ps_{s}"] = e[e.sector == s].groupby(["corte", "cvegeo"]).size().reindex(f.index, fill_value=0) / f.E
f = f.reset_index()

base = f[f.corte.isin(ANIOS)].rename(columns={"corte": "anio"}).set_index(["cvegeo", "anio"])
nov = base.reindex(idx).add_suffix("_nov")
fijo = (f[f.corte == CORTE_FIJO].drop(columns="corte").set_index("cvegeo")
        .reindex(idx.get_level_values("cvegeo")).set_axis(idx).add_suffix("_fijo"))
denue = pd.concat([nov, fijo], axis=1)
# AGEB sin establecimientos: E = 0 y proporciones indefinidas (NaN). Se excluyen al modelar (supuesto declarado).
for v in ("nov", "fijo"):
    for c in ["E", "po_est", "n_grandes", "n_autoserv_grande"]:
        denue[f"{c}_{v}"] = denue[f"{c}_{v}"].fillna(0).astype(int)

# ============================================================ controles y etiquetas
dist = pd.read_parquet(INTERIM / "ageb_dist_estacion.parquet").set_index(["cvegeo", "anio"]).reindex(idx)
p = pd.concat([y, denue, dist], axis=1).reset_index()
p = p.merge(at[["cvegeo", "cve_mun", "alcaldia", "cve_col", "colonia", "POBTOT", "area_km2", "x", "y", "lon", "lat"]],
            on="cvegeo", how="left")
p = p.rename(columns={"POBTOT": "pobtot"})
p["log_pob"] = np.log1p(p.pobtot)
# Supuesto: POBTOT = 0 identifica AGEB no residenciales (comerciales o de equipamiento; 17 AGEB).
p["no_residencial"] = (p.pobtot == 0).astype(int)
p["dist_estacion_km"] = p.dist_estacion_m / 1000
p = p.drop(columns="dist_estacion_m")

assert len(p) == len(at) * len(ANIOS) and not p.duplicated(["cvegeo", "anio"]).any()
p.to_parquet(PROCESSED / "panel_ageb_anio.parquet", index=False)
print(p.shape)
print(p.groupby("anio")[["y_v1_negocio", "y_v2_sin_farderos", "y_v3_via_publica", "y_farderos", "E_nov", "E_fijo"]].sum())
print("AGEB con E_fijo = 0:", int((p[p.anio == 2023].E_fijo == 0).sum()),
      "| AGEB-año con E_nov = 0:", p[p.E_nov == 0].groupby("anio").size().to_dict())
