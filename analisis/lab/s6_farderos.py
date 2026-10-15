"""S6 · ¿Con control de autoservicios conviene volver a meter farderos y "S/V dentro"? (regla en analisis/lab/README.md §S6)

Variables V2b / V1 / V1c (V1 + clientes dentro de negocio, restaurante, hotel y cine) × controles (0) actuales /
(1) + conteos de cadenas del DENUE. Métricas: % de autoservicio grande en el top 10% residual, ρ prospectivo
2020–2022 → 2023–2024 y el mismo ρ con la variante adelgazada al volumen de V2b.

Uso: .venv/bin/python analisis/lab/s6_farderos.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common import CORTE_FIJO, GRANDES, INTERIM  # noqa: E402
from modelado import (SEED, ajusta_nb, cols_m0, cols_m1, cols_m2, contraccion, cv_espacial, datos,  # noqa: E402
                      jaccard, ranking_residual, top)

RES = Path(__file__).resolve().parent / "resultados"
RES.mkdir(exist_ok=True)
JARDINES = "0901000011260"
ANTES, DESPUES = [2020, 2021, 2022], [2023, 2024]
B_THIN = 50

# ------------------------------------------------------------ variable V1c: clientes dentro de negocios
d = pd.read_parquet(INTERIM / "delitos_ageb.parquet")
d = d[(d.tipo_asig == "URBANA") & d.anio_hecho.between(2020, 2024)]
LUGARES = ("EN NEGOCIO", "EN RESTAURANT", "EN HOTEL", "EN CINE")
cli = d[(d.grupo == "transeunte_en_lugar") & d.delito_norm.str.contains("|".join(LUGARES))]
clientes = cli.groupby(["cvegeo", "anio_hecho"]).size().rename_axis(["cvegeo", "anio"]).rename("clientes")

# ------------------------------------------------------------ controles (1): cadenas del DENUE 2023-11
e = pd.read_parquet(INTERIM / "denue_ageb.parquet")
e = e[e.corte == CORTE_FIJO]
grande = e.per_ocu.isin(GRANDES)
CADENAS = {
    "c_autoserv": e.codigo_act.str.startswith("4621"),
    "c_autoserv_grande": e.codigo_act.str.startswith("4621") & grande,
    "c_departamental": e.codigo_act.str.startswith("4622"),
    "c_conveniencia": e.codigo_act == "462112",
}
cadenas = pd.DataFrame({k: np.log1p(e[m].groupby(e.cvegeo).size()) for k, m in CADENAS.items()})


def panel(var):
    p = datos(y="V1" if var != "V2b" else "V2b", anios=list(range(2020, 2025)))
    if var == "V1c":
        p["y"] += p.set_index(["cvegeo", "anio"]).index.map(clientes).fillna(0).values
    p = p.join(cadenas, on="cvegeo")
    p[list(CADENAS)] = p[list(CADENAS)].fillna(0)
    return p


def rr_sub(p, mu, anios, O=None):
    m = p.anio.isin(anios).values
    O = p.y.values if O is None else O
    g = pd.DataFrame({"cvegeo": p.cvegeo.values[m], "O": O[m], "E": mu[m]}).groupby("cvegeo").sum()
    s = contraccion(g.O, g.E)
    return pd.DataFrame({"rr": s["rr"], "p_exc": s["p_exc"]}, index=g.index)


def prospectivo(p, mu, O=None):
    a, b = rr_sub(p, mu, ANTES, O), rr_sub(p, mu, DESPUES, O)
    rho = stats.spearmanr(a.rr, b.rr.reindex(a.index))[0]
    t = top(a.p_exc + 1e-6 * a.rr)
    sigue = float((b.p_exc.reindex(list(t)) > 0.5).mean())
    return rho, sigue


ag = panel("V2b").groupby("cvegeo").n_autoserv_grande_fijo.first() > 0
n_v2b = panel("V2b").y.sum()
filas, tops = [], {}
for var in ["V2b", "V1", "V1c"]:
    p = panel(var)
    for ctl in [0, 1]:
        c2 = cols_m2(p) + (list(CADENAS) if ctl else [])
        mu, dev = cv_espacial(p, c2, modelo="m2")
        g = ranking_residual(p, mu)
        t = top(g.score)
        tops[(var, ctl)] = t
        rho, sigue = prospectivo(p, mu)
        # control de volumen: thinning binomial al total de V2b
        q = min(1.0, n_v2b / p.y.sum())
        rng = np.random.default_rng(SEED)
        rho_thin = [prospectivo(p, mu * q, rng.binomial(p.y.values.astype(int), q).astype(float))[0]
                    for _ in range(B_THIN)] if q < 1 else [rho]
        # M1 (NB) solo para comparar con S1
        c1 = cols_m1(p) + (list(CADENAS) if ctl else [])
        mu1, _ = cv_espacial(p, c1)
        t1 = top(ranking_residual(p, mu1).score)
        m0 = ajusta_nb(p[cols_m0(p)].values, p.y.values)
        orden = g.score.rank(ascending=False, method="first")
        fila = {"variable": var, "control": ctl, "carpetas": int(p.y.sum()),
                "pct_autoserv_top10_M2": 100 * ag.reindex(list(t)).mean(),
                "pct_autoserv_top10_M1": 100 * ag.reindex(list(t1)).mean(),
                "rho_prospectivo": rho, "pct_top_sigue_2023_24": 100 * sigue,
                "rho_adelgazado_mediana": float(np.median(rho_thin)),
                "rho_adelgazado_p05": float(np.percentile(rho_thin, 5)),
                "dev_M2": float(np.mean(list(dev.values()))),
                "beta_M0": float(m0.params[1]),
                "jaccard_vs_V2b0": jaccard(t, tops[("V2b", 0)]),
                "rank_jardines": int(orden[JARDINES]), "rr_jardines": float(g.rr[JARDINES])}
        filas.append(fila)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in fila.items()}, flush=True)

r = pd.DataFrame(filas)
r.to_csv(RES / "s6_farderos.csv", index=False)

# ------------------------------------------------------------ regla (ex ante, README §S6)
base = r[(r.variable == "V2b") & (r.control == 0)].iloc[0]
cand = r[(r.variable != "V2b") & (r.control == 1)].copy()
cand["i_validez"] = cand.pct_autoserv_top10_M2 <= base.pct_autoserv_top10_M2 + 3
cand["ii_estabilidad"] = cand.rho_prospectivo >= base.rho_prospectivo + 0.03
cand["iii_no_es_volumen"] = cand.rho_adelgazado_mediana >= base.rho_prospectivo
cand["pasa"] = cand.i_validez & cand.ii_estabilidad & cand.iii_no_es_volumen
pasan = cand[cand.pasa]
if pasan.empty:
    rec = "Mantener V2b como variable principal."
elif len(pasan) == 2:
    v1, v1c = pasan.set_index("variable").loc["V1"], pasan.set_index("variable").loc["V1c"]
    mejor_c = (v1c.pct_autoserv_top10_M2 < v1.pct_autoserv_top10_M2) and (v1c.rho_prospectivo > v1.rho_prospectivo)
    rec = f"Cambiar a {'V1c' if mejor_c else 'V1'} con control de cadenas."
else:
    rec = f"Cambiar a {pasan.variable.iloc[0]} con control de cadenas."
print("\nBase V2b(0):", base[["pct_autoserv_top10_M2", "rho_prospectivo"]].to_dict())
print(cand[["variable", "i_validez", "ii_estabilidad", "iii_no_es_volumen", "pasa"]].to_string(index=False))
print("Recomendación (regla S6):", rec)
