"""T7 · Parte 0b: diagnósticos de V5 y V6, escritos DESPUÉS de ver que fallaron tal como estaban escritas.
No cambian el veredicto registrado; documentan por qué fallan y si es la regla o el esperado.

- V5 (desviación mínima): el estadístico "nº de AGEB con P(θ>1) > 0.9" es degenerado bajo el azar porque el nivel
  global de cada simulación (ΣO*/ΣcÊ) fluctúa alrededor de 1 y la contracción colapsa a ese nivel (a, b → ∞). Se
  aplica a cada simulación la misma regla de escala que a los datos reales: contracción con c*·cÊ, c* = ΣO*/ΣcÊ.
- V6 (diagnóstico): (i) ¿el Ê_2024 temporal de M2 ya contiene el nivel propio de cada AGEB? Se compara con un
  Ê_2024 fuera de muestra espacial (M2 entrenado con 2020–2023 de los otros folds); (ii) misma prueba con Ê_2024
  de M1 temporal (el escenario de la exploración previa).

Uso: .venv/bin/python analisis/lab/mecanismos/parte0b_diagnosticos.py   (~5 min)
"""
from t7_comun import md, RES, SEED, B, Log, guarda_json, temporal_mu, ic  # noqa: I001

md.lightgbm()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

from common import ROOT  # noqa: E402

log = Log("parte0b")
N_SIM = 500

# ================================================================ V5 con reescala por simulación
g2 = pd.read_parquet(ROOT / "data/processed/modelos/riesgo_periodo.parquet").set_index("cvegeo")
c = g2.O.sum() / g2.Ehat.sum()
cE = c * g2.Ehat.values
real = md.contraccion(g2.O.values, cE)
real_var, real_n = real["a"] / real["b"] ** 2, int((real["p_exc"] > 0.9).sum())
rng = np.random.default_rng(SEED)  # misma semilla: mismas O* que parte0
sims = []
for _ in range(N_SIM):
    o = rng.poisson(cE)
    cs = o.sum() / cE.sum()
    s = md.contraccion(o, cs * cE)
    sims.append({"c_sim": cs, "var_theta": s["a"] / s["b"] ** 2, "n_p09": int((s["p_exc"] > 0.9).sum()),
                 "media_prior": s["a"] / s["b"]})
sims = pd.DataFrame(sims)
sims.to_csv(RES / "parte0b_V5_simulaciones_reescaladas.csv", index=False)
p99v, p99n = float(np.percentile(sims.var_theta, 99)), float(np.percentile(sims.n_p09, 99))
V5b = real_var > p99v and real_n > p99n
log(f"V5 (reescala por simulación): real var {real_var:.4f}, n {real_n}; p99 sim var {p99v:.5f}, n {p99n:.1f}; "
    f"máx n {sims.n_p09.max()}; sims con n = 2419: {(sims.n_p09 == len(g2)).mean():.1%} → {'pasa' if V5b else 'NO pasa'}")
log(sims.describe().round(5).to_string())

# ================================================================ V6 · diagnósticos
p = md.datos(md.PRINCIPAL)
fold, bloque = p.cvegeo.map(md.bloques().fold).values, p.cvegeo.map(md.bloques().bloque).values
pre = p[p.anio <= 2023].reset_index(drop=True)
mu_pre, _ = md.cv_espacial(pre, md.cols_m2(pre), modelo="m2")
gpre = md.ranking_residual(pre, mu_pre)
c2, c1 = md.cols_m2(p), md.cols_m1(p)

# Ê_2024 fuera de muestra espacial: M2 entrenado con 2020–2023 de los otros folds, predice 2024 del fold
X, y = p[c2].values.astype(float), p.y.values
mu_oof24 = np.full(len(p), np.nan)
for f in range(5):
    tr = (fold != f) & (p.anio.values <= 2023)
    te = (fold == f) & (p.anio.values == 2024)
    mu_oof24[te] = md.ajusta_m2(X[tr], y[tr], bloque[tr]).predict(X[te])
es24 = (p.anio == 2024).values
mu_t2, _ = temporal_mu(p, c2, "m2", bloque)
mu_t1, _ = temporal_mu(p, c1, "nb", bloque)

q = p[es24][["cvegeo", "y", "E"]].reset_index(drop=True)
q["E24_M2_temporal"], q["E24_M2_espacial"], q["E24_M1_temporal"] = mu_t2, mu_oof24[es24], mu_t1
q = q.join(gpre[["O", "Ehat", "rr"]].rename(columns={"O": "O_pre", "Ehat": "E_pre", "rr": "rr_pre"}), on="cvegeo")
q["log_rr"] = np.log(q.rr_pre)

# (i) ¿el Ê_2024 temporal de M2 ya trae el exceso pasado? razón temporal/espacial vs rr_pre
rho_mem = float(stats.spearmanr(q.E24_M2_temporal / q.E24_M2_espacial, q.rr_pre)[0])
rho_mem1 = float(stats.spearmanr(q.E24_M1_temporal / q.E24_M2_espacial, q.rr_pre)[0])
log(f"ρ(Ê24 M2 temporal / Ê24 espacial, rr_pre) = {rho_mem:.3f}; con M1 temporal: {rho_mem1:.3f}")

b_ = q.cvegeo.map(md.bloques().bloque)
grupos = {k: g.index.values for k, g in q.groupby(b_)}
ids = np.array(list(grupos))


def pend(d, col):
    Xc = sm.add_constant(d.log_rr.values, has_constant="add")
    return sm.GLM(d.y.values, Xc, family=sm.families.Poisson(), offset=np.log(d[col].values)).fit().params[1]


filas = []
T = q.y.sum()
for col in ["E24_M2_temporal", "E24_M2_espacial", "E24_M1_temporal"]:
    rng = np.random.default_rng(SEED)
    bt = np.array([pend(q.iloc[np.concatenate([grupos[k] for k in rng.choice(ids, size=len(ids), replace=True)])], col)
                   for _ in range(B)])
    lo, hi = ic(bt)
    e = q[col] * T / q[col].sum()
    er = q[col] * q.rr_pre
    er = er * T / er.sum()
    dE, dR = md.devianza(q.y, e), md.devianza(q.y, er)
    filas.append({"Ê_2024": col, "pendiente": float(pend(q, col)), "ic_lo": lo, "ic_hi": hi, "dev_E": dE,
                  "dev_E_x_rr": dR, "mejora": (dE - dR) / dE,
                  "pasaria_V6": bool(lo > 0.5 and (dE - dR) / dE >= 0.10)})
    log(filas[-1])
d6 = pd.DataFrame(filas)
d6.to_csv(RES / "parte0b_V6_diagnostico.csv", index=False)
log(d6.round(4).to_string(index=False))
guarda_json({"parte0b": {"V5_reescalado": {"var_real": real_var, "n_real": real_n, "p99_var": p99v, "p99_n": p99n,
                                           "max_n": int(sims.n_p09.max()), "pasa": bool(V5b)},
                         "V6_memoria": {"rho_razon_M2": rho_mem, "rho_razon_M1": rho_mem1},
                         "V6_diagnostico": d6.to_dict("records")}}, "veredictos.json")
log("listo")
