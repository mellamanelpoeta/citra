"""T7 · Parte 0: ¿Ê (M2) es una vara válida? Reglas V1–V6 (protocolo: analisis/lab/mecanismos/README.md).

Uso: .venv/bin/python analisis/lab/mecanismos/parte0_validez.py   (~10 min)
"""
from t7_comun import md, RES, SEED, B, Log, guarda_json, temporal_mu, ic  # noqa: I001

md.lightgbm()  # en macOS puede relanzar el proceso: va primero

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ROOT  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
log = Log("parte0")
N_SIM = 500


def dev_total(y, mu):
    return md.devianza(y, mu) * len(y)


# ================================================================ Ê fuera de muestra (M0, M1, M2)
p = md.datos(md.PRINCIPAL)
fold, bloque = p.cvegeo.map(md.bloques().fold).values, p.cvegeo.map(md.bloques().bloque).values
cols = {"M0": md.cols_m0(p), "M1": md.cols_m1(p), "M2": md.cols_m2(p)}
oof = {}
for m in cols:
    oof[m] = md.cv_espacial(p, cols[m], modelo="nb" if m != "M2" else "m2")
    log(m, "devianza espacial media por fold:", round(np.mean(list(oof[m][1].values())), 4))

# verificación contra el Ê guardado por 04_modelos.py (solo lectura)
guardado = pd.read_parquet(ROOT / "data/processed/modelos/riesgo_periodo.parquet").set_index("cvegeo")
g2 = md.ranking_residual(p, oof["M2"][0])
dif = float(np.max(np.abs(g2.Ehat.reindex(guardado.index) - guardado.Ehat) / guardado.Ehat))
log(f"Máx. diferencia relativa Ê recalculado vs guardado: {dif:.2e}")
if dif > 1e-6:
    log("AVISO: Ê no coincide con el guardado; se usa el guardado para V2–V5 (aclaración 1)")
    g2 = guardado[["O", "Ehat", "rr", "p_exc", "score"]].copy()
else:
    g2 = guardado[["O", "Ehat", "rr", "p_exc", "score"]].copy()  # idéntico; se usa el guardado por trazabilidad

# ================================================================ V1 · devianza explicada
filas = []
nulo_anio = p.groupby("anio").y.transform("mean").values
for m, (mu, _) in oof.items():
    d_ay = 1 - dev_total(p.y, mu) / dev_total(p.y, nulo_anio)
    per = p.assign(mu=mu).groupby("cvegeo").agg(O=("y", "sum"), E=("mu", "sum"))
    d_per = 1 - dev_total(per.O, per.E) / dev_total(per.O, np.full(len(per), per.O.mean()))
    filas.append({"modelo": m, "dev_expl_ageb_anio": d_ay, "dev_expl_periodo": d_per,
                  "dev_espacial_media_fold": float(np.mean(list(oof[m][1].values())))})
v1 = pd.DataFrame(filas)
log("V1\n", v1.round(4).to_string(index=False))
v1.to_csv(RES / "parte0_V1_devianza.csv", index=False)

# ================================================================ V2 · calibración por decil de Ê
g2["decil"] = pd.qcut(g2.Ehat, 10, labels=False) + 1
v2 = g2.groupby("decil").agg(n=("O", "size"), O=("O", "sum"), Ehat=("Ehat", "sum"),
                             Ehat_min=("Ehat", "min"), Ehat_max=("Ehat", "max"))
v2["O_sobre_E"] = v2.O / v2.Ehat
v2.to_csv(RES / "parte0_V2_calibracion.csv")
V2 = bool(v2.O_sobre_E.between(0.85, 1.15).all())
log("V2\n", v2.round(3).to_string(), "\n→ pasa" if V2 else "\n→ NO pasa")

# ================================================================ V3 · rr vs Ê
rho3 = float(stats.spearmanr(g2.rr, g2.Ehat)[0])
V3 = abs(rho3) < 0.10
log(f"V3 ρ(rr, Ê) = {rho3:.4f} → {'pasa' if V3 else 'NO pasa'}")

# ================================================================ V4 · M1 vs M2
g1 = md.ranking_residual(p, oof["M1"][0])
jac4 = md.jaccard(md.top(g1.score), md.top(g2.score))
rho4 = float(stats.spearmanr(g1.rr, g2.rr.reindex(g1.index))[0])
V4 = jac4 >= 0.50
log(f"V4 Jaccard top 10% M1 vs M2 = {jac4:.3f}; ρ(rr) = {rho4:.3f} → {'pasa' if V4 else 'NO pasa'}")

# ================================================================ V5 · heterogeneidad vs azar
c = g2.O.sum() / g2.Ehat.sum()
cE = c * g2.Ehat.values
real = md.contraccion(g2.O.values, cE)
real_var, real_n = real["a"] / real["b"] ** 2, int((real["p_exc"] > 0.9).sum())
sin_esc = md.contraccion(g2.O.values, g2.Ehat.values)
rng = np.random.default_rng(SEED)
sims = []
for k in range(N_SIM):
    o = rng.poisson(cE)
    s = md.contraccion(o, cE)
    sims.append({"var_theta": s["a"] / s["b"] ** 2, "n_p09": int((s["p_exc"] > 0.9).sum()), "a": s["a"], "b": s["b"]})
sims = pd.DataFrame(sims)
sims.to_csv(RES / "parte0_V5_simulaciones.csv", index=False)
p99v, p99n = float(np.percentile(sims.var_theta, 99)), float(np.percentile(sims.n_p09, 99))
V5 = real_var > p99v and real_n > p99n
log(f"V5 c = {c:.4f}; real var(θ) = {real_var:.4f} (sin reescalar {sin_esc['a'] / sin_esc['b'] ** 2:.4f}), "
    f"n P>0.9 = {real_n} (sin reescalar {(sin_esc['p_exc'] > 0.9).sum()}); p99 sim: var {p99v:.5f}, n {p99n:.1f}; "
    f"máx sim: var {sims.var_theta.max():.5f}, n {sims.n_p09.max()} → {'pasa' if V5 else 'NO pasa'}")

# ================================================================ V6 · validez predictiva 2020–2023 → 2024
pre = p[p.anio <= 2023].reset_index(drop=True)
c2_pre = md.cols_m2(pre)
mu_pre, _ = md.cv_espacial(pre, c2_pre, modelo="m2")
gpre = md.ranking_residual(pre, mu_pre)
mu24, met24 = temporal_mu(p, cols["M2"], "m2", bloque)
log("Ê_2024 temporal (M2):", {k: round(v, 4) for k, v in met24.items()})
q = p[p.anio == 2024][["cvegeo", "y", "E"]].reset_index(drop=True)
q["E24"] = mu24
q = q.join(gpre[["O", "Ehat", "rr"]].rename(columns={"O": "O_pre", "Ehat": "E_pre", "rr": "rr_pre"}), on="cvegeo")
q["log_rr"] = np.log(q.rr_pre)
assert q.notna().all().all()


def glm_pend(d):
    X = sm.add_constant(d.log_rr.values, has_constant="add")
    m = sm.GLM(d.y.values, X, family=sm.families.Poisson(), offset=np.log(d.E24.values)).fit()
    return m.params


par = glm_pend(q)
b_ = q.cvegeo.map(md.bloques().bloque)
grupos = {k: g.index.values for k, g in q.groupby(b_)}
ids = np.array(list(grupos))
rng = np.random.default_rng(SEED)
boot = []
for _ in range(B):
    i = np.concatenate([grupos[k] for k in rng.choice(ids, size=len(ids), replace=True)])
    boot.append(glm_pend(q.iloc[i])[1])
boot = np.array(boot)
lo6, hi6 = ic(boot)
# pendiente con rr crudo (complemento)
qc = q[q.O_pre > 0].assign(log_rr=lambda d: np.log(d.O_pre / d.E_pre))
pend_crudo = float(glm_pend(qc)[1])

T = q.y.sum()
pred = {"∝ E": q.E, "conteo 2020–23": q.O_pre, "Ê_2024": q.E24, "Ê_2024 × rr crudo": q.E24 * q.O_pre / q.E_pre,
        "Ê_2024 × rr contraído": q.E24 * q.rr_pre}
filas = []
for k, v in pred.items():
    v = np.asarray(v, float) * T / np.sum(v)
    cero = (v == 0) & (q.y.values > 0)
    pos = v > 0
    filas.append({"predictor": k, "devianza": np.inf if cero.any() else md.devianza(q.y, v),
                  "n_cero_con_robo": int(cero.sum()),
                  "devianza_solo_pred_pos": md.devianza(q.y.values[pos], v[pos]), "n_pred_pos": int(pos.sum())})
v6 = pd.DataFrame(filas)
d_E, d_rr = v6.set_index("predictor").devianza[["Ê_2024", "Ê_2024 × rr contraído"]]
mejora6 = (d_E - d_rr) / d_E
V6 = lo6 > 0.5 and mejora6 >= 0.10
v6.to_csv(RES / "parte0_V6_devianza_2024.csv", index=False)
log("V6\n", v6.round(4).to_string(index=False))
log(f"V6 pendiente log rr_pre = {par[1]:.3f} [{lo6:.3f}, {hi6:.3f}] (crudo, O_pre>0: {pend_crudo:.3f}); "
    f"mejora devianza Ê×rr vs Ê = {mejora6:.1%} → {'pasa' if V6 else 'NO pasa'}")

veredicto = all([V2, V3, V4, V5, V6])
log("Veredicto parte 0:", "Ê válido" if veredicto else "Ê NO válido en al menos una regla")
guarda_json({"parte0": {
    "V1": v1.to_dict("records"), "dif_rel_Ehat_guardado": dif,
    "V2": {"min": float(v2.O_sobre_E.min()), "max": float(v2.O_sobre_E.max()), "pasa": V2},
    "V3": {"rho": rho3, "pasa": bool(V3)},
    "V4": {"jaccard": jac4, "rho_rr": rho4, "pasa": bool(V4)},
    "V5": {"c": c, "var_real": real_var, "n_real": real_n, "var_real_sin_reescalar": sin_esc["a"] / sin_esc["b"] ** 2,
           "n_real_sin_reescalar": int((sin_esc["p_exc"] > 0.9).sum()), "p99_var": p99v, "p99_n": p99n,
           "max_var": float(sims.var_theta.max()), "max_n": int(sims.n_p09.max()), "pasa": bool(V5)},
    "V6": {"pendiente": float(par[1]), "ic": [lo6, hi6], "pendiente_crudo_Opos": pend_crudo, "mejora_dev": float(mejora6),
           "dev_2024": v6.to_dict("records"), "temporal_M2": met24, "pasa": bool(V6)},
    "valido": bool(veredicto)}}, "veredictos.json")

# ================================================================ figuras
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
axs[0].plot(v2.index, v2.O_sobre_E, "o-", color="#1f4e79")
axs[0].axhspan(0.85, 1.15, color="#2ecc71", alpha=0.12)
axs[0].axhline(1, color="#333", lw=0.8)
axs[0].set_xlabel("Decil de Ê del periodo")
axs[0].set_ylabel("O / Ê")
axs[0].set_title("V2 · Calibración (banda: [0.85, 1.15])", loc="left", fontsize=9)
axs[1].hist(sims.var_theta, bins=30, color="#9fb6cd")
axs[1].axvline(real_var, color="#c0392b")
axs[1].set_xscale("symlog", linthresh=1e-4)
axs[1].set_xlabel("var(θ) = a/b²")
axs[1].set_title(f"V5 · Real {real_var:.2f} vs 500 simulaciones (p99 {p99v:.4f})", loc="left", fontsize=9)
o = v6.set_index("predictor")
y_ = o.devianza.replace(np.inf, np.nan)
axs[2].barh(o.index, o.devianza_solo_pred_pos, color="#ddd", label="solo predicción > 0")
axs[2].barh(o.index, y_, color="#1f4e79", label="todas las AGEB")
axs[2].set_xlabel("Devianza Poisson media, 2024")
axs[2].set_title(f"V6 · Pendiente {par[1]:.2f} [{lo6:.2f}, {hi6:.2f}]; mejora {mejora6:.0%}", loc="left", fontsize=9)
axs[2].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(RES / "fig_T7_parte0_validez.png")
log("listo")
