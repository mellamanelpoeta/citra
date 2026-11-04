"""T7 · Parte 5: curva de ganancia 2024 y ranking por cuantil inferior de θ. Hecha el 2026-10-14,
POSTERIOR a los veredictos y SIN regla ex ante: es descriptiva, no cambia veredictos, el Ê, la app ni el contrato.

1. Curva de ganancia 2024. Cada orden se arma solo con 2020–2023 (Ê fuera de muestra espacial). En 2024 se usa el
   Ê_2024 fuera de muestra en espacio y tiempo (M2 entrenado con 2020–2023 de los otros folds: el de parte0b, el que
   no "vio" a la AGEB). Ê_2024 se reescala a ΣO_2024 (c = ΣO/ΣÊ) para que el conjunto de la ciudad valga 1 y el
   lift no dependa de haber acertado el nivel global de 2024.
   - Lift(k) = ΣO_2024 / Σ c·Ê_2024 en el top k% de cada orden: cuántas veces su esperado tuvieron en 2024.
   - Exceso(k) = Σ(O_2024 − c·Ê_2024) en el top k%: robos de 2024 por encima de lo esperado.
   - Órdenes: P(θ>1) (el de la app), media posterior rr, cuantil 5% de θ, conteo, O/Ê crudo y P(θ>1) con 2023 solo.
   - Referencia de azar: 1000 listas al azar del mismo tamaño. IC 95% del lift en el top 10% por bootstrap de bloques
     espaciales con las listas fijas (incertidumbre de 2024 dada la lista).
2. Ranking por cuantil 5% de la posterior de θ ("exceso mínimo creíble") frente al de P(θ>1), con el periodo
   2020–2024 de la app: Jaccard, zonas que entran y salen, tamaño de las zonas y si C se sigue confirmando.

Uso: .venv/bin/python analisis/lab/mecanismos/parte5_ganancia_cuantil.py   (~5 min)
Salidas: resultados/parte5_*.csv, fig_T7_parte5_ganancia.png, log_parte5.txt, veredictos_parte5.json
"""
from t7_comun import md, RES, SEED, B, Log, guarda_json, ic  # noqa: I001

md.lightgbm()

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ROOT  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
log = Log("parte5")
Q = 0.05  # cuantil inferior de θ


def cuantil(O, E, a, b, q=Q):
    return stats.gamma.ppf(q, a + np.asarray(O, float), scale=1 / (b + np.asarray(E, float)))


def con_cuantil(g):
    """Añade el cuantil q de θ con la misma previa (a, b) que usa la contracción."""
    s = md.contraccion(g.O, g.Ehat)
    assert np.allclose(s["rr"], g.rr), "la previa recalculada no reproduce el rr guardado"
    return g.assign(q05=cuantil(g.O, g.Ehat, s["a"], s["b"])), s["a"], s["b"]


# ================================================================ insumos: órdenes con 2020–2023, Ê_2024 sin ver la AGEB
p = md.datos(md.PRINCIPAL)
fold, bloque = p.cvegeo.map(md.bloques().fold).values, p.cvegeo.map(md.bloques().bloque).values
pre_m = (p.anio <= 2023).values
pre = p[pre_m].reset_index(drop=True)
mu_pre, _ = md.cv_espacial(pre, md.cols_m2(pre), modelo="m2")
gpre, _, _ = con_cuantil(md.ranking_residual(pre, mu_pre))

a23 = pre.assign(mu=mu_pre)[pre.anio == 2023]
s23 = md.contraccion(a23.y, a23.mu)
p23 = pd.Series(s23["p_exc"] + 1e-6 * s23["rr"], index=a23.cvegeo.values)

c2 = md.cols_m2(p)
X, y = p[c2].values.astype(float), p.y.values
mu24 = np.full(len(p), np.nan)
for f in range(5):
    tr, te = (fold != f) & pre_m, (fold == f) & (p.anio.values == 2024)
    mu24[te] = md.ajusta_m2(X[tr], y[tr], bloque[tr]).predict(X[te])
es24 = (p.anio == 2024).values
d24 = pd.DataFrame({"y": y[es24], "mu": mu24[es24]}, index=p.cvegeo.values[es24])
c = d24.y.sum() / d24.mu.sum()
d24["cE"] = c * d24.mu
log(f"Ê_2024 fuera de muestra: ΣO = {d24.y.sum():.0f}, ΣÊ = {d24.mu.sum():.1f}, c = {c:.3f}")

ordenes = {"P(θ>1) 2020–23 (app)": gpre.score, "rr (media posterior)": gpre.rr, f"cuantil {Q:.0%} de θ": gpre.q05,
           "Conteo 2020–23": gpre.O, "O/Ê crudo": gpre.O / gpre.Ehat, "P(θ>1) solo 2023": p23}
ordenes = {k: v.reindex(d24.index) for k, v in ordenes.items()}

# ================================================================ 1. curva de ganancia
ks = np.arange(1, 51) / 100
n = len(d24)
O24, cE24 = d24.y.values, d24.cE.values


def lift_exceso(idx):
    return O24[idx].sum() / cE24[idx].sum(), (O24[idx] - cE24[idx]).sum()


filas = []
for nom, s in ordenes.items():
    orden = np.argsort(-s.values, kind="stable")
    for k in ks:
        L, X_ = lift_exceso(orden[: int(round(n * k))])
        filas.append({"orden": nom, "k": k, "lift": L, "exceso": X_})
rng = np.random.default_rng(SEED)
for k in ks:
    m = int(round(n * k))
    sims = np.array([lift_exceso(rng.choice(n, m, replace=False)) for _ in range(B)])
    filas.append({"orden": "Azar (mediana)", "k": k, "lift": np.median(sims[:, 0]), "exceso": np.median(sims[:, 1]),
                  "lift_p2.5": np.percentile(sims[:, 0], 2.5), "lift_p97.5": np.percentile(sims[:, 0], 97.5),
                  "exceso_p2.5": np.percentile(sims[:, 1], 2.5), "exceso_p97.5": np.percentile(sims[:, 1], 97.5)})
gan = pd.DataFrame(filas)
gan.to_csv(RES / "parte5_ganancia.csv", index=False)

# top 10%: tabla con IC por bootstrap de bloques (listas fijas, se remuestrean bloques de AGEB en 2024)
m10 = int(round(n * 0.10))
listas = {nom: set(d24.index[np.argsort(-s.values, kind="stable")[:m10]]) for nom, s in ordenes.items()}
blq = pd.Series(d24.index.map(md.bloques().bloque), index=d24.index)
grupos = {b_: np.flatnonzero(blq.values == b_) for b_ in blq.unique()}
ids = np.array(list(grupos))
muestras = [np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)]) for _ in range(B)]
en = {nom: d24.index.isin(L) for nom, L in listas.items()}
ref = en["Conteo 2020–23"]
t10 = []
for nom, m in en.items():
    bt = np.array([[O24[i][m[i]].sum() / cE24[i][m[i]].sum(),
                    O24[i][m[i]].sum() / cE24[i][m[i]].sum() - O24[i][ref[i]].sum() / cE24[i][ref[i]].sum()]
                   for i in muestras])
    L, X_ = lift_exceso(np.flatnonzero(m))
    t10.append({"orden": nom, "n": int(m.sum()), "lift": L, "lift_ic": ic(bt[:, 0]),
                "lift_menos_conteo_ic": ic(bt[:, 1]), "exceso_robos": X_,
                "pct_robo_2024": O24[m].sum() / O24.sum(), "pct_O_mayor_E": float((O24[m] > cE24[m]).mean()),
                "jaccard_vs_app": md.jaccard(listas["P(θ>1) 2020–23 (app)"], listas[nom]),
                "mediana_E": float(np.median(p[es24].set_index("cvegeo").E.reindex(list(listas[nom]))))})
t10 = pd.DataFrame(t10)
az = gan[(gan.orden == "Azar (mediana)") & np.isclose(gan.k, 0.10)].iloc[0]
log(f"\nTop 10% ({m10} AGEB) · azar: lift {az.lift:.3f} [{az['lift_p2.5']:.3f}, {az['lift_p97.5']:.3f}], "
    f"exceso {az.exceso:.0f} [{az['exceso_p2.5']:.0f}, {az['exceso_p97.5']:.0f}]")
log(t10.round(3).to_string(index=False))
t10.to_csv(RES / "parte5_top10_2024.csv", index=False)

# ================================================================ 2. cuantil 5% de θ vs P(θ>1), periodo de la app
g = pd.read_parquet(ROOT / "data/processed/modelos/riesgo_periodo.parquet").set_index("cvegeo")
g, a, b = con_cuantil(g)
E = p[p.anio == 2023].set_index("cvegeo").E.reindex(g.index)
Tapp, Tq, Tc = md.top(g.score), md.top(g.q05), md.top(g.O)
inv = Tapp - Tc
comp = []
for nom, T in [("P(θ>1) (app)", Tapp), (f"cuantil {Q:.0%} de θ", Tq), ("solo en P(θ>1)", Tapp - Tq),
               (f"solo en cuantil {Q:.0%}", Tq - Tapp), ("Ciudad", set(g.index))]:
    x = g.reindex(list(T))
    comp.append({"lista": nom, "n": len(T), "mediana_E": float(E.reindex(list(T)).median()),
                 "mediana_Ehat": float(x.Ehat.median()), "mediana_O": float(x.O.median()),
                 "mediana_rr": float(x.rr.median()), "min_rr": float(x.rr.min()),
                 "mediana_p_exc": float(x.p_exc.median()), "mediana_q05": float(x.q05.median())})
comp = pd.DataFrame(comp)
comp.to_csv(RES / "parte5_cuantil_vs_pexc.csv", index=False)
res2 = {"a": a, "b": b, "jaccard_app_vs_q05": md.jaccard(Tapp, Tq), "comunes": len(Tapp & Tq),
        "rho_score_q05": float(stats.spearmanr(g.score, g.q05)[0]),
        "rho_pexc_Ehat": float(stats.spearmanr(g.p_exc, g.Ehat)[0]),
        "rho_q05_Ehat": float(stats.spearmanr(g.q05, g.Ehat)[0]),
        "jaccard_conteo_vs_q05": md.jaccard(Tc, Tq), "C_con_q05": "confirmada" if md.jaccard(Tc, Tq) <= 0.5 else "rechazada",
        "invisibles_retenidas_q05": len(inv & Tq) / len(inv),
        "top20_app_en_top_q05": len(set(g.score.nlargest(20).index) & Tq),
        "peor_puesto_q05_de_top20_app": int(g.q05.rank(ascending=False).reindex(g.score.nlargest(20).index).max())}
log(f"\nPrevia del periodo: a = {a:.3f}, b = {b:.3f}")
log(res2)
log(comp.round(3).to_string(index=False))
sal = g.reindex(list(Tapp - Tq)).assign(puesto_q05=g.q05.rank(ascending=False)).sort_values("score", ascending=False)
ent = g.reindex(list(Tq - Tapp)).assign(puesto_app=g.score.rank(ascending=False)).sort_values("q05", ascending=False)
log("\nSalen con el cuantil (10 con mayor P(θ>1)):\n" + sal.head(10).round(3).to_string())
log("\nEntran con el cuantil (10 con mayor cuantil):\n" + ent.head(10).round(3).to_string())

# ================================================================ figura
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
col = {"P(θ>1) 2020–23 (app)": "#e74c3c", "rr (media posterior)": "#f39c12", f"cuantil {Q:.0%} de θ": "#7f3c8d",
       "Conteo 2020–23": "#3498db", "O/Ê crudo": "#95a5a6", "P(θ>1) solo 2023": "#27ae60"}
az = gan[gan.orden == "Azar (mediana)"]
for j, (v, tit) in enumerate([("lift", "Robo 2024 / esperado 2024 en el top k%"),
                              ("exceso", "Robos de 2024 por encima de lo esperado (acumulado)")]):
    ax[j].fill_between(az.k * 100, az[f"{v}_p2.5"], az[f"{v}_p97.5"], color="#cccccc", alpha=0.6, label="Azar (IC 95%)")
    for nom, cc in col.items():
        x = gan[gan.orden == nom]
        ax[j].plot(x.k * 100, x[v], color=cc, lw=2 if "app" in nom else 1.2, label=nom)
    ax[j].axvline(10, color="k", lw=0.6, ls=":")
    ax[j].set_title(tit, loc="left", fontsize=9)
    ax[j].set_xlabel("Top k% de AGEB según el orden armado con 2020–2023")
ax[0].axhline(1, color="k", lw=0.6)
ax[0].legend(fontsize=7, frameon=False)
fig.suptitle("¿El exceso de 2020–2023 anticipa el exceso de 2024? (Ê_2024 fuera de muestra en espacio y tiempo)",
             x=0.02, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(RES / "fig_T7_parte5_ganancia.png")

guarda_json({"parte5": {"c_2024": c, "top10": t10.to_dict("records"),
                        "azar_top10": {"lift": az[np.isclose(az.k, 0.10)].lift.iloc[0]}, "cuantil": res2}})
log("listo")
