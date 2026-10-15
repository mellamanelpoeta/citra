"""T7 · Partes 3 y 4: condiciones sociales (Censo 2020) y sensibilidad de la lista de C a lo que incluye Ê.
Protocolo: analisis/lab/mecanismos/README.md.

Uso: .venv/bin/python analisis/lab/mecanismos/parte3_4_social.py   (~10 min)
"""
from t7_comun import md, RES, SEED, B, Log, guarda_json, indice_social, ic, INDICADORES  # noqa: I001

md.lightgbm()  # en macOS puede relanzar el proceso: va primero

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import CORTE_FIJO, ROOT  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
log = Log("parte3_4")

# ================================================================ Parte 3 · índice
soc, info = indice_social()
log("Índice socioeconómico:", info)
g = pd.read_parquet(ROOT / "data/processed/modelos/riesgo_periodo.parquet").set_index("cvegeo")
g["log_rr"] = np.log(g.rr)
comp = md.composicion("b").xs(CORTE_FIJO, level=0)
g["s_efectivo_barrio"] = comp.s_efectivo_barrio.reindex(g.index)
g = g.join(soc[INDICADORES + ["indice", "pobtot"]])
top_principal = md.top(g.score)
u = g.dropna(subset=["indice"]).copy()
info.update({"n_analisis": len(g), "n_analisis_con_indice": len(u),
             "n_top_sin_indice": int(len(top_principal - set(u.index)))})
log(f"AGEB del análisis: {len(g)}; con índice: {len(u)}; del top 10% sin índice: {info['n_top_sin_indice']}")

# correlación de cada indicador con el índice (para leer las cargas)
corr_ind = {k: float(stats.spearmanr(u[k], u.indice)[0]) for k in INDICADORES}

# S-exceso: ρ(log rr, índice) con IC por bootstrap de bloques AGEB
rho = float(stats.spearmanr(u.log_rr, u.indice)[0])
bl = u.index.map(md.bloques().bloque)
grupos = {k: np.flatnonzero(bl == k) for k in np.unique(bl)}
ids = np.array(list(grupos))
rng = np.random.default_rng(SEED)
bt = []
for _ in range(B):
    i = np.concatenate([grupos[k] for k in rng.choice(ids, size=len(ids), replace=True)])
    bt.append(stats.spearmanr(u.log_rr.values[i], u.indice.values[i])[0])
lo, hi = ic(np.array(bt))
S_exceso = abs(rho) < 0.30
log(f"S-exceso: ρ(log rr, índice) = {rho:.3f} [{lo:.3f}, {hi:.3f}] → "
    f"{'el exceso no se reduce a condiciones sociales' if S_exceso else 'NO pasa (|ρ| ≥ 0.30)'}; "
    f"IC dentro de (−0.3, 0.3): {-0.3 < lo and hi < 0.3}")
# complemento: ρ por indicador
rho_ind = {k: float(stats.spearmanr(u.log_rr, u[k], nan_policy="omit")[0]) for k in INDICADORES}
log("ρ(log rr, indicador):", {k: round(v, 3) for k, v in rho_ind.items()})
# conteo vs índice (para contrastar con el residual)
rho_conteo = float(stats.spearmanr(u.O, u.indice)[0])
rho_E = float(stats.spearmanr(u.Ehat, u.indice)[0])
log(f"Complemento: ρ(O, índice) = {rho_conteo:.3f}; ρ(Ê, índice) = {rho_E:.3f}")

# top 10% por quintil del índice
u["quintil"] = pd.qcut(u.indice, 5, labels=[f"Q{k}" for k in range(1, 6)])
u["top"] = u.index.isin(top_principal)
u["top_conteo"] = u.index.isin(md.top(g.O))
quint = u.groupby("quintil", observed=True).agg(n=("top", "size"), top_riesgo=("top", "sum"),
                                                top_conteo=("top_conteo", "sum"), rr_mediano=("rr", "median"),
                                                indice_min=("indice", "min"), indice_max=("indice", "max"))
quint["pct_top_riesgo"] = quint.top_riesgo / quint.top_riesgo.sum()
quint["pct_top_conteo"] = quint.top_conteo / quint.top_conteo.sum()
log("Top 10% por quintil del índice (Q1 = peores condiciones)\n", quint.round(3).to_string())
quint.to_csv(RES / "parte3_top_por_quintil.csv")

# S-composición
rho_comp = float(stats.spearmanr(u.indice, u.s_efectivo_barrio)[0])
rho_formal = float(stats.spearmanr(u.indice, comp.s_efectivo_formal.reindex(u.index))[0])
log(f"S-composición: ρ(índice, s_efectivo_barrio) = {rho_comp:.3f} (complemento, formal: {rho_formal:.3f})")
u.to_csv(RES / "parte3_ageb_indice.csv")

# ================================================================ Parte 4 · variantes de Ê
p = md.datos(md.PRINCIPAL)
p["log_area"] = np.log(p.area_km2)
med = float(soc.indice.median())
p["indice"] = p.cvegeo.map(soc.indice)
p["indice_falta"] = p.indice.isna().astype(float)
p["indice"] = p.indice.fillna(med)
log(f"Parte 4: AGEB con índice imputado (mediana {med:.3f}): {int(p.drop_duplicates('cvegeo').indice_falta.sum())}")
base = md.cols_m2(p)
variantes = {"principal": base, "(a) + log_area": base + ["log_area"],
             "(b) + índice": base + ["indice", "indice_falta"],
             "(c) + log_area + índice": base + ["log_area", "indice", "indice_falta"]}
filas, ranks = [], {}
for nombre, cols in variantes.items():
    mu, dev = md.cv_espacial(p, cols, modelo="m2")
    r = md.ranking_residual(p, mu)
    rho_p, pares = md.persistencia(p, mu)
    ranks[nombre] = r
    filas.append({"variante": nombre, "dev_espacial": float(np.mean(list(dev.values()))),
                  "jaccard_conteo": md.jaccard(md.top(r.O), md.top(r.score)), "rho_persistencia": rho_p})
    log(nombre, {k: round(v, 4) for k, v in filas[-1].items() if k != "variante"})
t4 = pd.DataFrame(filas).set_index("variante")
assert np.allclose(ranks["principal"].Ehat.reindex(g.index), g.Ehat), "la principal recalculada no coincide"
d0, r0 = t4.dev_espacial["principal"], ranks["principal"]
t4["mejora_dev_vs_principal"] = (d0 - t4.dev_espacial) / d0
t4["jaccard_vs_principal"] = [md.jaccard(md.top(r.score), md.top(r0.score)) for r in ranks.values()]
t4["rho_rr_vs_principal"] = [float(stats.spearmanr(r.rr, r0.rr.reindex(r.index))[0]) for r in ranks.values()]
t4["C_confirmada"] = (t4.jaccard_conteo <= 0.5) & (t4.rho_persistencia >= 0.3)
t4["lista_robusta"] = (t4.jaccard_vs_principal >= 0.60).astype(object)
t4.loc["principal", "lista_robusta"] = np.nan
cand_a = bool(t4.mejora_dev_vs_principal["(a) + log_area"] >= 0.05)
log("Parte 4\n", t4.round(4).to_string())
log(f"(a) candidata (mejora ≥ 5%): {cand_a}")
t4.to_csv(RES / "parte4_variantes.csv")

guarda_json({"parte3": {"indice": info, "corr_indicador_indice": corr_ind, "S_exceso": {
    "rho": rho, "ic": [lo, hi], "pasa": bool(S_exceso), "ic_dentro_0.3": bool(-0.3 < lo and hi < 0.3)},
    "rho_indicadores": rho_ind, "rho_conteo_indice": rho_conteo, "rho_Ehat_indice": rho_E,
    "S_composicion": {"rho_barrio": rho_comp, "rho_formal": rho_formal}, "quintiles": quint.reset_index().to_dict("records")},
    "parte4": {"variantes": t4.reset_index().to_dict("records"), "a_candidata": cand_a}}, "veredictos.json")

# ================================================================ figuras
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].scatter(u.indice, u.log_rr, s=4, alpha=0.35, color="#1f4e79")
axs[0].scatter(u.indice[u.top], u.log_rr[u.top], s=6, color="#c0392b", label="Top 10% por riesgo")
axs[0].axhline(0, color="#333", lw=0.8)
axs[0].set_xlabel("Índice socioeconómico (más alto = mejores condiciones)")
axs[0].set_ylabel("log rr contraído 2020–2024")
axs[0].set_title(f"S-exceso · ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]", loc="left", fontsize=9)
axs[0].legend(fontsize=7, frameon=False)
x = np.arange(5)
axs[1].bar(x - 0.2, quint.pct_top_riesgo, 0.4, color="#c0392b", label="Top 10% por riesgo")
axs[1].bar(x + 0.2, quint.pct_top_conteo, 0.4, color="#3498db", label="Top 10% por conteo")
axs[1].axhline(0.2, color="#333", ls="--", lw=0.8)
axs[1].set_xticks(x, ["Q1\n(peores)", "Q2", "Q3", "Q4", "Q5\n(mejores)"])
axs[1].set_ylabel("% del top 10%")
axs[1].set_title("Top 10% por quintil del índice", loc="left", fontsize=9)
axs[1].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(RES / "fig_T7_parte3_social.png")
log("listo")
