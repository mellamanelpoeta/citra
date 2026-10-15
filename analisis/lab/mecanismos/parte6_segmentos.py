"""T7 · Parte 6: ¿dónde le gana el modelo al conteo? Hecha el 2026-10-14, POSTERIOR a los veredictos y
SIN regla ex ante: descriptiva, no cambia veredictos, el Ê, la app ni el contrato.

Mismos insumos que la parte 5: órdenes armados con 2020–2023 (Ê fuera de muestra espacial) y Ê_2024 que no vio la AGEB,
reescalado a ΣO_2024 (c = ΣO/ΣÊ).

1. Dentro de cada quintil de robo 2020–2023 (y, aparte, de número de negocios E): se toma el 10% del quintil con mayor
   P(θ>1) (modelo) y el 10% con más robos (conteo; empates al azar) y se compara su robo 2024 frente al esperado.
   Referencia: 1000 selecciones al azar del mismo tamaño dentro del quintil (el nivel propio del quintil no es 1).
   IC por bootstrap de bloques espaciales con las listas fijas.
2. La lista global del modelo (top 10% de la ciudad por P(θ>1) 2020–2023): cuántas zonas caen en cada quintil de robo
   y cómo les fue en 2024 frente al resto de su quintil. Incluye las "zonas que el conteo no ve".

Uso: .venv/bin/python analisis/lab/mecanismos/parte6_segmentos.py   (~5 min)
Salidas: resultados/parte6_*.csv, fig_T7_parte6_segmentos.png, log_parte6.txt, veredictos_parte6.json
"""
from t7_comun import md, RES, SEED, B, Log, guarda_json, ic  # noqa: I001

md.lightgbm()

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
log = Log("parte6")
QS = ["Q1 (menos)", "Q2", "Q3", "Q4", "Q5 (más)"]

# ================================================================ insumos (como en la parte 5)
p = md.datos(md.PRINCIPAL)
fold, bloque = p.cvegeo.map(md.bloques().fold).values, p.cvegeo.map(md.bloques().bloque).values
pre_m = (p.anio <= 2023).values
pre = p[pre_m].reset_index(drop=True)
mu_pre, _ = md.cv_espacial(pre, md.cols_m2(pre), modelo="m2")
gpre = md.ranking_residual(pre, mu_pre)

c2 = md.cols_m2(p)
X, y = p[c2].values.astype(float), p.y.values
mu24 = np.full(len(p), np.nan)
for f in range(5):
    tr, te = (fold != f) & pre_m, (fold == f) & (p.anio.values == 2024)
    mu24[te] = md.ajusta_m2(X[tr], y[tr], bloque[tr]).predict(X[te])
es24 = (p.anio == 2024).values
d = pd.DataFrame({"y": y[es24], "mu": mu24[es24], "E": p.E.values[es24]}, index=p.cvegeo.values[es24])
d["cE"] = d.mu * d.y.sum() / d.mu.sum()
d = d.join(gpre[["O", "Ehat", "rr", "score"]].rename(columns={"O": "O_pre", "Ehat": "E_pre"}))
d.to_parquet(RES / "parte6_insumos.parquet")
log(f"{len(d)} AGEB; ΣO_2024 = {d.y.sum():.0f}")

rng = np.random.default_rng(SEED)
O24, cE24 = d.y.values, d.cE.values
blq = d.index.map(md.bloques().bloque).values
grupos = {b_: np.flatnonzero(blq == b_) for b_ in np.unique(blq)}
ids = np.array(list(grupos))
muestras = [np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)]) for _ in range(B)]


def lift(m, i=slice(None)):
    o, e = O24[i][m[i]], cE24[i][m[i]]
    return o.sum() / e.sum()


def boot(m, ref=None):
    """IC del lift de la máscara m y, si se da ref, de la diferencia m − ref."""
    bt = np.array([[lift(m, i), lift(m, i) - lift(ref, i) if ref is not None else np.nan] for i in muestras])
    return ic(bt[:, 0]), (ic(bt[:, 1]) if ref is not None else (np.nan, np.nan))


def resumen(m):
    return {"lift": lift(m), "exceso_por_zona": float((O24[m] - cE24[m]).mean()), "robo_2024_por_zona": float(O24[m].mean()),
            "robo_pre_anual_por_zona": float(d.O_pre.values[m].mean() / 4), "pct_O_mayor_E": float((O24[m] > cE24[m]).mean())}


# ================================================================ 1. dentro de cada quintil
filas = []
jit = rng.random(len(d)) * 1e-3  # desempate al azar del conteo
for nom_seg, s in [("robo 2020–23", d.O_pre), ("negocios (E)", d.E)]:
    quint = pd.qcut(s.rank(method="first"), 5, labels=QS).values
    for qn in QS:
        en_q = quint == qn
        k = int(round(en_q.sum() * 0.10))
        iq = np.flatnonzero(en_q)
        m_mod = np.zeros(len(d), bool); m_mod[iq[np.argsort(-d.score.values[iq])[:k]]] = True
        m_cnt = np.zeros(len(d), bool); m_cnt[iq[np.argsort(-(d.O_pre.values[iq] + jit[iq]))[:k]]] = True
        az = []
        for _ in range(B):
            m = np.zeros(len(d), bool); m[rng.choice(iq, k, replace=False)] = True
            az.append(lift(m))
        (lo_m, hi_m), (lo_d, hi_d) = boot(m_mod, m_cnt)
        (lo_c, hi_c), _ = boot(m_cnt)
        r_mod, r_cnt = resumen(m_mod), resumen(m_cnt)
        filas.append({"segmento": nom_seg, "quintil": qn, "n_quintil": int(en_q.sum()), "k": k,
                      "rango_segmento": f"{s.values[en_q].min():.0f}–{s.values[en_q].max():.0f}",
                      "lift_quintil": lift(en_q), "lift_azar_med": float(np.median(az)),
                      "lift_azar_ic": ic(np.array(az)),
                      "lift_modelo": r_mod["lift"], "lift_modelo_ic": (lo_m, hi_m),
                      "lift_conteo": r_cnt["lift"], "lift_conteo_ic": (lo_c, hi_c),
                      "dif_modelo_conteo_ic": (lo_d, hi_d),
                      "exceso_zona_modelo": r_mod["exceso_por_zona"], "exceso_zona_conteo": r_cnt["exceso_por_zona"],
                      "robo24_zona_modelo": r_mod["robo_2024_por_zona"], "robo24_zona_conteo": r_cnt["robo_2024_por_zona"],
                      "pctOE_modelo": r_mod["pct_O_mayor_E"], "pctOE_conteo": r_cnt["pct_O_mayor_E"],
                      "jaccard": md.jaccard(set(np.flatnonzero(m_mod)), set(np.flatnonzero(m_cnt)))})
seg = pd.DataFrame(filas)
seg.to_csv(RES / "parte6_por_quintil.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    log(seg.round(2).to_string(index=False))

# ================================================================ 2. la lista global del modelo, por quintil de robo
quint = pd.qcut(d.O_pre.rank(method="first"), 5, labels=QS).values
L_mod = d.index.isin(md.top(gpre.score))
L_cnt = d.index.isin(md.top(gpre.O))
filas = []
for qn in QS + ["Todas"]:
    en_q = np.ones(len(d), bool) if qn == "Todas" else quint == qn
    for nom, m in [("lista del modelo", L_mod & en_q), ("  · que el conteo no ve", L_mod & ~L_cnt & en_q),
                   ("resto del quintil", ~L_mod & en_q)]:
        if m.sum() == 0:
            continue
        (lo, hi), (dlo, dhi) = boot(m, ~L_mod & en_q) if "resto" not in nom else boot(m)
        filas.append({"quintil_robo": qn, "grupo": nom, "n": int(m.sum()), **resumen(m), "lift_ic": (lo, hi),
                      "dif_vs_resto_ic": (dlo, dhi)})
glob = pd.DataFrame(filas)
glob.to_csv(RES / "parte6_lista_global.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    log("\n" + glob.round(2).to_string(index=False))

# ================================================================ figura
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
for j, nom_seg in enumerate(["robo 2020–23", "negocios (E)"]):
    x = seg[seg.segmento == nom_seg].reset_index(drop=True)
    pos = np.arange(5)
    for dx, col, lab, cc in [(-0.25, "lift_azar_med", "Azar dentro del quintil", "#bbbbbb"),
                             (0, "lift_conteo", "10% con más robos", "#3498db"),
                             (0.25, "lift_modelo", "10% con más exceso (modelo)", "#e74c3c")]:
        ax[j].bar(pos + dx, x[col], width=0.25, color=cc, label=lab)
        icc = {"lift_azar_med": "lift_azar_ic", "lift_conteo": "lift_conteo_ic", "lift_modelo": "lift_modelo_ic"}[col]
        lo, hi = np.array([v[0] for v in x[icc]]), np.array([v[1] for v in x[icc]])
        ax[j].errorbar(pos + dx, x[col], yerr=[x[col] - lo, hi - x[col]], fmt="none", ecolor="k", lw=0.7, capsize=2)
    ax[j].set_xticks(pos, [f"{q}\n{r}" for q, r in zip(x.quintil, x.rango_segmento)], fontsize=7)
    ax[j].axhline(1, color="k", lw=0.6)
    ax[j].set_title(f"Quintiles de {nom_seg}", loc="left", fontsize=9)
ax[0].set_ylabel("Robo 2024 / esperado 2024")
ax[0].legend(fontsize=7, frameon=False)
fig.suptitle("Dentro de zonas parecidas: ¿quién anticipa mejor el exceso de 2024, el modelo o el conteo?",
             x=0.02, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(RES / "fig_T7_parte6_segmentos.png")
guarda_json({"parte6": {"por_quintil": seg.to_dict("records"), "lista_global": glob.to_dict("records")}})
log("listo")
