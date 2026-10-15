"""04 · Modelos, prueba de las hipótesis A/B/C, robustez y verificación de supuestos (T2b).

Ejecuta docs/5.Diseno-Experimental.md sin rediseñarlo. Los umbrales de rechazo son los declarados ex ante (§2)
y la regla de selección del "esperado" es la de §4 (M1 salvo que M2 mejore la devianza fuera de muestra ≥ 5%).

Entradas: data/processed/panel_ageb_anio.parquet, data/interim/{delitos,denue}_ageb.parquet, ageb_geom.parquet
Salidas:
- data/processed/modelos/riesgo_anual.parquet, riesgo_periodo.parquet, coef_m1.csv, resumen.json  (→ 05_export)
- analisis/pipeline/outputs/modelos/*.csv y fig1N_*.png
- analisis/pipeline/outputs/resultados.md (se conserva lo que haya después del marcador de interpretación)

Uso: .venv/bin/python analisis/pipeline/04_modelos.py   (~5 min)
"""
import json
from datetime import date

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import INTERIM, ROOT  # noqa: E402
from modelado import (MODELOS, OUT_MOD, PRINCIPAL, asigna_bloques, ajusta_m2, ajusta_nb, bloques,  # noqa: E402
                      bootstrap_beta, capture, capture_anual, cols_comp, cols_m0, cols_m1, cols_m2, cv_espacial,
                      datos, devianza, jaccard, lightgbm, panel_h3, persistencia, predice, ranking_residual,
                      riesgo_anual, top)

lightgbm()  # en macOS puede relanzar el proceso (ver modelado.lightgbm); conviene que sea lo primero
B = 1000    # réplicas del bootstrap de bloques
UMBRAL = {"A_ic_sup": 1.0, "B_dD": 0.05, "B_folds": 4, "C_jaccard": 0.5, "C_rho": 0.3, "M2_sel": 0.05}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# ============================================================ utilidades
def unidad_ageb(p):
    return p.cvegeo.map(bloques().fold).values, p.cvegeo.map(bloques().bloque).values


def unidad_h3(p):
    u = p.drop_duplicates("cvegeo")[["cvegeo", "cx", "cy"]].reset_index(drop=True)
    b = asigna_bloques(u[["cx", "cy"]].values)
    b.index = u.cvegeo
    return p.cvegeo.map(b.fold).values, p.cvegeo.map(b.bloque).values


def temporal(p, cols, modelo, bloque):
    """Entrenar 2020–2023 (o 2019–2023) → predecir 2024 (decisión S3: E fijo 2023-11). El efecto fijo de 2024 no
    es estimable al predecir el año siguiente; supuesto: persistencia del nivel (hereda el de 2023)."""
    q = p.copy()
    c = [x for x in cols if x != "fe_2024"]
    if "fe_2023" in q:
        q.loc[q.anio == 2024, "fe_2023"] = 1.0
    tr, te = (q.anio <= 2023).values, (q.anio == 2024).values
    X, y = q[c].values.astype(float), q.y.values
    if modelo == "nb":
        mu = predice(ajusta_nb(X[tr], y[tr]), X[te])
    else:
        mu = ajusta_m2(X[tr], y[tr], bloque[tr]).predict(X[te])
    return {"dev": devianza(y[te], mu), "mae": float(np.mean(np.abs(y[te] - mu))), "capture": capture(y[te], mu)}


def evalua(nombre, p, fold, bloque, modelo_C, B_boot=B, completo=False):
    """Veredicto de A, B y C para una variante. A: β de M0 con IC por bootstrap de bloques. B: ΔD M0 → M1
    fuera de muestra por fold. C: contracción sobre Ê fuera de muestra del modelo `modelo_C`."""
    c0, c1, c2 = cols_m0(p), cols_m1(p), cols_m2(p)
    m0 = ajusta_nb(p[c0].values.astype(float), p.y.values)
    boot = bootstrap_beta(p, c0, bloque, B=B_boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    mu0, dev0 = cv_espacial(p, c0, fold=fold, bloque=bloque)
    mu1, dev1 = cv_espacial(p, c1, fold=fold, bloque=bloque)
    mejora = {f: (dev0[f] - dev1[f]) / dev0[f] for f in dev0}
    dD, n_mej = float(np.mean(list(mejora.values()))), int(sum(m > 0 for m in mejora.values()))
    if modelo_C == "M2":
        muC, devC = cv_espacial(p, c2, modelo="m2", fold=fold, bloque=bloque)
    else:
        muC, devC = mu1, dev1
    g = ranking_residual(p, muC)
    jac = jaccard(top(g.O), top(g.score))
    rho, pares = persistencia(p, muC)
    r = {"variante": nombre, "n_unidades": p.cvegeo.nunique(), "anios": f"{p.anio.min()}–{p.anio.max()}",
         "beta": float(m0.params[1]), "beta_lo": float(lo), "beta_hi": float(hi),
         "A": "confirmada" if hi < UMBRAL["A_ic_sup"] else "rechazada",
         "dD": dD, "folds_mejora": n_mej,
         "B": "confirmada" if dD >= UMBRAL["B_dD"] and n_mej >= UMBRAL["B_folds"] else "rechazada",
         "modelo_C": modelo_C, "jaccard": jac, "rho": rho,
         "C": "confirmada" if jac <= UMBRAL["C_jaccard"] and rho >= UMBRAL["C_rho"] else "rechazada"}
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    if completo:
        return r, dict(m0=m0, boot=boot, mu0=mu0, mu1=mu1, muC=muC, dev0=dev0, dev1=dev1, devC=devC, mejora=mejora,
                       g=g, pares=pares)
    return r


# ============================================================ 1. Variante principal: modelos y validación
print("== Principal:", PRINCIPAL)
p = datos(PRINCIPAL)
fold, bloque = unidad_ageb(p)
c0, c1, c2 = cols_m0(p), cols_m1(p), cols_m2(p)
oof = {"M0": cv_espacial(p, c0), "M1": cv_espacial(p, c1), "M2": cv_espacial(p, c2, modelo="m2")}
filas = []
for m, (mu, dev) in oof.items():
    t = temporal(p, {"M0": c0, "M1": c1, "M2": c2}[m], "nb" if m != "M2" else "m2", bloque)
    filas.append({"modelo": m, "dev_espacial": np.mean(list(dev.values())), "mae_espacial": float(np.mean(np.abs(p.y - mu))),
                  "capture_espacial": capture_anual(p, mu), "dev_temporal_2024": t["dev"], "mae_temporal_2024": t["mae"],
                  "capture_temporal_2024": t["capture"]})
tab_mod = pd.DataFrame(filas).set_index("modelo")
mejora_m2 = (tab_mod.dev_espacial["M1"] - tab_mod.dev_espacial["M2"]) / tab_mod.dev_espacial["M1"]
MODELO_E = "M2" if mejora_m2 >= UMBRAL["M2_sel"] else "M1"
print(tab_mod.round(4), f"\nM2 vs M1: {mejora_m2:.1%} → Ê con {MODELO_E}")
tab_mod.to_csv(OUT_MOD / "tabla_modelos.csv")

# ============================================================ 2. Hipótesis A, B, C (principal)
res, det = evalua(f"{PRINCIPAL} · AGEB (principal)", p, fold, bloque, MODELO_E, completo=True)
assert np.allclose(det["mu1"], oof["M1"][0])

# β de M1 (complementario a A) e IRR de la composición, con IC por bootstrap de bloques
m1 = ajusta_nb(p[c1].values.astype(float), p.y.values)
rng = np.random.default_rng(42)
grupos = {b_: g.index.values for b_, g in p.groupby(bloque)}
ids = np.array(list(grupos))
X1, y1 = p[c1].values.astype(float), p.y.values
boot1 = []
for _ in range(B):
    i = np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)])
    boot1.append(ajusta_nb(X1[i], y1[i]).params)
boot1 = np.array(boot1)
nombres = ["const"] + c1 + ["alpha"]
coef = pd.DataFrame({"variable": nombres, "coef": m1.params, "ee_modelo": m1.bse,
                     "ic_lo": np.percentile(boot1, 2.5, axis=0), "ic_hi": np.percentile(boot1, 97.5, axis=0)})
# Composición: IRR por +10 puntos porcentuales del giro, a costa de la categoría de referencia (serv_prof_fin)
es_comp = coef.variable.isin(cols_comp(p))
for c in ["coef", "ic_lo", "ic_hi"]:
    coef.loc[es_comp, f"irr10_{c}"] = np.exp(0.1 * coef.loc[es_comp, c])
coef.to_csv(MODELOS / "coef_m1.csv", index=False)
coef.to_csv(OUT_MOD / "coef_m1.csv", index=False)
print(coef.round(3).to_string())

# ============================================================ 3. Robustez entre variantes (diseño §8)
print("== Robustez")
rob = [res]
otro = "M1" if MODELO_E == "M2" else "M2"
r = evalua(f"{PRINCIPAL} · Ê con {otro} (sensibilidad del esperado)", p, fold, bloque, otro)
rob.append({**r, "A": "—", "B": "—", "beta": np.nan, "beta_lo": np.nan, "beta_hi": np.nan, "dD": np.nan,
            "folds_mejora": np.nan})  # A y B no dependen del modelo del esperado
for v in ["V1", "V2", "V3"]:
    q = datos(v)
    f_, b_ = unidad_ageb(q)
    rob.append(evalua(f"{v} · AGEB", q, f_, b_, MODELO_E))
q = datos(PRINCIPAL, e="nov")
f_, b_ = unidad_ageb(q)
rob.append(evalua(f"{PRINCIPAL} · E corte de noviembre de cada año (S2-a)", q, f_, b_, MODELO_E))
for lista, txt in [("a", "lista tentativa de efectivo (un grupo)"), ("c", "sin grupos de efectivo")]:
    q = datos(PRINCIPAL, lista=lista)
    rob.append(evalua(f"{PRINCIPAL} · {txt}", q, fold, bloque, MODELO_E))
for v in [PRINCIPAL, "V1"]:
    h = panel_h3(v)
    f_, b_ = unidad_h3(h)
    rob.append(evalua(f"{v} · malla H3 res. 8 (MAUP)", h, f_, b_, MODELO_E))
rob = pd.DataFrame(rob)
rob.to_csv(OUT_MOD / "robustez.csv", index=False)

# ============================================================ 4. Verificación de supuestos (diseño §7)
print("== Supuestos")
sup = {}
# 4.1 Sobredispersión: razón de verosimilitud Poisson vs NB2 (M1). H0: α = 0 está en la frontera → p/2.
pois = sm.GLM(p.y.values, sm.add_constant(X1, has_constant="add"), family=sm.families.Poisson()).fit()
lr = 2 * (m1.llf - pois.llf)
sup["sobredispersion"] = {"alpha": float(m1.params[-1]), "LR": float(lr), "p": float(stats.chi2.sf(lr, 1) / 2),
                          "llf_nb": float(m1.llf), "llf_poisson": float(pois.llf)}
# 4.2 Autocorrelación espacial de los residuales de M1 (Pearson, promedio por AGEB), pesos queen
import esda  # noqa: E402
from libpysal.weights import Queen  # noqa: E402

geo = gpd.read_parquet(INTERIM / "ageb_geom.parquet")
mu_in = predice(m1, X1)
alpha = m1.params[-1]
pr = (p.y - mu_in) / np.sqrt(mu_in + alpha * mu_in ** 2)
res_ageb = pd.Series(pr.values).groupby(p.cvegeo.values).mean()
gg = geo[geo.cvegeo.isin(res_ageb.index)].reset_index(drop=True)
w = Queen.from_dataframe(gg, use_index=False, silence_warnings=True)
keep = [i for i in range(len(gg)) if i not in set(w.islands)]
if w.islands:
    gg = gg.iloc[keep].reset_index(drop=True)
    w = Queen.from_dataframe(gg, use_index=False, silence_warnings=True)
w.transform = "r"
np.random.seed(42)  # esda usa el generador global para las permutaciones
mi = esda.Moran(res_ageb.reindex(gg.cvegeo).values, w, permutations=999)
# lo mismo sobre el residual del riesgo contraído (log rr del periodo): ¿quedan zonas contiguas en exceso?
mi_rr = esda.Moran(np.log(det["g"].rr.reindex(gg.cvegeo).values), w, permutations=999)
sup["moran_residuales_M1"] = {"I": float(mi.I), "p_sim": float(mi.p_sim), "n": len(gg)}
sup["moran_log_rr"] = {"I": float(mi_rr.I), "p_sim": float(mi_rr.p_sim)}
# 4.3 Estabilidad temporal: β de M0 por año (principal y V1 para comparar 2019 con 2020+)
beta_anio = []
for v in [PRINCIPAL, "V1", "V3"]:
    q = datos(v)
    for a, g in q.groupby("anio"):
        m = ajusta_nb(np.log(g.E.values)[:, None], g.y.values)
        beta_anio.append({"variante": v, "anio": a, "beta": m.params[1], "ee": m.bse[1]})
beta_anio = pd.DataFrame(beta_anio)
beta_anio.to_csv(OUT_MOD / "beta_por_anio.csv", index=False)
bp = beta_anio[beta_anio.variante == PRINCIPAL]
sup["estabilidad_beta"] = {"rango": [float(bp.beta.min()), float(bp.beta.max())],
                           "todos_en_ic": bool(((bp.beta >= res["beta_lo"]) & (bp.beta <= res["beta_hi"])).all())}
json.dump(sup, open(OUT_MOD / "supuestos.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(sup, indent=1))

# ============================================================ 5. Salidas para el export (variante principal)
g = det["g"]
ra = riesgo_anual(p, det["muC"])
ra.to_parquet(MODELOS / "riesgo_anual.parquet", index=False)
g.reset_index().to_parquet(MODELOS / "riesgo_periodo.parquet", index=False)
resumen = {"generado": date.today().isoformat(), "variable": PRINCIPAL, "anios": sorted(int(a) for a in p.anio.unique()),
           "modelo_esperado": MODELO_E, "mejora_m2_sobre_m1": mejora_m2, "umbral": UMBRAL, "principal": res,
           "tabla_modelos": tab_mod.reset_index().to_dict("records"), "supuestos": sup,
           "pares_rho": det["pares"], "folds_dD": det["mejora"], "B_boot": B}


# ============================================================ 6. Figuras
# fig10 · A: β con IC 95% por variante
fa = rob.dropna(subset=["beta"]).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.errorbar(fa.beta, range(len(fa)), xerr=[fa.beta - fa.beta_lo, fa.beta_hi - fa.beta], fmt="o", color="#1f4e79", capsize=3)
ax.axvline(1, color="#c0392b", ls="--", lw=1)
ax.text(1.0, len(fa) - 0.4, " β = 1 (proporcional)", color="#c0392b", fontsize=8, va="top")
ax.set_yticks(range(len(fa)), fa.variante)
ax.set_xlabel("β: elasticidad del robo respecto a los establecimientos (IC 95%)")
fig.suptitle("Hipótesis A · ¿Cuánto? β < 1: el robo crece menos que proporcionalmente", x=0.02, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_MOD / "fig10_A_beta_forest.png")

# fig11 · B: ΔD por fold
fig, ax = plt.subplots(figsize=(5.2, 3))
mj = pd.Series(det["mejora"]) * 100
ax.bar([f"Fold {k + 1}" for k in mj.index], mj.values, color=np.where(mj.values >= 5, "#1f4e79", "#9fb6cd"))
ax.axhline(5, color="#c0392b", ls="--", lw=1)
ax.axhline(mj.mean(), color="#333", lw=1)
ax.text(len(mj) - 0.5, 5, " umbral 5%", color="#c0392b", fontsize=8, va="bottom", ha="right")
ax.text(-0.4, mj.mean(), f"media {mj.mean():.1f}%", fontsize=8, va="bottom")
ax.set_ylabel("Mejora de la devianza fuera de muestra (%)")
ax.set_title("Hipótesis B · ¿Cuáles? M0 → M1 (composición + controles)", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_MOD / "fig11_B_deltaD_folds.png")

# fig12 · C: mapa de P(θ > 1) y top 10% por conteo vs por riesgo
geo = geo.merge(g[["O", "p_exc", "score"]], left_on="cvegeo", right_index=True, how="left")
t_c, t_r = top(g.O), top(g.score)
geo["grupo"] = np.select([geo.cvegeo.isin(t_c & t_r), geo.cvegeo.isin(t_r - t_c), geo.cvegeo.isin(t_c - t_r)],
                         ["ambos", "solo riesgo", "solo conteo"], "resto")
fig, axs = plt.subplots(1, 2, figsize=(10, 5.4))
geo.plot(ax=axs[0], column="p_exc", cmap="RdBu_r", vmin=0, vmax=1, legend=True,
         legend_kwds={"label": "P(θ > 1 | O): prob. de exceso de riesgo", "shrink": 0.6},
         missing_kwds={"color": "#dddddd"}, linewidth=0)
axs[0].set_title(f"Riesgo residual {p.anio.min()}–{p.anio.max()} ({PRINCIPAL}, Ê con {MODELO_E})", fontsize=10)
col = {"ambos": "#7f3c8d", "solo riesgo": "#e74c3c", "solo conteo": "#3498db", "resto": "#eeeeee"}
for k, c in col.items():
    geo[geo.grupo == k].plot(ax=axs[1], color=c, linewidth=0, label=k)
n_ = {k: int((geo.grupo == k).sum()) for k in col}
axs[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in ["solo riesgo", "ambos", "solo conteo"]],
              labels=[f"Solo top riesgo ({n_['solo riesgo']})", f"En ambos ({n_['ambos']})", f"Solo top conteo ({n_['solo conteo']})"],
              loc="lower left", fontsize=8, frameon=False)
axs[1].set_title(f"Top 10%: conteo vs riesgo (Jaccard = {res['jaccard']:.2f})", fontsize=10)
for a in axs:
    a.set_axis_off()
fig.tight_layout()
fig.savefig(OUT_MOD / "fig12_C_mapa_riesgo_top10.png")

# fig13 · C: persistencia (ρ interanual) por variante y por par de años
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
pr_ = pd.Series(det["pares"])
axs[0].bar([k.replace("-20", "–") for k in pr_.index], pr_.values, color="#1f4e79")
axs[0].axhline(0.3, color="#c0392b", ls="--", lw=1)
axs[0].set_ylim(0, 1)
axs[0].set_title(f"ρ de Spearman entre años consecutivos ({PRINCIPAL})", loc="left", fontsize=9)
rr_ = rob.iloc[::-1]
axs[1].barh(rr_.variante, rr_.rho, color="#1f4e79", label="ρ medio (persistencia)")
axs[1].scatter(rr_.jaccard, rr_.variante, color="#e67e22", zorder=3, label="Jaccard top 10% conteo vs riesgo")
axs[1].axvline(0.3, color="#c0392b", ls="--", lw=1)
axs[1].axvline(0.5, color="#e67e22", ls=":", lw=1)
axs[1].set_xlim(0, 1)
axs[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
axs[1].set_title("Hipótesis C por variante (umbrales: ρ ≥ 0.3, Jaccard ≤ 0.5)", loc="left", fontsize=9)
axs[1].tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(OUT_MOD / "fig13_C_persistencia_robustez.png")

# fig14 · estabilidad de β por año
fig, ax = plt.subplots(figsize=(5.6, 3))
for v, gb in beta_anio.groupby("variante"):
    ax.errorbar(gb.anio, gb.beta, yerr=1.96 * gb.ee, marker="o", capsize=2, label=v)
ax.axhline(1, color="#c0392b", ls="--", lw=1)
ax.set_ylabel("β de M0 ajustado por año")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Estabilidad temporal de β", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_MOD / "fig14_estabilidad_beta.png")
plt.close("all")


# ============================================================ 7. resultados.md
def f3(x):
    return "—" if pd.isna(x) else f"{x:.3f}"


def pct(x):
    return "—" if pd.isna(x) else f"{x * 100:.1f}%"


tabla_rob = pd.DataFrame({
    "Variante": rob.variante, "Años": rob.anios, "Unidades": rob.n_unidades,
    "β [IC 95%]": [("—" if pd.isna(b) else f"{b:.3f} [{lo:.3f}, {hi:.3f}]") for b, lo, hi in zip(rob.beta, rob.beta_lo, rob.beta_hi)],
    "A": rob.A, "ΔD (folds)": [("—" if pd.isna(d) else f"{d * 100:.1f}% ({int(k)}/5)") for d, k in zip(rob.dD, rob.folds_mejora)],
    "B": rob.B, "Ê": rob.modelo_C, "Jaccard": rob.jaccard.map(f3), "ρ": rob.rho.map(f3), "C": rob.C})
tm = tab_mod.copy()
tm_md = pd.DataFrame({"Modelo": tm.index, "Devianza (espacial)": tm.dev_espacial.map(lambda x: f"{x:.4f}"),
                      "MAE (espacial)": tm.mae_espacial.map(lambda x: f"{x:.3f}"),
                      "Capture top 10% (espacial)": tm.capture_espacial.map(pct),
                      "Devianza 2024 (temporal)": tm.dev_temporal_2024.map(lambda x: f"{x:.4f}"),
                      "MAE 2024": tm.mae_temporal_2024.map(lambda x: f"{x:.3f}"),
                      "Capture top 10% 2024": tm.capture_temporal_2024.map(pct)})
s = sup
cmp_ = coef[es_comp].assign(g=lambda x: x.variable.str[2:])
irr_md = pd.DataFrame({"Grupo de giros": cmp_.g, "IRR por +10 pp": cmp_.irr10_coef.map(lambda x: f"{x:.3f}"),
                       "IC 95% (bootstrap)": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(cmp_.irr10_ic_lo, cmp_.irr10_ic_hi)]})
ctrl = coef[coef.variable.isin(["logE", "p_grandes", "log_pob", "no_residencial", "log_dist"])]
ctrl_md = pd.DataFrame({"Variable": ctrl.variable, "Coeficiente": ctrl.coef.map(f3),
                        "IC 95% (bootstrap)": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(ctrl.ic_lo, ctrl.ic_hi)]})
n_top = len(t_r)
invisibles = len(t_r - t_c)
consist = {k: (rob[k][rob[k] != "—"] == res[k]).mean() for k in "ABC"}
h3 = rob[rob.variante.str.contains("H3")]
matiz_A = (f" **Matiz (MAUP):** en malla H3 β = {h3.beta.iloc[0]:.2f} [{h3.beta_lo.iloc[0]:.2f}, {h3.beta_hi.iloc[0]:.2f}]: "
           "la sublinealidad depende de la escala de agregación." if (h3.A != res["A"]).any() else "")

md = f"""# Resultados (T2b)

Generado por `analisis/pipeline/04_modelos.py` el {date.today().isoformat()}. Diseño: `docs/5.Diseno-Experimental.md`. Los umbrales son los declarados ex ante en §2 y **no se movieron**.

**Especificación (decidida en `analisis/lab/`, §3.1):** variable principal **{PRINCIPAL}** (robo a negocio sin farderos ni "S/V dentro de negocios"), {p.anio.min()}–{p.anio.max()}, {p.cvegeo.nunique():,} AGEB con E > 0 ({len(p):,} AGEB-año); `E` = establecimientos del DENUE 2023-11; composición con dos grupos de efectivo (barrio / formal); validación espacial por bloques (k-means, 30 bloques, 5 *folds*) y temporal (2020–2023 → 2024). Bootstrap de bloques espaciales: {B} réplicas.

## Veredictos

| Hipótesis | Estadístico | Criterio de rechazo (ex ante) | Resultado | Implicación |
|---|---|---|---|---|
| **A. ¿Cuánto?** El robo crece menos que proporcionalmente con los establecimientos | β = {res['beta']:.3f}, IC 95% [{res['beta_lo']:.3f}, {res['beta_hi']:.3f}] | Límite superior del IC ≥ 1 | **{res['A'].upper()}** | {"Duplicar los establecimientos aumenta el robo en ~" + f"{(2 ** res['beta'] - 1) * 100:.0f}%" + ", no 100%: el conteo crudo sobrerrepresenta a los polos comerciales; el riesgo por negocio es mayor donde hay pocos negocios" if res['A'] == 'confirmada' else "El robo crece al menos proporcionalmente con los establecimientos: no hay evidencia de que la densidad comercial proteja"}{matiz_A} |
| **B. ¿Cuáles?** La composición de giros explica el robo mejor que el conteo | ΔD = {res['dD'] * 100:.1f}% fuera de muestra; mejora en {res['folds_mejora']} de 5 *folds* | ΔD < 5% **o** mejora en < 4 de 5 *folds* | **{res['B'].upper()}** | {"El tipo de negocio importa, con dos lecturas: los giros formales de efectivo concentran robo con violencia; el IRR < 1 del comercio de barrio refleja menos denuncia, no menos robo (ver IRR abajo e interpretación)" if res['B'] == 'confirmada' else "El tipo de negocio no añade capacidad predictiva suficiente: focalizar por zona"} |
| **C. ¿Dónde?** Priorizar por riesgo residual señala zonas distintas a las del conteo y persistentes | Jaccard = {res['jaccard']:.3f}; ρ medio = {res['rho']:.3f} | Jaccard > 0.5 **o** ρ < 0.3 | **{res['C'].upper()}** | {f"{invisibles} de las {n_top} AGEB prioritarias por riesgo no aparecen en el top 10% por conteo: lista de \"zonas que el conteo no ve\" para SEDECO" if res['C'] == 'confirmada' else "La priorización residual no es distinta o no es estable: usar con cautela"} |

Consistencia del veredicto entre variantes de robustez (fracción que coincide con la principal): A {consist['A']:.0%}, B {consist['B']:.0%}, C {consist['C']:.0%}.

## Modelos y validación

Regla de selección del esperado (§4): M1 salvo que M2 mejore la devianza espacial ≥ 5%. M2 mejora {mejora_m2 * 100:.1f}% → **Ê con {MODELO_E}**.

{tm_md.to_markdown(index=False)}

Devianza Poisson media por observación (AGEB-año). *Capture rate*: fracción de los delitos del año que cae en el 10% de AGEB con mayor μ predicha (espacial: promedio de los años).

### M1: composición (IRR por +10 puntos porcentuales del grupo, a costa de servicios profesionales y financieros)

{irr_md.to_markdown(index=False)}

### M1: exposición y controles

{ctrl_md.to_markdown(index=False)}

α (NB2) = {s['sobredispersion']['alpha']:.3f}.

## Robustez (diseño §8)

{tabla_rob.to_markdown(index=False)}

"—" = no aplica (la fila solo cambia el modelo del esperado, no A ni B). En H3, población y distancia se interpolan por área desde las AGEB.

## Verificación de supuestos (diseño §7)

| Supuesto | Prueba | Resultado | Consecuencia |
|---|---|---|---|
| Sobredispersión (NB sobre Poisson) | Razón de verosimilitud Poisson vs NB2 en M1 | α = {s['sobredispersion']['alpha']:.3f}; LR = {s['sobredispersion']['LR']:,.0f}; p {"< 0.001" if s['sobredispersion']['p'] < 0.001 else f"= {s['sobredispersion']['p']:.3g}"} | {"Se confirma: NB2 es la especificación adecuada" if s['sobredispersion']['p'] < 0.05 else "No hay sobredispersión: Poisson bastaría"} |
| Independencia espacial de los residuales | I de Moran de los residuales de Pearson de M1 (media por AGEB), pesos *queen*, 999 permutaciones | I = {s['moran_residuales_M1']['I']:.3f}; p = {s['moran_residuales_M1']['p_sim']:.3f} | {"Hay autocorrelación residual: se justifica el IC de β por bootstrap de bloques y la validación espacial por bloques (ya usados); los EE del modelo serían optimistas" if s['moran_residuales_M1']['p_sim'] < 0.05 else "Sin autocorrelación residual significativa"} |
| (complemento) Agrupamiento del riesgo contraído | I de Moran de log(rr) del periodo | I = {s['moran_log_rr']['I']:.3f}; p = {s['moran_log_rr']['p_sim']:.3f} | {"El exceso forma conglomerados contiguos: la priorización puede leerse por corredores, no solo AGEB sueltas" if s['moran_log_rr']['p_sim'] < 0.05 else "El exceso no forma conglomerados"} |
| Estabilidad temporal | β de M0 ajustado por año (fig14); desempeño en el corte temporal 2024 | β anual entre {s['estabilidad_beta']['rango'][0]:.3f} y {s['estabilidad_beta']['rango'][1]:.3f}; {"todos" if s['estabilidad_beta']['todos_en_ic'] else "no todos"} dentro del IC del periodo. Devianza 2024 de {MODELO_E}: {tab_mod.dev_temporal_2024[MODELO_E]:.3f} | {"La relación es estable en el periodo" if s['estabilidad_beta']['todos_en_ic'] else "Reportar β por subperiodo"} |
| MAUP / escala | A, B y C en malla H3 resolución 8 | Ver filas H3 de la tabla de robustez | {"Los veredictos no dependen de la unidad espacial" if (rob[rob.variante.str.contains("H3")][["A", "B", "C"]] == rob.iloc[0][["A", "B", "C"]].values).all().all() else "Algún veredicto cambia con la escala: matizar"} |
| Propensión a denunciar homogénea respecto a la densidad | Canales probados en el análisis posterior (04b–04d): tamaño de empresa (ENVE), alcaldía, tipo de robo y giro | Ninguno explica A ni la lista de C; el comercio de barrio denuncia menos (ver interpretación) | Verificado en parte; la variación zona por zona sigue siendo limitación explícita |
| Asociación, no causalidad | Declarado | — | Sin pretensión causal |

Figuras: `analisis/pipeline/outputs/modelos/fig10_A_beta_forest.png`, `fig11_B_deltaD_folds.png`, `fig12_C_mapa_riesgo_top10.png`, `fig13_C_persistencia_robustez.png`, `fig14_estabilidad_beta.png`.
"""
resumen["robustez"] = rob.to_dict("records")
json.dump(resumen, open(MODELOS / "resumen.json", "w"), indent=2, ensure_ascii=False, default=float)

MARCA = "<!-- INTERPRETACION: lo que sigue se escribe a mano y 04_modelos.py lo conserva -->"
ruta = ROOT / "analisis/pipeline/outputs/resultados.md"
previo = ruta.read_text().split(MARCA, 1)[1] if ruta.exists() and MARCA in ruta.read_text() else "\n## Interpretación\n\n(pendiente)\n"
ruta.write_text(md + "\n" + MARCA + previo)
print("listo →", ruta)
