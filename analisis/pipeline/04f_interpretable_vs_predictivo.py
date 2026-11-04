"""04f · Modelo interpretable (M1) frente a modelo predictivo (M2). Análisis posterior de presentación (2026-10-14):
no modifica veredictos, umbrales ni salidas de 04_modelos.py. Hermano de 04b–04e.

1. Contraste por bloques dentro de M1: exposición → + controles → + composición → completo (M1). Para cada
   especificación: devianza fuera de muestra (validación espacial por bloques), devianza explicada, ΔD respecto a la
   exposición sola y β con IC 95% por bootstrap de bloques espaciales. M0 = M1 restringido a la exposición.
2. ¿M2 aprende lo mismo que M1? Contribuciones SHAP de LightGBM (nativas, `pred_contrib`) frente a las de M1
   (coeficiente × desviación respecto a la media, que es el SHAP exacto de un modelo lineal en la escala del log).
   - Importancia: media |contribución| por variable y por bloque, en ambos modelos.
   - Elasticidad implícita de M2: pendiente de la contribución SHAP de log E sobre log E (global y por tramo, con
     el corte en el p75 como la parte 1b de T7), comparada con β de M1.
   - IRR implícito de cada giro en M2 (pendiente SHAP por +10 pp) frente al IRR de M1.

Uso: .venv/bin/python analisis/pipeline/04f_interpretable_vs_predictivo.py   (~10 min)
Salidas: analisis/pipeline/outputs/modelos/interpretable_vs_predictivo.{csv,md}, fig19_importancia_M1_M2.png
"""
from modelado import lightgbm

lightgbm()  # en macOS puede relanzar el proceso: debe ir antes que todo lo demás

import warnings  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from modelado import (CONTROLES, OUT_MOD, SEED, ajusta_m2, ajusta_nb, bloques, cols_comp, cols_m0, cols_m1,  # noqa: E402
                      cols_m2, cv_espacial, datos)

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
B = 500
FILAS = []
NOMBRES = {"logE": "log(negocios)", "p_grandes": "% negocios grandes", "log_pob": "log(población)",
           "no_residencial": "AGEB sin población", "log_dist": "log(distancia a estación)", "anio": "año"}


def reg(seccion, metrica, valor, nota=""):
    FILAS.append({"seccion": seccion, "metrica": metrica, "valor": valor, "nota": nota})
    v = f"{valor:.3f}" if isinstance(valor, float) else valor
    print(f"  [{seccion}] {metrica}: {v}  {nota}")


def dev_total(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.sum(2 * (t - (y - mu))))


def boot_beta(p, cols, B=B):
    """β (coeficiente de log E) con IC 95% percentil por bootstrap de bloques espaciales."""
    X, y = p[cols].values.astype(float), p.y.values
    grupos = {b_: g.index.values for b_, g in p.groupby(p.cvegeo.map(bloques().bloque))}
    ids = np.array(list(grupos))
    rng = np.random.default_rng(SEED)
    muestras = [np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)]) for _ in range(B)]
    k = cols.index("logE") + 1
    out = np.array(Parallel(n_jobs=-1)(delayed(lambda i: ajusta_nb(X[i], y[i]).params[k])(i) for i in muestras))
    return float(ajusta_nb(X, y).params[k]), *np.nanpercentile(out, [2.5, 97.5])


p = datos()
nulo = p.groupby("anio").y.transform("mean").values
c0, cc = cols_m0(p), cols_comp(p)
fe = [c for c in c0 if c != "logE"]

# ============================================================ 1. Contraste por bloques dentro de M1
print("== 1. Contraste por bloques dentro de M1")
esp = {"Exposición (M0)": c0, "Exposición + controles": c0 + CONTROLES, "Exposición + composición": c0 + cc,
       "Completo (M1)": cols_m1(p)}
oof = {k: cv_espacial(p, v) for k, v in esp.items()}
oof["LightGBM (M2)"] = cv_espacial(p, cols_m2(p), modelo="m2")
base = oof["Exposición (M0)"][1]
tabla = []
for k, (mu, dev) in oof.items():
    dd = {f: (base[f] - dev[f]) / base[f] for f in dev}
    fila = {"especificación": k, "devianza media": float(np.mean(list(dev.values()))),
            "devianza explicada": 1 - dev_total(p.y, mu) / dev_total(p.y, nulo),
            "ΔD vs exposición": float(np.mean(list(dd.values()))), "folds con mejora": int(sum(v > 0 for v in dd.values()))}
    if k in esp:
        fila["β"], fila["β_lo"], fila["β_hi"] = boot_beta(p, esp[k])
    tabla.append(fila)
    for m in ["devianza explicada", "ΔD vs exposición", "β"]:
        if m in fila:
            reg("1", f"{k} · {m}", fila[m])
tabla = pd.DataFrame(tabla)

# ============================================================ 2. SHAP de M2 frente a contribuciones de M1
print("== 2. Contribuciones M1 vs M2")
c1, c2 = cols_m1(p), cols_m2(p)
X1, X2 = p[c1].values.astype(float), p[c2].values.astype(float)
m1 = ajusta_nb(X1, p.y.values)
coef = pd.Series(np.asarray(m1.params)[1:len(c1) + 1], index=c1)
contrib1 = pd.DataFrame((X1 - X1.mean(axis=0)) * coef.values, columns=c1)
contrib1["anio"] = contrib1[fe].sum(axis=1)  # los efectos fijos de año se suman en una sola contribución
contrib1 = contrib1.drop(columns=fe)

m2 = ajusta_m2(X2, p.y.values, p.cvegeo.map(bloques().bloque).values)
shap = pd.DataFrame(m2.predict(X2, pred_contrib=True)[:, :-1], columns=c2)
reg("2", "M2 · hiperparámetros", str(m2.config_))

imp = pd.DataFrame({"M1": contrib1.abs().mean(), "M2": shap.abs().mean()})
bloque_de = {c: ("composición" if c in cc else "controles" if c in CONTROLES else NOMBRES.get(c, c)) for c in imp.index}
imp_b = pd.DataFrame({"M1": contrib1.T.groupby(bloque_de).sum().T.abs().mean(),
                      "M2": shap.T.groupby(bloque_de).sum().T.abs().mean()})
for t in (imp, imp_b):
    t["% M1"], t["% M2"] = t.M1 / t.M1.sum(), t.M2 / t.M2.sum()
    t.sort_values("M2", ascending=False, inplace=True)
rho = float(stats.spearmanr(imp.M1, imp.M2)[0])
reg("2", "Spearman de la importancia por variable M1 vs M2", rho, f"{len(imp)} variables")
for k, r in imp_b.iterrows():
    reg("2", f"bloque {k} · % de la importancia", float(r["% M2"]), f"M1: {r['% M1']:.3f}")

# Elasticidad implícita de M2: pendiente de SHAP(log E) sobre log E
k75 = np.quantile(p.drop_duplicates("cvegeo").logE, 0.75)
alto = (p.logE > k75).values
elas = {"global": (np.ones(len(p), bool), float(coef["logE"]))}
m1q = ajusta_nb(np.column_stack([p.logE, np.maximum(p.logE - k75, 0), p[[c for c in c1 if c != "logE"]]]), p.y.values)
bq = np.asarray(m1q.params)
elas["T1–T3"] = (~alto, float(bq[1]))
elas["T4 (cuartil superior)"] = (alto, float(bq[1] + bq[2]))
filas_e = []
for k, (m, b1) in elas.items():
    pend = float(np.polyfit(p.logE[m], shap.logE[m], 1)[0])
    filas_e.append({"tramo": k, "β M1": b1, "pendiente SHAP M2": pend})
    reg("2", f"elasticidad {k}", pend, f"M1: {b1:.3f}")
elas = pd.DataFrame(filas_e)

# IRR implícito por giro: pendiente SHAP por +10 pp frente al IRR de M1
filas_g = []
for c in cc:
    pend = float(np.polyfit(p[c], shap[c], 1)[0]) if p[c].std() > 0 else np.nan
    filas_g.append({"giro": c.removeprefix("s_"), "IRR M1 (+10 pp)": float(np.exp(0.1 * coef[c])),
                    "IRR implícito M2 (+10 pp)": float(np.exp(0.1 * pend))})
giros = pd.DataFrame(filas_g).sort_values("IRR M1 (+10 pp)", ascending=False)
mismo = int(((giros["IRR M1 (+10 pp)"] > 1) == (giros["IRR implícito M2 (+10 pp)"] > 1)).sum())
reg("2", "giros con la misma dirección en M1 y M2", float(mismo), f"de {len(giros)}")

# ============================================================ figura
fig, ax = plt.subplots(figsize=(6.5, 3.6))
v = imp.iloc[::-1]
y_ = np.arange(len(v))
ax.barh(y_ + 0.2, v["% M1"] * 100, height=0.4, color="#3498db", label="M1 (binomial negativa)")
ax.barh(y_ - 0.2, v["% M2"] * 100, height=0.4, color="#e67e22", label="M2 (LightGBM, SHAP)")
ax.set_yticks(y_, [NOMBRES.get(c, c.removeprefix("s_").replace("_", " ")) for c in v.index])
ax.set_xlabel("% de la contribución media absoluta (escala log del robo esperado)")
ax.legend(frameon=False, loc="lower right")
fig.suptitle(f"¿Qué usa cada modelo para predecir el robo? (Spearman entre rankings = {rho:.2f})",
             x=0.02, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_MOD / "fig19_importancia_M1_M2.png")

# ============================================================ salidas
pd.DataFrame(FILAS).to_csv(OUT_MOD / "interpretable_vs_predictivo.csv", index=False, float_format="%.4f")
t1 = tabla.copy()
t1["β [IC 95%]"] = [f"{r.β:.3f} [{r.β_lo:.3f}, {r.β_hi:.3f}]" if pd.notna(r.get("β")) else "—" for _, r in t1.iterrows()]
t1 = t1.drop(columns=["β", "β_lo", "β_hi"])
(OUT_MOD / "interpretable_vs_predictivo.md").write_text(
    "# 04f · Modelo interpretable (M1) frente a predictivo (M2) (salida automática)\n\n"
    f"Panel V2b, {p.cvegeo.nunique():,} AGEB × {p.anio.nunique()} años. Validación espacial por bloques (5 *folds*). "
    f"Bootstrap de bloques: {B} réplicas.\n\n## 1. Contraste por bloques dentro de M1\n\n"
    + t1.round(4).to_markdown(index=False)
    + "\n\n## 2. Importancia por bloque (media |contribución|, escala log)\n\n" + imp_b.round(4).to_markdown()
    + "\n\n### Por variable\n\n" + imp.round(4).to_markdown()
    + f"\n\nSpearman entre rankings de importancia: {rho:.3f}.\n\n## 3. Elasticidad implícita de M2\n\n"
    + elas.round(3).to_markdown(index=False)
    + f"\n\n## 4. IRR por giro (+10 pp, a costa de servicios profesionales)\n\n" + giros.round(3).to_markdown(index=False)
    + f"\n\nMisma dirección en {mismo} de {len(giros)} giros.\n", encoding="utf-8")
print("listo →", OUT_MOD / "interpretable_vs_predictivo.md")
