"""04c · Revisión de la hipótesis C (y dos cabos de A y B) tras T2b. Análisis exploratorio posterior: no modifica
veredictos, umbrales ni salidas de 04_modelos.py. Hermano de 04b_sesgo_denuncia.py.

1. Denuncia en C: ¿la lista de prioridades sobrevive con solo robo con violencia (el más denunciado)? Referencia de
   ruido: V2b submuestreado al azar al mismo tamaño. Complementos: Ê sin grupos de efectivo (la lista ¿depende del
   IRR de barrio?) e IRR de barrio por tipo de robo (¿0.78 es menos robo o menos denuncia?).
2. Exposición: ¿cuántas zonas prioritarias tienen muy pocos negocios? Sensibilidad con personal ocupado como E.
3. Causalidad inversa en E (A): β con E del DENUE 2016-01 (marco CE 2014, antes del periodo) y prueba directa del
   mecanismo: ¿el robo 2016–2018 predice pérdida de establecimientos 2016 → 2023?
4. B: aporte de la composición separado del de los controles.
5. Validación operativa de C: zonas señaladas con 2020–2023 → ¿siguen por encima de lo esperado en 2024?

Uso: .venv/bin/python analisis/pipeline/04c_robustez_C.py   (~10 min)
Salidas: analisis/pipeline/outputs/modelos/robustez_C.csv, robustez_C.md, fig16_C_validacion_2024.png
"""
from modelado import lightgbm

lightgbm()  # en macOS puede relanzar el proceso: debe ir antes que todo lo demás

import importlib  # noqa: E402
import warnings  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import CONFIG, INTERIM, RAW, asigna_ageb, lee_mg, lee_panel, norm  # noqa: E402
from modelado import (CONTROLES, MODELOS, OUT_MOD, SEED, ajusta_m2, bloques, cols_comp, cols_m0, cols_m1,  # noqa: E402
                      cols_m2, contraccion, cv_espacial, datos, jaccard, ranking_residual, top)

s04b = importlib.import_module("04b_sesgo_denuncia")
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
B = 500
FILAS, MD = [], []


def reg(prueba, metrica, valor, nota=""):
    FILAS.append({"prueba": prueba, "metrica": metrica, "valor": valor, "nota": nota})
    v = f"{valor:.3f}" if isinstance(valor, float) else valor
    print(f"  [{prueba}] {metrica}: {v}  {nota}")


def boot_coefs(p, cols, nombres, B=B):
    """Coeficientes NB2 de `nombres` con IC 95% por bootstrap de bloques espaciales (esquema de 04b/modelado)."""
    X, y = p[cols].values.astype(float), p.y.values
    grupos = {b_: g.index.values for b_, g in p.groupby(p.cvegeo.map(bloques().bloque))}
    ids = np.array(list(grupos))
    rng = np.random.default_rng(SEED)
    muestras = [np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)]) for _ in range(B)]
    k = [cols.index(n) + 1 for n in nombres]
    out = np.array(Parallel(n_jobs=-1)(delayed(lambda i: s04b.ajusta(X[i], y[i]).params[k])(i) for i in muestras))
    punto = s04b.ajusta(X, y).params[k]
    return {n: (punto[j], *np.nanpercentile(out[:, j], [2.5, 97.5])) for j, n in enumerate(nombres)}


def ranking_m2(q, cols=None):
    mu, _ = cv_espacial(q, cols or cols_m2(q), modelo="m2")
    return ranking_residual(q, mu)


def compara(g_ref, g, etiqueta, prueba):
    """Parecido entre dos rankings residuales: Jaccard del top 10%, ρ de Spearman del rr, y cuántas de las
    'zonas que el conteo no ve' de la referencia siguen en el top del otro."""
    t_ref, t, t_cnt = top(g_ref.score), top(g.score), top(g_ref.O)
    comun = g_ref.index.intersection(g.index)
    inv = t_ref - t_cnt
    r = {"jaccard_vs_principal": jaccard(t_ref, t), "jaccard_vs_conteo": jaccard(t_cnt, t),
         "rho_rr": stats.spearmanr(g_ref.rr[comun], g.rr[comun])[0], "invisibles_retenidas": len(inv & t) / len(inv)}
    for k, v in r.items():
        reg(prueba, f"{etiqueta} · {k}", float(v))
    return r


# ============================================================ insumos comunes
p = datos("V2b")
c2 = cols_m2(p)
g0 = pd.read_parquet(MODELOS / "riesgo_periodo.parquet").set_index("cvegeo")  # ranking principal (04_modelos)
pan = lee_panel()
at = pan[pan.anio == 2023].set_index("cvegeo")[["alcaldia", "colonia", "E_fijo", "po_est_fijo", "ps_efectivo_barrio_fijo"]]
T0, C0 = top(g0.score), top(g0.O)

# ============================================================ 1. Denuncia en C
print("== 1. C con solo robo con violencia")
tipos = s04b.panel_por_tipo(p)
g_cv = ranking_m2(tipos["cv"])
# Lo usa 05_export.py para el aviso de la app "señal dominada por robo sin violencia" (decisión del 2026-10-11)
g_cv.rename_axis("cvegeo").reset_index().to_parquet(MODELOS / "riesgo_periodo_cv.parquet", index=False)
r_cv = compara(g0, g_cv, "solo con violencia", "1")
g_sv = ranking_m2(tipos["sv"])
r_sv = compara(g0, g_sv, "solo sin violencia", "1")
reg("1", "C propia con violencia · jaccard conteo vs riesgo", jaccard(top(g_cv.O), top(g_cv.score)))

# Referencia de ruido: adelgazamiento binomial de V2b a la fracción de carpetas con violencia. Si cada carpeta se
# conserva con prob. π de forma independiente, el esperado de la AGEB es π·Ê: no hace falta reajustar el modelo.
pi = tipos["cv"].y.sum() / p.y.sum()
rng = np.random.default_rng(SEED)
ref = []
for _ in range(200):
    O = rng.binomial(g0.O.astype(int), pi)
    s = contraccion(O, pi * g0.Ehat)
    gt = pd.DataFrame({"O": O, "rr": s["rr"], "score": s["p_exc"] + 1e-6 * s["rr"]}, index=g0.index)
    ref.append([jaccard(T0, top(gt.score)), stats.spearmanr(g0.rr, gt.rr)[0], len((T0 - C0) & top(gt.score)) / len(T0 - C0)])
ref = np.array(ref)
for j, m in enumerate(["jaccard_vs_principal", "rho_rr", "invisibles_retenidas"]):
    reg("1", f"ruido puro (V2b al {pi:.0%}, 200 réplicas) · {m}", float(np.median(ref[:, j])),
        f"P5–P95 [{np.percentile(ref[:, j], 5):.3f}, {np.percentile(ref[:, j], 95):.3f}]")

# ¿La lista depende del IRR de barrio? Ê sin grupos de efectivo (lista c: barrio y formal vuelven a su sector)
g_c = ranking_m2(datos("V2b", lista="c"))
compara(g0, g_c, "Ê sin grupos de efectivo", "1")

# Peso del comercio de barrio en cada lista
for nom, t in [("top riesgo principal", T0), ("top conteo", C0), ("conteo no ve (principal)", T0 - C0),
               ("top riesgo con violencia", top(g_cv.score)), ("top riesgo sin violencia", top(g_sv.score))]:
    reg("1", f"% efectivo de barrio · {nom}", float(at.ps_efectivo_barrio_fijo.reindex(list(t)).mean()))

# ¿0.78 es menos robo o menos denuncia? Si es denuncia, el IRR de barrio debe acercarse a 1 en el robo con violencia
for k, nom in [("cv", "con violencia"), ("sv", "sin violencia")]:
    q = tipos[k]
    c1 = cols_m1(q)
    b = boot_coefs(q, c1, ["s_efectivo_barrio", "s_efectivo_formal"])
    for n, (pt, lo, hi) in b.items():
        reg("1", f"IRR +10 pp {n[2:]} · {nom}", float(np.exp(0.1 * pt)), f"IC 95% [{np.exp(0.1 * lo):.3f}, {np.exp(0.1 * hi):.3f}]")

# ============================================================ 2. Exposición
print("== 2. Exposición")
E = at.E_fijo.reindex(g0.index)
for umbral in [20, 50]:
    reg("2", f"AGEB del top riesgo con E < {umbral}", int((E.reindex(list(T0)) < umbral).sum()),
        f"de {len(T0)}; en toda la ciudad: {(E < umbral).mean():.1%}")
reg("2", "E mediana · top riesgo / todas", float(E.reindex(list(T0)).median()), f"todas: {E.median():.0f}")
reg("2", "E mediana · top conteo", float(E.reindex(list(C0)).median()))
q = p.copy()
q["logE"] = np.log(q.cvegeo.map(at.po_est_fijo))
assert np.isfinite(q.logE).all()
g_po = ranking_m2(q)
compara(g0, g_po, "E = personal ocupado", "2")
top20 = g0.sort_values("score", ascending=False).head(20).join(at).assign(
    puesto_po=lambda x: x.index.map(g_po.score.rank(ascending=False)),
    puesto_cv=lambda x: x.index.map(g_cv.score.rank(ascending=False)))

# ============================================================ 3. Causalidad inversa en E
print("== 3. E anterior al periodo (DENUE 2016-01, marco CE 2014)")
cache = INTERIM / "_eda/denue2016_robo1618_ageb.parquet"
if not cache.exists():
    urb = lee_mg("09a")[["CVEGEO", "geometry"]]
    rur = lee_mg("09ar")[["CVEGEO", "geometry"]]
    e = pd.read_parquet(RAW / "denue/parquet/denue_09_2016_01.parquet", columns=["latitud", "longitud"])
    e = e.apply(pd.to_numeric, errors="coerce").dropna()
    e["cvegeo"] = asigna_ageb(e, urb, rur)
    # mismo filtro de V1 (robo a negocio agregado, lo único comparable antes de 2020) y de coordenadas que 01_datos
    api = pd.read_parquet(RAW / "fgj/fgj_delitos_objetivo_2016_2025.parquet")
    api["anio"] = pd.to_datetime(api.fecha_hecho.astype(str).str[:10], errors="coerce").dt.year
    api = api[api.anio.between(2016, 2018)].copy()
    api["delito_norm"] = api.delito.map(norm)
    api = api.merge(pd.read_csv(CONFIG / "homologacion_fgj.csv")[["delito_norm", "objetivo"]], on="delito_norm", how="left")
    api = api[api.objetivo == "negocio"]
    for c in ["latitud", "longitud"]:
        api[c] = pd.to_numeric(api[c], errors="coerce")
    api = api[api.latitud.between(19.0, 19.65) & api.longitud.between(-99.40, -98.90)]
    api["cvegeo"] = asigna_ageb(api, urb, rur)
    h = pd.DataFrame({"E2016": e.cvegeo.value_counts(), "robo1618": api.cvegeo.value_counts()}).fillna(0)
    h[h.index.str.len() == 13].rename_axis("cvegeo").reset_index().to_parquet(cache, index=False)
h = pd.read_parquet(cache).set_index("cvegeo")
u = at.join(h, how="left").fillna({"E2016": 0, "robo1618": 0})
u = u[(u.E_fijo > 0) & (u.E2016 > 0)]
reg("3", "corr log E 2016-01 vs 2023-11", float(np.corrcoef(np.log(u.E2016), np.log(u.E_fijo))[0, 1]), f"{len(u):,} AGEB")
q = p[p.cvegeo.isin(u.index)].reset_index(drop=True)
c0 = cols_m0(q)
b_ref, boot_ref = s04b.beta_boot(q, c0)
q16 = q.assign(logE=np.log(q.cvegeo.map(u.E2016)))
b16, boot16 = s04b.beta_boot(q16, c0)
for nom, b_, bt in [("β con E 2023-11 (misma muestra)", b_ref, boot_ref), ("β con E 2016-01", b16, boot16)]:
    lo, hi = np.percentile(bt, [2.5, 97.5])
    reg("3", nom, float(b_), f"IC 95% [{lo:.3f}, {hi:.3f}] → A {'confirmada' if hi < 1 else 'rechazada'}")
# Mecanismo: Δ log E (2016 → 2023) ~ tasa de robo previa, a igual tamaño inicial. Si el robo cierra negocios, < 0.
u["dlogE"] = np.log(u.E_fijo / u.E2016)
u["log_tasa"] = np.log((u.robo1618 + 0.5) / u.E2016)
X = sm.add_constant(u[["log_tasa"]].assign(logE2016=np.log(u.E2016)))
ols = sm.OLS(u.dlogE, X).fit(cov_type="cluster", cov_kwds={"groups": u.index.map(bloques().bloque).fillna(-1)})
lo, hi = ols.conf_int().loc["log_tasa"]
reg("3", "Δ log E 2016→2023 por +1 log tasa de robo 2016–18", float(ols.params.log_tasa),
    f"IC 95% [{lo:.3f}, {hi:.3f}] (EE agrupados por bloque); duplicar la tasa de robo → {2 ** ols.params.log_tasa - 1:+.1%} en E")

# ============================================================ 4. B: composición vs controles
print("== 4. B: composición aislada de los controles")
c0, cc = cols_m0(p), cols_comp(p)
esp = {"M0": c0, "M0 + controles": c0 + CONTROLES, "M0 + composición": c0 + cc, "M1": cols_m1(p)}
dev = {k: cv_espacial(p, v)[1] for k, v in esp.items()}


def delta(a, b):
    m = {f: (dev[a][f] - dev[b][f]) / dev[a][f] for f in dev[a]}
    return float(np.mean(list(m.values()))), int(sum(v > 0 for v in m.values()))


for a, b, nota in [("M0", "M1", "ΔD de B (reproduce 04_modelos)"), ("M0", "M0 + composición", "composición sola"),
                   ("M0", "M0 + controles", "controles solos"), ("M0 + controles", "M1", "composición dada controles"),
                   ("M0 + composición", "M1", "controles dada composición")]:
    d, k = delta(a, b)
    reg("4", f"ΔD {a} → {b}", d, f"{nota}; mejora en {k}/5 folds")

# ============================================================ 5. Validación operativa de C
print("== 5. Lista 2020–2023 → ¿sigue por encima de lo esperado en 2024?")
fold = p.cvegeo.map(bloques().fold).values
bloque = p.cvegeo.map(bloques().bloque).values
pre = (p.anio <= 2023).values
p23 = p[pre].reset_index(drop=True)
g23 = ranking_m2(p23)
# Ê 2024 fuera de muestra en espacio y tiempo: cada fold se predice con un modelo entrenado en los otros 4 folds
# y solo con 2020–2023 (M2 corta el año numérico: 2024 hereda el nivel de 2023, como en 04_modelos.temporal).
mu24 = np.full(len(p), np.nan)
X, y = p[c2].values.astype(float), p.y.values
for f in range(5):
    tr, te = pre & (fold != f), ~pre & (fold == f)
    mu24[te] = ajusta_m2(X[tr], y[tr], bloque[tr]).predict(X[te])
d24 = p.loc[~pre, ["cvegeo", "y"]].assign(mu=mu24[~pre]).set_index("cvegeo")
s = contraccion(d24.y, d24.mu)
d24["rr"], d24["p_exc"] = s["rr"], s["p_exc"]
t24 = top(d24.p_exc + 1e-6 * d24.rr)
listas = {"Top 10% por riesgo 2020–23": top(g23.score), "Top 10% por conteo 2020–23": top(g23.O),
          "Riesgo, no conteo (el conteo no las ve)": top(g23.score) - top(g23.O),
          "Conteo, no riesgo": top(g23.O) - top(g23.score), "Todas las AGEB": set(d24.index)}
val = []
for nom, L in listas.items():
    x = d24.reindex(list(L))
    val.append({"lista": nom, "n": len(L), "% con O > Ê en 2024": (x.y > x.mu).mean(),
                "% con P(θ>1) > 0.8 en 2024": (x.p_exc > 0.8).mean(), "rr mediano 2024": x.rr.median(),
                "% en top 10% por riesgo 2024": len(L & t24) / len(L), "% del robo 2024": x.y.sum() / d24.y.sum(),
                "robos 2024 por encima de Ê": (x.y - x.mu).sum()})
val = pd.DataFrame(val)
print(val.round(3).to_string(index=False))
for _, r in val.iterrows():
    for k in val.columns[2:]:
        reg("5", f"{r.lista} · {k}", float(r[k]))

fig, ax = plt.subplots(figsize=(6.5, 2.6))
v = val.iloc[[0, 1, 2, 4]].iloc[::-1]
ax.barh(v.lista, v["% con O > Ê en 2024"] * 100, color=["#bbbbbb", "#e74c3c", "#3498db", "#7f3c8d"])
for i, x in enumerate(v["% con O > Ê en 2024"] * 100):
    ax.text(x + 1, i, f"{x:.0f}%", va="center", fontsize=8)
ax.set_xlim(0, 100)
ax.set_xlabel("% de las AGEB de la lista con más robo del esperado en 2024")
fig.suptitle("Zonas señaladas con 2020–2023: ¿siguen por encima de lo esperado en 2024?", x=0.02, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_MOD / "fig16_C_validacion_2024.png")

# ============================================================ salidas
pd.DataFrame(FILAS).to_csv(OUT_MOD / "robustez_C.csv", index=False, float_format="%.4f")
cols_t = ["alcaldia", "colonia", "E_fijo", "O", "Ehat", "rr", "ps_efectivo_barrio_fijo", "puesto_cv", "puesto_po"]
(OUT_MOD / "robustez_C.md").write_text(
    "# 04c · Revisión de C (salida automática)\n\n## Top 20 del ranking principal\n\n"
    "`puesto_cv`: puesto con solo robo con violencia; `puesto_po`: con personal ocupado como E (de "
    f"{len(g0):,}).\n\n" + top20[cols_t].round(2).to_markdown() + "\n\n## Validación 2024\n\n"
    + val.round(3).to_markdown(index=False) + "\n", encoding="utf-8")
print(top20[cols_t].round(2).to_string())
print("listo →", OUT_MOD / "robustez_C.csv")
