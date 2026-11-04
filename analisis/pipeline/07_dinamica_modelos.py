"""07 · Hipótesis D, D′ y E de T6 (inseguridad → dinámica empresarial) y prueba de medición M.

Protocolo y umbrales registrados antes de correr: analisis/lab/dinamica/README.md (copia de docs/tickets/T6-dinamica-empresarial.md
§2–§4 con las aclaraciones de implementación). Los umbrales no se mueven después de ver resultados.

Entradas: data/processed/panel_dinamica.parquet (06_dinamica_datos.py), data/interim/ce_alc_sector_estrato.parquet,
data/interim/denue_alcaldia_sector.parquet, data/interim/ageb_geom.parquet.
Salidas: analisis/pipeline/outputs/dinamica/*.csv y fig2N_*.png; data/processed/modelos/dinamica_ageb.parquet;
analisis/pipeline/outputs/resultados_dinamica.md (conserva lo escrito a mano después del marcador de interpretación).

Uso: .venv/bin/python analisis/pipeline/07_dinamica_modelos.py   (~10 min)
"""
from modelado import lightgbm

lgb = lightgbm()  # en macOS puede relanzar el proceso: debe ir antes que todo lo demás

from datetime import date  # noqa: E402

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import EFECTIVO, INTERIM, PROCESSED, ROOT, SECTORES  # noqa: E402
from modelado import (MODELOS, REF_SECTOR, SEED, ajusta_nb, bloques, contraccion, cv_espacial, devianza,  # noqa: E402
                      predice)

OUT = ROOT / "analisis/pipeline/outputs/dinamica"
OUT.mkdir(parents=True, exist_ok=True)
B = 1000
UMBRAL = {"M_rho": 0.5, "M_min_ue": 30, "D_ic_sup": 1.0, "Dp_dD": 0.02, "Dp_folds": 4, "E_auc": 0.60, "E_mejora": 0.03,
          "P_signo": 0.8, "E_min_clase": 30}
GRUPOS = [*EFECTIVO, *SECTORES, "otros"]
TAMANOS = {"t0_10": "0 a 10 personas", "t11_50": "11 a 50", "t51_mas": "51 o más"}
TIPOS = {"v1": "robo a negocio (V1)", "cv": "robo a negocio con violencia", "sv": "robo a negocio sin violencia",
         "v3": "robo en vía pública (V3)"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
FILAS = []


def reg(seccion, metrica, valor, nota=""):
    FILAS.append({"seccion": seccion, "metrica": metrica, "valor": valor, "nota": nota})
    v = f"{valor:.4f}" if isinstance(valor, float) else valor
    print(f"  [{seccion}] {metrica}: {v}  {nota}")


pan = pd.read_parquet(PROCESSED / "panel_dinamica.parquet")
blq = bloques()

# ============================================================ M. ¿El DENUE mide el cambio que mide el censo?
print("== M: DENUE vs Censos Económicos (alcaldía × sector)")
ce = pd.read_parquet(INTERIM / "ce_alc_sector_estrato.parquet")
ce = ce[(ce.estrato == "total") & (ce.mun != "CDMX") & (ce.sector != "todos")].pivot_table(
    index=["mun", "sector"], columns="anio", values="UE")
dn = pd.read_parquet(INTERIM / "denue_alcaldia_sector.parquet").pivot_table(
    index=["cve_mun", "sector"], columns="corte", values="N").rename_axis(["mun", "sector"])
m = ce.join(dn, how="inner", lsuffix="_ce", rsuffix="_dn")
m.columns = ["ue19", "ue24", "n19", "n24"]
m = m[(m.ue19 >= UMBRAL["M_min_ue"]) & (m.ue24 >= UMBRAL["M_min_ue"]) & (m.n19 > 0) & (m.n24 > 0)]
m["d_ce"], m["d_dn"] = np.log(m.ue24 / m.ue19), np.log(m.n24 / m.n19)
rho_M = float(stats.spearmanr(m.d_ce, m.d_dn)[0])
M_OK = rho_M >= UMBRAL["M_rho"]
reg("M", "Spearman Δlog DENUE vs Δlog CE", rho_M, f"{len(m)} celdas alcaldía × sector con ≥ 30 UE → {'pasa' if M_OK else 'NO pasa'}")
reg("M", "Pearson", float(np.corrcoef(m.d_ce, m.d_dn)[0, 1]))
reg("M", "Δlog total CDMX · DENUE", float(np.log(m.n24.sum() / m.n19.sum())))
reg("M", "Δlog total CDMX · CE", float(np.log(m.ue24.sum() / m.ue19.sum())))
# Lo mismo con los efectos de alcaldía quitados (lo que usa D: variación dentro de alcaldía)
dm = m.groupby(level="mun")[["d_ce", "d_dn"]].transform(lambda x: x - x.mean())
rho_M_dentro = float(stats.spearmanr(dm.d_ce, dm.d_dn)[0])
reg("M", "Spearman dentro de alcaldía (complementario)", rho_M_dentro)
m.to_csv(OUT / "prueba_M.csv")
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(m.d_ce, m.d_dn, s=np.sqrt(m.ue19), alpha=0.5, color="#3498db")
lim = [min(m.d_ce.min(), m.d_dn.min()) - 0.05, max(m.d_ce.max(), m.d_dn.max()) + 0.05]
ax.plot(lim, lim, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Δ log unidades económicas, Censos Económicos 2019 → 2024")
ax.set_ylabel("Δ log establecimientos, DENUE 2019-11 → 2024-11")
ax.set_title(f"Prueba M: ¿el DENUE mide el cambio del censo?\nρ de Spearman = {rho_M:.2f} ({len(m)} celdas alcaldía × sector)",
             fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig20_prueba_M.png")
plt.close(fig)

# ============================================================ Universo y riesgo relativo por periodo
U = pan[(pan.N_2018 > 0) & (pan.N_2019 > 0) & (pan.N_2023 > 0) & pan.cvegeo.isin(blq.index)].reset_index(drop=True)
reg("datos", "AGEB en el universo", len(U), f"de {len(pan)} (fuera: sin negocios en 2018, 2019 o 2023, o sin bloque)")
COMP = [g for g in GRUPOS if g != REF_SECTOR]


def panel_robo(tipo, anios, corte):
    """AGEB × año con y, log E del corte y las variables de M1 (composición, p_grandes y controles)."""
    filas = []
    for a in anios:
        q = pd.DataFrame({"cvegeo": U.cvegeo, "anio": a, "y": U[f"y_{tipo}_{a}"].astype(float),
                          "logE": np.log(U[f"N_{corte}"]), "p_grandes": U[f"p_grandes_{corte}"],
                          "log_pob": U.log_pob, "no_residencial": U.no_residencial,
                          "log_dist": np.log(U[f"dist_km_{a}"] + 0.05)})
        for g in COMP:
            q[f"s_{g}"] = U[f"s_{g}_{corte}"].fillna(0)
        filas.append(q)
    return pd.concat(filas, ignore_index=True)


def rr_periodo(tipo, anios, corte, modelo="M1"):
    """Ê fuera de muestra (bloques espaciales) y contracción Poisson-gamma del periodo agregado."""
    p = panel_robo(tipo, anios, corte)
    fe = []
    for a in anios[1:]:
        p[f"fe_{a}"] = (p.anio == a).astype(float)
        fe.append(f"fe_{a}")
    cols = ["logE"] + fe if modelo == "M0" else ["logE"] + [f"s_{g}" for g in COMP] + \
        ["p_grandes", "log_pob", "no_residencial", "log_dist"] + fe
    mu, dev = cv_espacial(p, cols)
    g = p.assign(mu=mu).groupby("cvegeo").agg(O=("y", "sum"), Ehat=("mu", "sum")).reindex(U.cvegeo)
    s = contraccion(g.O, g.Ehat)
    g["rr"], g["a"], g["b"] = s["rr"], s["a"], s["b"]
    return g


print("== Riesgo relativo previo (2016–2018, E = DENUE 2018-11) y posterior (2021–2023, E = 2023-11)")
PRE, POST = [2016, 2017, 2018], [2021, 2022, 2023]
rr = {f"pre_{t}": rr_periodo(t, PRE, 2018) for t in TIPOS}
rr["pre_v1_M0"] = rr_periodo("v1", PRE, 2018, modelo="M0")
rr["post_v1"] = rr_periodo("v1", POST, 2023)
for k, g in rr.items():
    reg("rr", f"{k}: O total, a, b, mediana rr", float(g.O.sum()), f"a = {g.a.iloc[0]:.2f}, b = {g.b.iloc[0]:.2f}, mediana rr = {g.rr.median():.3f}")
# Riesgo relativo de cada variante por AGEB del universo: lo reusa 08_rotacion.py (D2) para usar el mismo predictor que D
pd.DataFrame({k: g.rr.values for k, g in rr.items()}, index=U.cvegeo).reset_index().to_parquet(
    MODELOS / "dinamica_rr.parquet", index=False)
reg("rr", "Spearman rr_pre V1 vs rr_post V1", float(stats.spearmanr(rr["pre_v1"].rr, rr["post_v1"].rr)[0]))
reg("rr", "Spearman rr_pre V1 (M1) vs (M0)", float(stats.spearmanr(rr["pre_v1"].rr, rr["pre_v1_M0"].rr)[0]))

# ============================================================ D. ¿La inseguridad previa anticipa la salida neta de negocios?
ALC = pd.get_dummies(U.cve_mun, prefix="alc", drop_first=True).astype(float)
CTRL = ["logN19"] + [f"s_{g}" for g in COMP] + ["p_grandes", "log_pob", "no_residencial", "log_dist"] + list(ALC.columns)


def marco(y, logN19, rr_key="pre_v1", filtro=None):
    """AGEB con y = N_2024 (o el del subgrupo), predictor z(log rr_pre) y controles de D. Aclaración 1: log N_2019
    entra como covariable libre, equivalente a offset + control."""
    d = pd.DataFrame({"cvegeo": U.cvegeo, "y": y.astype(float), "z": np.log(rr[rr_key].rr.values), "logN19": logN19,
                      "p_grandes": U.p_grandes_2019, "log_pob": U.log_pob, "no_residencial": U.no_residencial,
                      "log_dist": np.log(U.dist_km_2019 + 0.05)})
    for g in COMP:
        d[f"s_{g}"] = U[f"s_{g}_2019"].fillna(0)
    d = pd.concat([d, ALC], axis=1)
    if filtro is not None:
        d = d[filtro.values]
    d["z"] = (d.z - d.z.mean()) / d.z.std()  # estandarizado en la muestra del modelo
    return d.reset_index(drop=True)


def _ajusta_gamma(X, y):
    """Réplica bootstrap; si el ajuste no es factible (p. ej., un subgrupo pequeño sin variación en la réplica) se
    descarta como NaN y se reporta el número de réplicas válidas."""
    try:
        return ajusta_nb(X, y).params[1]
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def estima_D(d, nombre, B=B):
    """γ (IRR por +1 DE de log rr_pre) con IC 95% por bootstrap de bloques espaciales y p-valor bootstrap bilateral."""
    cols = ["z"] + CTRL
    X, y = d[cols].values.astype(float), d.y.values
    punto = float(ajusta_nb(X, y).params[1])
    grupos = {b_: g.index.values for b_, g in d.groupby(d.cvegeo.map(blq.bloque))}
    ids = np.array(list(grupos))
    rng = np.random.default_rng(SEED)
    muestras = [np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)]) for _ in range(B)]
    bt = np.array(Parallel(n_jobs=-1)(delayed(_ajusta_gamma)(X[i], y[i]) for i in muestras))
    bt = bt[np.isfinite(bt)]
    assert len(bt) >= 0.95 * B, f"{nombre}: solo {len(bt)} réplicas válidas"
    lo, hi = np.percentile(bt, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bt >= 0).mean(), (bt <= 0).mean())))
    r = {"modelo": nombre, "n": len(d), "irr": np.exp(punto), "irr_lo": np.exp(lo), "irr_hi": np.exp(hi), "p_boot": p,
         "replicas": len(bt), "veredicto": "confirmada" if np.exp(hi) < UMBRAL["D_ic_sup"] else "rechazada"}
    print(f"  {nombre}: IRR = {r['irr']:.3f} [{r['irr_lo']:.3f}, {r['irr_hi']:.3f}], p = {p:.3f}, n = {len(d)} → {r['veredicto']}")
    return r


print("== D")
d0 = marco(U.N_2024, np.log(U.N_2019))
D = estima_D(d0, "Principal: N 2019-11 → 2024-11, rr_pre V1 (M1)")
reg("D", "IRR por +1 DE de log rr_pre", D["irr"], f"IC 95% [{D['irr_lo']:.3f}, {D['irr_hi']:.3f}] → {D['veredicto']}")
m_full = ajusta_nb(d0[["z"] + CTRL].values.astype(float), d0.y.values)
reg("D", "α (NB2) del modelo principal", float(m_full.params[-1]))
reg("D", "DE de log rr_pre (escala de la IRR)", float(np.log(rr["pre_v1"].rr).std()),
    "1 DE ≈ multiplicar el riesgo relativo previo por " + f"{np.exp(np.log(rr['pre_v1'].rr).std()):.2f}")
reg("D", "Δ log N mediano 2019 → 2024", float(np.median(np.log((U.N_2024 + 0.5) / U.N_2019))))

# ============================================================ D′. ¿Sirve para predecir?
print("== D′")
fold = d0.cvegeo.map(blq.fold).values
dev = {"base": {}, "con_inseguridad": {}}
alc_falta = []
for f in range(5):
    tr, te = fold != f, fold == f
    for nom, cols in [("base", CTRL), ("con_inseguridad", ["z"] + CTRL)]:
        X = d0[cols].values.astype(float)
        dev[nom][f] = devianza(d0.y.values[te], predice(ajusta_nb(X[tr], d0.y.values[tr]), X[te]))
    alc_falta += [c for c in ALC.columns if d0.loc[tr, c].sum() == 0 and d0.loc[te, c].sum() > 0]
mej = {f: (dev["base"][f] - dev["con_inseguridad"][f]) / dev["base"][f] for f in range(5)}
dD, n_mej = float(np.mean(list(mej.values()))), int(sum(v > 0 for v in mej.values()))
Dp = "confirmada" if dD >= UMBRAL["Dp_dD"] and n_mej >= UMBRAL["Dp_folds"] else "rechazada"
reg("D′", "ΔD relativa media (base → base + inseguridad)", dD, f"mejora en {n_mej}/5 folds → {Dp}")
reg("D′", "alcaldías ausentes del entrenamiento en algún fold", len(set(alc_falta)), ", ".join(sorted(set(alc_falta))) or "ninguna")
for f in range(5):
    reg("D′", f"fold {f}: devianza base / con inseguridad", dev["base"][f], f"{dev['con_inseguridad'][f]:.4f} ({mej[f]:+.2%})")

# ============================================================ Heterogeneidad de D (Holm)
print("== Heterogeneidad")
het = []
for g in GRUPOS:
    n19 = U[f"n_{g}_2019"]
    het.append({"familia": "giro", "subgrupo": g, **estima_D(marco(U[f"n_{g}_2024"], np.log(n19.clip(lower=1)), filtro=n19 > 0), f"giro · {g}")})
for t, nom in TAMANOS.items():
    n19 = U[f"n_{t}_2019"]
    het.append({"familia": "tamaño", "subgrupo": nom, **estima_D(marco(U[f"n_{t}_2024"], np.log(n19.clip(lower=1)), filtro=n19 > 0), f"tamaño · {nom}")})
for t in ["cv", "sv", "v3"]:
    het.append({"familia": "tipo de delito (predictor)", "subgrupo": TIPOS[t],
                **estima_D(marco(U.N_2024, np.log(U.N_2019), rr_key=f"pre_{t}"), f"predictor · {TIPOS[t]}")})
het = pd.DataFrame(het)
het["p_holm"] = multipletests(het.p_boot, method="holm")[1]
het["vulnerable"] = (het.p_holm < 0.05) & (het.irr < 1)
het["mas_negocios"] = (het.p_holm < 0.05) & (het.irr > 1)
het.to_csv(OUT / "heterogeneidad_D.csv", index=False)
print(het[["familia", "subgrupo", "n", "irr", "irr_lo", "irr_hi", "p_boot", "p_holm", "vulnerable"]].round(3).to_string())

# ============================================================ Robustez de D
print("== Robustez de D")
rob = [D]
rob.append(estima_D(marco(U.N6_2024, np.log(U.N6_2019.clip(lower=1)), filtro=U.N6_2019 > 0), "Solo establecimientos de 6+ personas (supuesto 1)"))
rob += [r for _, r in het[het.familia == "tipo de delito (predictor)"].iterrows()]
rob.append(estima_D(marco(U.N_2024, np.log(U.N_2019), filtro=U.N_2019 >= 5), "Excluye AGEB con N_2019 < 5"))
rob.append(estima_D(marco(U.N_2024, np.log(U.N_2019), rr_key="pre_v1_M0"), "rr_pre con M0 (solo exposición)"))
rob.append(estima_D(marco(U.po_est_2024.round(), np.log(U.po_est_2019)), "Secundaria: personal ocupado estimado"))
rob = pd.DataFrame([{k: r[k] for k in ["modelo", "n", "irr", "irr_lo", "irr_hi", "p_boot", "veredicto"]} for r in rob])
rob["modelo"] = rob.modelo.str.replace("predictor · ", "rr_pre de ")
rob.to_csv(OUT / "robustez_D.csv", index=False)
reg("D", "consistencia del veredicto en robustez", float((rob.veredicto == D["veredicto"]).mean()))
# Aclaración 8: H3 solo si D se confirma en AGEB
H3_NOTA = ("No se corre: D se rechaza en AGEB y la regla registrada (aclaración 8) solo pide H3 para descartar MAUP en un "
           "hallazgo positivo." if D["veredicto"] == "rechazada" else "PENDIENTE: D se confirmó; la regla pide repetir en H3.")
reg("D", "malla H3", 0, H3_NOTA)

# Forest plot de heterogeneidad y robustez
fig, ax = plt.subplots(figsize=(6.8, 6.4))
filas = [("Principal", D["irr"], D["irr_lo"], D["irr_hi"], "#7f3c8d")]
FAM = {"giro": "Giro", "tamaño": "Tamaño", "tipo de delito (predictor)": "Predictor"}
filas += [(f"{FAM[r.familia]} · {r.subgrupo}", r.irr, r.irr_lo, r.irr_hi, "#e74c3c" if r.vulnerable else "#3498db")
          for r in het.itertuples()]
filas += [(r.modelo, r.irr, r.irr_lo, r.irr_hi, "#999999") for r in rob.iloc[1:].itertuples() if not r.modelo.startswith("rr_pre de ")]
for i, (nom, x, lo, hi, c) in enumerate(filas[::-1]):
    ax.errorbar(x, i, xerr=[[x - lo], [hi - x]], fmt="o", color=c, ms=4, capsize=2)
ax.set_yticks(range(len(filas)))
ax.set_yticklabels([f[0] for f in filas[::-1]], fontsize=7.5)
ax.axvline(1, color="black", lw=0.8)
ax.set_xlabel("Razón de tasas de N 2024 por +1 DE de inseguridad previa\n(IC 95%, bootstrap de bloques; < 1 = menos negocios)")
fig.suptitle("D: ¿la inseguridad 2016–2018 anticipa\nla evolución de los negocios 2018 → 2023?", fontsize=9, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig21_D_forest.png")
plt.close(fig)

# Relación cruda: Δ log N (dentro de alcaldía) por decil de rr_pre
U["dlogN"] = np.log((U.N_2024 + 0.5) / U.N_2019)
U["act"] = U.dlogN - U.groupby("cve_mun").dlogN.transform("median")
U["log_rr_pre"] = np.log(rr["pre_v1"].rr.values)
dec = U.groupby(pd.qcut(U.log_rr_pre, 10, labels=False)).agg(rr=("log_rr_pre", lambda x: np.exp(x.median())), act=("act", "median"),
                                                              q25=("act", lambda x: x.quantile(0.25)), q75=("act", lambda x: x.quantile(0.75)))
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.errorbar(range(1, 11), dec.act * 100, yerr=[(dec.act - dec.q25) * 100, (dec.q75 - dec.act) * 100], fmt="o-", color="#3498db", capsize=2)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f"{v:.2f}" for v in dec.rr], fontsize=7)
ax.set_xlabel("Decil de riesgo relativo previo (2016–2018; mediana del decil)")
ax.set_ylabel("Δ establecimientos 2019→2024\nvs mediana de la alcaldía (%)")
fig.suptitle("Cambio de negocios según la inseguridad previa (mediana y rango intercuartil)", fontsize=9, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig22_D_deciles.png")
plt.close(fig)

# ============================================================ E. Perfiles y ¿quiénes resisten?
print("== E")
rng = np.random.default_rng(SEED)
S = 4000
pre, post = rr["pre_v1"], rr["post_v1"]
th_pre = rng.gamma(pre.a.values + pre.O.values, 1 / (pre.b.values + pre.Ehat.values), size=(S, len(U)))
th_post = rng.gamma(post.a.values + post.O.values, 1 / (post.b.values + post.Ehat.values), size=(S, len(U)))
U["eje_inseg"] = np.log(post.rr.values / pre.rr.values)
U["p_inseg_sube"] = (th_post > th_pre).mean(axis=0)
# Eje actividad: Δlog N menos la mediana observada de la alcaldía; incertidumbre por remuestreo Poisson de N_2019 y N_2024
# (+0.5 evita log(0) en las réplicas)
med = U.groupby("cve_mun").dlogN.transform("median").values
n19 = rng.poisson(U.N_2019.values, size=(S, len(U)))
n24 = rng.poisson(U.N_2024.values, size=(S, len(U)))
U["p_act_sube"] = ((np.log((n24 + 0.5) / (n19 + 0.5)) - med) > 0).mean(axis=0)
del th_pre, th_post, n19, n24
P = UMBRAL["P_signo"]
sube, baja = U.p_inseg_sube >= P, U.p_inseg_sube <= 1 - P
act_sube, act_baja = U.p_act_sube >= P, U.p_act_sube <= 1 - P
U["perfil"] = np.select([sube & act_baja, sube & act_sube, baja & act_baja, baja & act_sube],
                        ["vulnerable", "resiliente", "estancamiento", "dinámica favorable"], "sin cambio claro")
cnt = U.perfil.value_counts()
for k, v in cnt.items():
    reg("E", f"AGEB por perfil · {k}", int(v), f"{v / len(U):.1%}")
reg("E", "AGEB con inseguridad claramente al alza", int(sube.sum()))
reg("E", "AGEB con inseguridad claramente a la baja", int(baja.sum()))

# Rasgos (fijados en el protocolo)
U["log_densidad"] = np.log(U.N_2019 / U.area_km2)
U["log_dist"] = np.log(U.dist_km_2019 + 0.05)
U["log_area"] = np.log(U.area_km2)
for g in COMP:
    U[f"comp_{g}"] = U[f"s_{g}_2019"].fillna(0)
RASGOS = [f"comp_{g}" for g in COMP] + ["log_densidad", "escolaridad", "pct_pea", "pct_internet", "pct_deshabitadas",
                                         "log_dist", "pct_no_residencial", "log_area"]
desc = U.groupby("perfil")[RASGOS + ["N_2019", "dlogN", "eje_inseg"]].median().T
desc["toda la ciudad"] = U[RASGOS + ["N_2019", "dlogN", "eje_inseg"]].median()
desc.to_csv(OUT / "perfiles_E.csv")
print(desc.round(3).to_string())

# Clasificador: entre las AGEB con inseguridad al alza, resiliente (1) vs vulnerable (0)
E_mues = U[U.perfil.isin(["resiliente", "vulnerable"])].reset_index(drop=True)
yE = (E_mues.perfil == "resiliente").astype(int).values
nE = {"resiliente": int(yE.sum()), "vulnerable": int((1 - yE).sum())}
reg("E", "muestra del clasificador", len(E_mues), str(nE))
ALC_E = pd.get_dummies(E_mues.cve_mun, prefix="alc").astype(float)
foldE = E_mues.cvegeo.map(blq.fold).values
auc, shap_imp, coef_E = {}, None, None
if min(nE.values()) < UMBRAL["E_min_clase"]:
    E_ver = "no medible (tamaño de muestra)"
    reg("E", "veredicto", 0, f"una clase tiene < {UMBRAL['E_min_clase']} AGEB")
else:
    def logit():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", max_iter=5000, random_state=SEED))

    def gbm():
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)

    XS = {"solo alcaldía": ALC_E, "logística: alcaldía + rasgos": pd.concat([E_mues[RASGOS], ALC_E], axis=1),
          "LightGBM: alcaldía + rasgos": pd.concat([E_mues[RASGOS], E_mues.cve_mun.astype("category")], axis=1)}
    pred = {k: np.full(len(E_mues), np.nan) for k in XS}
    for f in range(5):
        tr, te = foldE != f, foldE == f
        if len(np.unique(yE[tr])) < 2 or te.sum() == 0:
            continue
        for k, X in XS.items():
            mdl = gbm() if k.startswith("LightGBM") else logit()
            mdl.fit(X[tr], yE[tr])
            pred[k][te] = mdl.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(pred["solo alcaldía"])
    for k in XS:
        auc[k] = float(roc_auc_score(yE[ok], pred[k][ok]))
        reg("E", f"AUC fuera de muestra · {k}", auc[k])
    mejora = auc["logística: alcaldía + rasgos"] - auc["solo alcaldía"]
    E_ver = "confirmada" if auc["logística: alcaldía + rasgos"] >= UMBRAL["E_auc"] and mejora >= UMBRAL["E_mejora"] else "rechazada"
    reg("E", "mejora de AUC sobre solo alcaldía (logística, veredicto)", float(mejora), f"→ {E_ver}")
    reg("E", "mejora de AUC sobre solo alcaldía (LightGBM, sensibilidad)", float(auc["LightGBM: alcaldía + rasgos"] - auc["solo alcaldía"]))
    # Explicación: coeficientes estandarizados de la logística (muestra completa) y SHAP de LightGBM
    lg = logit().fit(XS["logística: alcaldía + rasgos"], yE)
    coef_E = pd.Series(lg[-1].coef_[0], index=XS["logística: alcaldía + rasgos"].columns).loc[RASGOS].sort_values()
    gb = gbm().fit(XS["LightGBM: alcaldía + rasgos"], yE)
    contrib = gb.predict(XS["LightGBM: alcaldía + rasgos"], pred_contrib=True)[:, :-1]
    shap_imp = pd.Series(np.abs(contrib).mean(axis=0), index=XS["LightGBM: alcaldía + rasgos"].columns).sort_values(ascending=False)
    pd.DataFrame({"coef_logistica_std": coef_E, "shap_medio_abs": shap_imp}).to_csv(OUT / "explicacion_E.csv")

# Mapa de perfiles
geo = gpd.read_parquet(INTERIM / "ageb_geom.parquet").merge(U[["cvegeo", "perfil"]], on="cvegeo", how="left")
COL = {"vulnerable": "#e74c3c", "resiliente": "#27ae60", "estancamiento": "#f39c12", "dinámica favorable": "#3498db",
       "sin cambio claro": "#e6e6e6"}
fig, ax = plt.subplots(figsize=(6, 7))
geo[geo.perfil.isna()].plot(ax=ax, color="white", edgecolor="#cccccc", lw=0.1)
for k, c in COL.items():
    sub = geo[geo.perfil == k]
    if len(sub):
        sub.plot(ax=ax, color=c, edgecolor="white", lw=0.05)
ax.legend(loc="lower left", fontsize=7, frameon=False, handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()],
          labels=[f"{k} ({cnt.get(k, 0)})" for k in COL])
ax.set_axis_off()
fig.suptitle("E: perfiles de las AGEB\ninseguridad 2016–18 → 2021–23 frente a negocios 2019 → 2024", fontsize=9, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig23_E_perfiles_mapa.png")
plt.close(fig)

# ============================================================ Salidas
cols_out = ["cvegeo", "alcaldia", "colonia", "N_2019", "N_2024", "dlogN", "act", "log_rr_pre", "eje_inseg", "p_inseg_sube",
            "p_act_sube", "perfil", "va_2018", "va_2023", "pb_2018", "pb_2023"]
U[cols_out].to_parquet(MODELOS / "dinamica_ageb.parquet", index=False)
pd.DataFrame(FILAS).to_csv(OUT / "metricas.csv", index=False)

# Caracterización monetaria (descriptiva): valor agregado imputado por perfil
mon = U.groupby("perfil")[["va_2018", "va_2023"]].sum()
mon["cambio_%"] = (mon.va_2023 / mon.va_2018 - 1) * 100
mon.loc["toda la ciudad"] = [U.va_2018.sum(), U.va_2023.sum(), (U.va_2023.sum() / U.va_2018.sum() - 1) * 100]
imp_val = pd.read_csv(OUT / "imputacion_validacion.csv")
imp_cob = pd.read_csv(OUT / "imputacion_cobertura.csv", index_col=0)


def f3(x):
    return f"{x:.3f}"


tabla_rob = rob.assign(**{"IRR [IC 95%]": [f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(rob.irr, rob.irr_lo, rob.irr_hi)]})[
    ["modelo", "n", "IRR [IC 95%]", "p_boot", "veredicto"]]
tabla_het = het.assign(**{"IRR [IC 95%]": [f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(het.irr, het.irr_lo, het.irr_hi)]})[
    ["familia", "subgrupo", "n", "IRR [IC 95%]", "p_boot", "p_holm", "vulnerable"]]
auc_txt = (f"AUC logística = {auc['logística: alcaldía + rasgos']:.3f} vs solo alcaldía = {auc['solo alcaldía']:.3f} "
           f"(LightGBM: {auc['LightGBM: alcaldía + rasgos']:.3f})" if auc else f"muestra: {nE}")
if not M_OK:
    ENTONCES = ("**Entonces qué (SEDECO).** El DENUE no reproduce el cambio que mide el Censo Económico, así que con estos datos "
                "no se puede saber si la inseguridad hace que se vayan los negocios. D, D′ y E se reportan como no medibles.")
else:
    ENTONCES = "**Entonces qué (SEDECO).** " + (
        "En 2018–2023, las zonas con más robo a negocio del esperado **no perdieron más negocios** que las demás; si acaso "
        f"ganaron un poco más (+{(D['irr'] - 1) * 100:.1f}% por desviación estándar de riesgo, un efecto muy chico). "
        if D["veredicto"] == "rechazada" and D["irr"] >= 1 else
        "En 2018–2023, las zonas con más robo a negocio del esperado no perdieron claramente más negocios. "
        if D["veredicto"] == "rechazada" else
        f"En 2018–2023, las zonas con más robo a negocio del esperado perdieron más negocios ({(D['irr'] - 1) * 100:.1f}% por "
        "desviación estándar de riesgo). ") + (
        "Las características de una zona tampoco permiten anticipar cuáles resisten un aumento del robo. "
        if E_ver != "confirmada" else "Hay rasgos que distinguen a las zonas que resisten un aumento del robo. ") + (
        "Por eso, la lista de riesgo residual (hipótesis A–C) sirve para **prevenir el robo**, no como alerta de cierres, y la "
        "resiliencia se vigila con seguimiento directo (próximo Censo Económico), no con un perfil." if D["veredicto"] == "rechazada"
        else "El riesgo residual también funciona como señal de alerta de pérdida de negocios.")
md = f"""# Resultados T6: inseguridad y dinámica empresarial (hipótesis D y E)

Generado por `analisis/pipeline/07_dinamica_modelos.py` el {date.today().isoformat()}. Protocolo y umbrales registrados antes de correr:
`analisis/lab/dinamica/README.md`. **Los umbrales no se movieron.**

**Especificación.** {len(U):,} AGEB urbanas con negocios en 2018, 2019 y 2023. Cambio de negocios: DENUE 2019-11 (Censo
Económico 2019, actividad de 2018) → 2024-11 (CE 2024, actividad de 2023). Inseguridad previa: riesgo relativo contraído de
robo a negocio agregado (V1) 2016–2018, con Ê fuera de muestra de M1 (bloques espaciales) y E = DENUE 2018-11. Controles:
log N 2019, composición 2019, % de grandes, población, distancia a estación y efectos fijos de alcaldía. Bootstrap de bloques
espaciales: {B} réplicas.

{ENTONCES}

## Veredictos

| | Hipótesis | Estadístico | Se confirma si | Resultado |
|---|---|---|---|---|
| **M** | El DENUE mide el cambio que mide el censo | ρ de Spearman = {rho_M:.3f} ({len(m)} celdas alcaldía × sector) | ρ ≥ 0.5 | **{'PASA' if M_OK else 'NO PASA → D, D′ y E no medibles'}** |
| **D. ¿Se van?** | Más inseguridad previa → peor evolución del número de negocios | IRR por +1 DE de log rr_pre = {D['irr']:.3f}, IC 95% [{D['irr_lo']:.3f}, {D['irr_hi']:.3f}] | Límite superior < 1 | **{D['veredicto'].upper()}** |
| **D′. ¿Sirve para predecir?** | La inseguridad previa mejora la predicción de ΔN | ΔD = {dD * 100:.2f}% fuera de muestra; mejora en {n_mej} de 5 *folds* | ΔD ≥ 2% **y** ≥ 4 de 5 | **{Dp.upper()}** |
| **E. ¿Quiénes resisten?** | Los rasgos distinguen resilientes de vulnerables | {auc_txt} | AUC ≥ 0.60 **y** mejora ≥ 0.03 | **{E_ver.upper()}** |

## M. Medición del cambio

{len(m)} celdas alcaldía × sector con ≥ 30 UE en ambos censos. ρ de Spearman = {rho_M:.3f}; Pearson = {np.corrcoef(m.d_ce, m.d_dn)[0, 1]:.3f};
dentro de alcaldía (complementario) ρ = {rho_M_dentro:.3f}. Δlog total: DENUE {np.log(m.n24.sum() / m.n19.sum()):+.3f}, CE
{np.log(m.ue24.sum() / m.ue19.sum()):+.3f}. Figura: `outputs/dinamica/fig20_prueba_M.png`.

## D. Inseguridad previa y evolución de los negocios

Una desviación estándar de log rr_pre equivale a multiplicar el riesgo relativo previo por {np.exp(np.log(rr['pre_v1'].rr).std()):.2f}.
α (NB2) = {m_full.params[-1]:.4f}. Cambio mediano de N 2019 → 2024 por AGEB: {np.expm1(np.median(U.dlogN)) * 100:+.1f}%.

### Robustez

{tabla_rob.to_markdown(index=False, floatfmt='.3f')}

Malla H3: {H3_NOTA}

### Heterogeneidad (Holm sobre {len(het)} pruebas)

"Vulnerable" = IRR < 1 con p de Holm < 0.05.

{tabla_het.to_markdown(index=False, floatfmt='.3f')}

Figuras: `fig21_D_forest.png`, `fig22_D_deciles.png`.

## D′. Capacidad predictiva

| Fold | Devianza base | Devianza + inseguridad | Mejora |
|---|---|---|---|
""" + "\n".join(f"| {f} | {dev['base'][f]:.4f} | {dev['con_inseguridad'][f]:.4f} | {mej[f]:+.2%} |" for f in range(5)) + f"""

## E. Perfiles

Ejes: inseguridad = log(rr 2021–2023 / rr 2016–2018), cada uno relativo a lo esperado en su periodo; actividad = Δlog N
2019 → 2024 menos la mediana de la alcaldía. Una AGEB entra a un cuadrante si P(signo) ≥ 0.8 en ambos ejes.

| Perfil | AGEB |
|---|---|
""" + "\n".join(f"| {k} | {v} ({v / len(U):.1%}) |" for k, v in cnt.items()) + f"""

Medianas de los rasgos por perfil (`perfiles_E.csv`):

{desc.to_markdown(floatfmt='.3f')}

""" + (f"""Clasificador (resiliente = 1, vulnerable = 0; {nE}). AUC fuera de muestra (bloques espaciales):

| Modelo | AUC |
|---|---|
""" + "\n".join(f"| {k} | {v:.3f} |" for k, v in auc.items()) + f"""

Coeficientes estandarizados de la logística (muestra completa; + = más probable resiliente) y SHAP medio absoluto de LightGBM:

{pd.DataFrame({'coef_logistica_std': coef_E, 'shap_medio_abs': shap_imp.reindex(coef_E.index)}).to_markdown(floatfmt='.3f')}
""" if auc else "") + f"""
Mapa: `fig23_E_perfiles_mapa.png`.

## Caracterización monetaria (descriptiva, no prueba hipótesis)

Valor agregado censal bruto imputado (millones de pesos de 2023; 2018 llevado con el INPC, factor 1.290).

{mon.to_markdown(floatfmt='.1f')}

**Validación de la imputación.** La imputación directa (supuesto 4) **no** reproduce el censo: la suma por alcaldía da una
mediana de {imp_val[imp_val.anio == 2018].razon_va_directa.median():.2f}× (2018) y {imp_val[imp_val.anio == 2023].razon_va_directa.median():.2f}× (2023) el valor
agregado del CE, porque el DENUE tiene más establecimientos de 251+ personas que el censo y las celdas suprimidas toman medias
de niveles más agregados. Se calibró a los totales del CE por alcaldía × sector (y el resto suprimido, al total de la alcaldía);
después de calibrar, la suma por alcaldía reproduce el censo por construcción, así que eso ya no es validación. Lo que sigue
dependiendo del supuesto es la **distribución entre AGEB dentro de la alcaldía**. Cobertura por nivel de imputación
(% de establecimientos):

{(imp_cob * 100).to_markdown(floatfmt='.1f')}

## Supuestos verificados

| Supuesto | Verificación | Resultado |
|---|---|---|
| 1. Cambio de método parejo dentro de la alcaldía | Prueba M (ρ = {rho_M:.3f}; dentro de alcaldía {rho_M_dentro:.3f}); efectos fijos de alcaldía; D con solo 6+ personas | Ver M y robustez |
| 2. Asociación, no causalidad | Declarado; el predictor se mide antes del cambio | — |
| 3. Robo agregado 2016–2018 comparable | V1 y totales con/sin violencia (la reclasificación de 2020 movió subtipos) | Declarado (factibilidad §3.1) |
| 4. Imputación monetaria | Suma por alcaldía vs CE | **Falla en su forma directa**; se calibra y se usa solo como descripción |
| 5. Un solo cambio con la pandemia dentro | Declarado | No separable |

"""
MARCA = "<!-- INTERPRETACION: lo que sigue se escribe a mano y 07_dinamica_modelos.py lo conserva -->"
ruta = ROOT / "analisis/pipeline/outputs/resultados_dinamica.md"
previo = ruta.read_text().split(MARCA, 1)[1] if ruta.exists() and MARCA in ruta.read_text() else "\n## Interpretación\n\n(pendiente)\n"
ruta.write_text(md + "\n" + MARCA + previo)
print("listo →", ruta)
