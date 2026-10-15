"""04d · ¿La lista de prioridades depende de la alcaldía? ¿La diferencia entre alcaldías parece denuncia o robo real?
Análisis exploratorio posterior a T2b: no modifica veredictos, umbrales ni salidas de 04_modelos.py. Hermano de
04b_sesgo_denuncia.py y 04c_robustez_C.py.

Motivo: el Ê del ranking principal (M2) no incluye la alcaldía. Si en una alcaldía se denuncia más (fiscalía, acceso
al MP), todas sus AGEB salen con exceso sin tener más robo.

1. Ê con efectos de alcaldía (15 dummies en M2, misma validación espacial). Cota conservadora: absorbe TODA diferencia
   entre alcaldías, sea denuncia o robo real.
2. Indicador de denuncia relativa por alcaldía, 2020–2024: r_alc ∝ (robo en vía pública denunciado a la FGJ por
   habitante) / (% de hogares con víctima de robo en calle o transporte, ENSU). IC por Monte Carlo con el EE de la ENSU.
3. ¿El patrón por alcaldía del residual se parece a r_alc o al robo "real" según residentes (ENSU)? Spearman, n = 16.
4. Ê corregido solo por denuncia: M2 con offset log(r_alc) (init_score de LightGBM). Verificación: offset 0 reproduce
   el principal, offset constante no cambia nada, y una simulación con offset conocido.
5. Puesto del top 20 principal con las Ê de 1 y 4.

Supuestos (prueba 2 y 4):
- Los hogares de una alcaldía denuncian el robo en calle o transporte en la misma proporción relativa (frente a las
  demás alcaldías) en que sus negocios denuncian el robo a negocio. No verificable con estos datos.
- Discordancia de constructo: V3 (FGJ) se registra donde ocurre el hecho e incluye víctimas que no residen en la
  alcaldía; la ENSU mide la victimización de residentes, donde sea que ocurra. En alcaldías con mucha población
  flotante (Cuauhtémoc, Miguel Hidalgo, Benito Juárez) r_alc sobreestima la denuncia.
- La ENSU cubre solo la zona urbana y no pregunta por denuncia; el numerador se toma como fijo (su ruido Poisson es
  despreciable frente al de la encuesta).

Uso: .venv/bin/python analisis/pipeline/04d_alcaldia_denuncia.py   (~2 min)
Salidas: analisis/pipeline/outputs/modelos/alcaldia_denuncia.csv, alcaldia_denuncia.md, fig17_alcaldia_denuncia.png
"""
from modelado import lightgbm

lightgbm()  # en macOS puede relanzar el proceso: debe ir antes que todo lo demás

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
from scipy import stats  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import INTERIM, lee_panel  # noqa: E402
from modelado import (MODELOS, OUT_MOD, SEED, GRID_M2, bloques, cols_m2, cv_espacial, datos, devianza,  # noqa: E402
                      jaccard, ranking_residual, top)

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
B_MC = 10_000
ANIOS = list(range(2020, 2025))
FILAS = []


def reg(prueba, metrica, valor, nota=""):
    FILAS.append({"prueba": prueba, "metrica": metrica, "valor": valor, "nota": nota})
    v = f"{valor:.3f}" if isinstance(valor, float) else valor
    print(f"  [{prueba}] {metrica}: {v}  {nota}")


def compara(g_ref, g, etiqueta, prueba):
    """Igual que 04c.compara: Jaccard del top 10%, ρ de Spearman del rr y 'zonas que el conteo no ve' retenidas."""
    t_ref, t, t_cnt = top(g_ref.score), top(g.score), top(g_ref.O)
    comun = g_ref.index.intersection(g.index)
    inv = t_ref - t_cnt
    r = {"jaccard_vs_principal": jaccard(t_ref, t), "jaccard_vs_conteo": jaccard(t_cnt, t),
         "rho_rr": stats.spearmanr(g_ref.rr[comun], g.rr[comun])[0], "invisibles_retenidas": len(inv & t) / len(inv)}
    for k, v in r.items():
        reg(prueba, f"{etiqueta} · {k}", float(v))
    return r


# ------------------------------------------------------------ M2 con offset
def ajusta_m2_off(X, y, grupos, off):
    """modelado.ajusta_m2 con offset: log μ = off + c + f(X). LightGBM no hace boost_from_average cuando recibe
    init_score, así que el promedio se pone a mano (c = log Σy / Σe^off en entrenamiento); con off = 0 equivale al
    arranque por defecto y un offset constante queda absorbido por c. Devuelve (modelo, c)."""
    lgb = lightgbm()
    rng = np.random.default_rng(SEED)
    ub = np.unique(grupos)
    val_b = rng.choice(ub, size=max(1, len(ub) // 4), replace=False)
    va = np.isin(grupos, val_b)
    base = dict(objective="poisson", learning_rate=0.03, n_estimators=2000, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)

    def init(m):
        c = np.log(y[m].sum() / np.exp(off[m]).sum())
        return off[m] + c, c

    mejor = None
    i_tr, _ = init(~va)
    i_va = off[va] + np.log(y[~va].sum() / np.exp(off[~va]).sum())
    for g in GRID_M2:
        m = lgb.LGBMRegressor(**base, **g)
        m.fit(X[~va], y[~va], init_score=i_tr, eval_set=[(X[va], y[va])], eval_init_score=[i_va],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        d = devianza(y[va], m.predict(X[va]) * np.exp(i_va))
        if mejor is None or d < mejor[0]:
            mejor = (d, g, max(m.best_iteration_, 50))
    _, g, n = mejor
    i_all, c = init(np.ones(len(y), bool))
    m = lgb.LGBMRegressor(**{**base, "n_estimators": n}, **g)
    m.fit(X, y, init_score=i_all)
    return m, c


def cv_espacial_off(p, cols, off):
    """cv_espacial (modelo m2) con offset por fila. μ_oof = predict(X) · exp(off + c)."""
    fold = p.cvegeo.map(bloques().fold).values
    bloque = p.cvegeo.map(bloques().bloque).values
    X, y, off = p[cols].values.astype(float), p.y.values, np.asarray(off, float)
    mu = np.full(len(p), np.nan)
    for f in range(5):
        tr, te = fold != f, fold == f
        m, c = ajusta_m2_off(X[tr], y[tr], bloque[tr], off[tr])
        mu[te] = m.predict(X[te]) * np.exp(off[te] + c)
    return mu


def por_alc(serie, alc):
    return serie.groupby(alc).mean()


# ============================================================ insumos
p = datos("V2b")
c2 = cols_m2(p)
p["cve_mun"] = p.cvegeo.str[2:5]
g0 = pd.read_parquet(MODELOS / "riesgo_periodo.parquet").set_index("cvegeo")  # ranking principal (04_modelos)
pan = lee_panel()
at = pan[pan.anio == 2023].set_index("cvegeo")[["cve_mun", "alcaldia", "colonia", "E_fijo"]]
nombre = at.drop_duplicates("cve_mun").set_index("cve_mun").alcaldia
alc0 = g0.index.str[2:5]
T0, C0 = top(g0.score), top(g0.O)
assert (at.cve_mun == at.index.str[2:5]).all()

print("== 0. Reproducción del principal y del reparto por alcaldía")
mu0, _ = cv_espacial(p, c2, modelo="m2")
g0r = ranking_residual(p, mu0)
reg("0", "máx |Ê reproducido − Ê principal| / Ê", float((g0r.Ehat / g0.Ehat - 1).abs().max()))
reg("0", "jaccard top reproducido vs principal", jaccard(T0, top(g0r.score)))
lrr0 = np.log(g0.rr)
eta2 = lambda s, a: float(s.groupby(a).transform("mean").var() / s.var())  # noqa: E731
reg("0", "% varianza de log rr explicada por la alcaldía · principal", eta2(lrr0, alc0))
# Folds espaciales vs alcaldías: si una alcaldía cae entera en un fold, su dummy nunca vale 1 al entrenar el modelo
# que la predice y su efecto no se puede aprender fuera de muestra (vale el de la referencia).
fa = pd.crosstab(at.reindex(g0.index).cve_mun, g0.index.map(bloques().fold), normalize="index")
enteras = [nombre[k] for k in fa.index[fa.max(axis=1) > 0.999]]
reg("0", "alcaldías con todas sus AGEB en un solo fold", len(enteras), ", ".join(enteras))
reg("0", "% máximo de una alcaldía dentro de un solo fold · mediana", float(fa.max(axis=1).median()))

# ============================================================ 2. Indicador de denuncia relativa
print("== 2. Denuncia relativa por alcaldía (FGJ V3 / ENSU)")
en = pd.read_parquet(INTERIM / "ensu_alcaldia.parquet")
en = en[en.anio.isin(ANIOS)]
vic = en.dropna(subset=["pct_hog_vict_robo_calle"])
n_sem = vic.groupby("cve_mun").size()
assert (n_sem == 9).all(), n_sem  # dic-2020 + jun/dic 2021–2024 (1S-2020 no se levantó; sep-2020 excluido)
ens = pd.DataFrame({
    "vict_calle": vic.groupby("cve_mun").pct_hog_vict_robo_calle.mean(),
    "se_vict_calle": vic.groupby("cve_mun").se_hog_vict_robo_calle.apply(lambda s: np.sqrt((s ** 2).sum()) / len(s)),
    "vict_robo": vic.groupby("cve_mun").pct_hog_vict_robo.mean(),
    "atestigua": en.dropna(subset=["pct_atestigua_robos"]).groupby("cve_mun").pct_atestigua_robos.mean(),
})
# Numerador: todas las AGEB del panel (no solo E > 0), robo en vía pública 2020–2024 por 1,000 habitantes al año.
pv = pan[pan.anio.isin(ANIOS)]
num = pv.groupby("cve_mun").y_v3_via_publica.sum() / (pan[pan.anio == 2023].groupby("cve_mun").pobtot.sum() * len(ANIOS)) * 1e3
ens["v3_por_mil"] = num
ens["v2b_por_mil"] = pv.groupby("cve_mun").y_v2b.sum() / (pan[pan.anio == 2023].groupby("cve_mun").pobtot.sum() * len(ANIOS)) * 1e3
r_bruto = ens.v3_por_mil / ens.vict_calle
k_norm = np.exp(np.log(r_bruto).mean())  # r relativo: media geométrica de las 16 alcaldías = 1
ens["r_alc"] = r_bruto / k_norm
rng = np.random.default_rng(SEED)
sim = rng.normal(ens.vict_calle.values, ens.se_vict_calle.values, size=(B_MC, len(ens)))
sim = np.clip(sim, 0.5, None)  # truncado: una victimización ≤ 0 no tiene sentido (ninguna alcaldía se acerca)
r_sim = ens.v3_por_mil.values / sim
r_sim = r_sim / np.exp(np.log(r_sim).mean(axis=1, keepdims=True))  # se renormaliza en cada réplica
ens["r_lo"], ens["r_hi"] = np.percentile(r_sim, [2.5, 97.5], axis=0)
se_log = np.log(r_sim).std(axis=0)
ens["se_log_r"] = se_log
ens["alcaldia"] = nombre.reindex(ens.index)
for k, r in ens.sort_values("r_alc").iterrows():
    reg("2", f"r_alc · {r.alcaldia}", float(r.r_alc),
        f"IC 95% [{r.r_lo:.2f}, {r.r_hi:.2f}]; V3 {r.v3_por_mil:.2f}/mil hab/año; vict. calle {r.vict_calle:.1f}% (EE {r.se_vict_calle:.1f})")
lr = np.log(ens.r_alc)
Q = float((((lr - lr.mean()) / se_log) ** 2).sum())
reg("2", "Q de homogeneidad de log r_alc (χ² 15 gl)", Q, f"p = {stats.chi2.sf(Q, 15):.2g}")
reg("2", "razón max/min de r_alc", float(ens.r_alc.max() / ens.r_alc.min()))
reg("2", "CV de la victimización ENSU entre alcaldías / EE mediano relativo",
    float(ens.vict_calle.std() / ens.vict_calle.mean()), f"EE relativo mediano {float((ens.se_vict_calle / ens.vict_calle).median()):.3f}")
fuera = ((ens.r_lo > 1) | (ens.r_hi < 1)).sum()
reg("2", "alcaldías con IC 95% de r_alc que excluye 1 (media geométrica)", int(fuera))
sin_flot = ens.drop(index=["015", "016", "014"])  # sin Cuauhtémoc, Miguel Hidalgo y Benito Juárez
lr2 = np.log(sin_flot.r_alc)
Q2 = float((((lr2 - lr2.mean()) / sin_flot.se_log_r) ** 2).sum())
reg("2", "Q sin Cuauhtémoc, M. Hidalgo y B. Juárez (χ² 12 gl)", Q2, f"p = {stats.chi2.sf(Q2, 12):.2g}")
reg("2", "Spearman r_alc vs V3 por habitante (numerador)", float(stats.spearmanr(ens.r_alc, ens.v3_por_mil)[0]))
reg("2", "Spearman r_alc vs victimización ENSU (denominador)", float(stats.spearmanr(ens.r_alc, ens.vict_calle)[0]))

# ============================================================ 1. Ê con efectos de alcaldía
print("== 1. M2 + dummies de alcaldía")
dum = pd.get_dummies(p.cve_mun, prefix="alc", drop_first=True).astype(float)
q = pd.concat([p, dum], axis=1)
mu1, _ = cv_espacial(q, c2 + list(dum.columns), modelo="m2")
g1 = ranking_residual(q, mu1)
compara(g0, g1, "Ê con alcaldía", "1")
reg("1", "C propia (jaccard conteo vs riesgo) · Ê con alcaldía", jaccard(top(g1.O), top(g1.score)),
    f"principal: {jaccard(C0, T0):.3f}")
reg("1", "% varianza de log rr explicada por la alcaldía · Ê con alcaldía", eta2(np.log(g1.rr), g1.index.str[2:5]))

# ============================================================ 4. Ê corregido solo por denuncia (offset)
print("== 4. M2 con offset log(r_alc)")
off_r = np.log(p.cve_mun.map(ens.r_alc).values)
# Verificaciones del offset: (a) offset 0 ≈ principal; (b) offset constante = offset 0 (lo absorbe c);
mu_z = cv_espacial_off(p, c2, np.zeros(len(p)))
g_z = ranking_residual(p, mu_z)
reg("4", "verif. offset 0 · jaccard top vs principal", jaccard(T0, top(g_z.score)),
    f"máx |ΔÊ|/Ê = {float((g_z.Ehat / g0r.Ehat - 1).abs().max()):.2e}")
mu_k = cv_espacial_off(p, c2, np.full(len(p), 0.7))
reg("4", "verif. offset constante 0.7 · máx |ΔÊ|/Ê vs offset 0", float(np.max(np.abs(mu_k / mu_z - 1))))
# (c) simulación: y* ~ Poisson(μ0 · r*) con r* conocido por alcaldía (log r* = 2 · log r_alc, para que se note).
# Sin offset, el exceso por alcaldía (log ΣO/ΣÊ) debe seguir a log r* con pendiente ≈ 1; con offset, ≈ 0.
lrs = 2 * np.log(ens.r_alc)
qs = p.copy()
qs["y"] = np.random.default_rng(SEED + 1).poisson(mu0 * np.exp(p.cve_mun.map(lrs).values)).astype(float)
for nom, mu_s in [("sin offset", cv_espacial(qs, c2, modelo="m2")[0]),
                  ("con offset", cv_espacial_off(qs, c2, qs.cve_mun.map(lrs).values))]:
    exc = np.log(qs.assign(mu=mu_s).groupby("cve_mun").apply(lambda d: d.y.sum() / d.mu.sum()))
    reg("4", f"verif. simulación ({nom}) · pendiente de log(ΣO/ΣÊ) por alcaldía sobre log r*",
        float(np.polyfit(lrs[exc.index], exc, 1)[0]), "esperado ≈ 1 sin offset, ≈ 0 con offset")
mu4 = cv_espacial_off(p, c2, off_r)
g4 = ranking_residual(p, mu4)
compara(g0, g4, "Ê con offset de denuncia", "4")
reg("4", "C propia (jaccard conteo vs riesgo) · Ê con offset", jaccard(top(g4.O), top(g4.score)),
    f"principal: {jaccard(C0, T0):.3f}")
reg("4", "% varianza de log rr explicada por la alcaldía · Ê con offset", eta2(np.log(g4.rr), g4.index.str[2:5]))
compara(g1, g4, "offset vs dummies (ref. = dummies)", "4")

# ============================================================ tabla por alcaldía
T1, T4 = top(g1.score), top(g4.score)
tab = pd.DataFrame({"n_ageb": pd.Series(alc0).value_counts()})
for nom, t in [("top_principal", T0), ("top_p1", T1), ("top_p4", T4), ("top_conteo", C0)]:
    cnt = pd.Series([c[2:5] for c in t]).value_counts().reindex(tab.index, fill_value=0)
    tab[f"pct_{nom}"] = cnt / tab.n_ageb * 100
    tab[f"part_{nom}"] = cnt / len(t) * 100
tab["media_log_rr"] = por_alc(lrr0, alc0)
tab["media_log_rr_p1"] = por_alc(np.log(g1.rr), g1.index.str[2:5])
tab["media_log_rr_p4"] = por_alc(np.log(g4.rr), g4.index.str[2:5])
tab["exceso_agregado"] = g0.groupby(alc0).O.sum() / g0.groupby(alc0).Ehat.sum()
tab["efecto_p1"] = por_alc(np.log(g1.Ehat.reindex(g0.index) / g0.Ehat), alc0)  # log(Ê1/Ê0) medio
tab = tab.join(ens).sort_values("media_log_rr", ascending=False)
tab.index.name = "cve_mun"
for _, r in tab.iterrows():
    reg("1/4", f"% de sus AGEB en el top · {r.alcaldia}", float(r.pct_top_principal),
        f"prueba 1: {r.pct_top_p1:.1f}; prueba 4: {r.pct_top_p4:.1f}; n = {int(r.n_ageb)}")

# ============================================================ 3. ¿El patrón por alcaldía se parece a la denuncia?
print("== 3. Patrón por alcaldía del residual vs denuncia y robo según residentes (Spearman, n = 16)")
patrones = {"media log rr (principal)": tab.media_log_rr, "% de la alcaldía en el top (principal)": tab.pct_top_principal,
            "exceso agregado ΣO/ΣÊ (principal)": tab.exceso_agregado,
            "efecto de alcaldía de la prueba 1 (media log Ê1/Ê0)": tab.efecto_p1}
indic = {"r_alc (denuncia relativa)": tab.r_alc, "victimización robo en calle (ENSU)": tab.vict_calle,
         "victimización cualquier robo (ENSU)": tab.vict_robo, "atestigua robos (ENSU)": tab.atestigua,
         "V3 por habitante (FGJ, numerador de r_alc)": tab.v3_por_mil}
centrales = tab.index.isin(["014", "015", "016"])  # B. Juárez, Cuauhtémoc, M. Hidalgo: población flotante
cor = []
for pn, ps in patrones.items():
    for iname, iv in indic.items():
        for sub, m in [("n = 16", np.ones(len(tab), bool)), ("sin centrales, n = 13", ~centrales)]:
            rho, pv_ = stats.spearmanr(iv[m], ps[m])
            cor.append({"patrón del residual": pn, "indicador": iname, "muestra": sub, "rho": rho, "p": pv_})
            reg("3", f"ρ({iname}, {pn}) · {sub}", float(rho), f"p = {pv_:.3f}")
reg("3", "Spearman V3 por habitante vs V2b por habitante (n = 16)", float(stats.spearmanr(tab.v3_por_mil, tab.v2b_por_mil)[0]))
cor = pd.DataFrame(cor)

# ============================================================ 5. Top 20
top20 = g0.sort_values("score", ascending=False).head(20).join(at[["alcaldia", "colonia"]]).assign(
    puesto=lambda x: np.arange(1, 21),
    puesto_p1=lambda x: x.index.map(g1.score.rank(ascending=False)).astype(int),
    puesto_p4=lambda x: x.index.map(g4.score.rank(ascending=False)).astype(int),
    r_alc=lambda x: x.index.str[2:5].map(ens.r_alc))
corte = int(round(len(g0) * 0.10))
reg("5", "del top 20 principal, siguen en el top 10% · prueba 1", int((top20.puesto_p1 <= corte).sum()))
reg("5", "del top 20 principal, siguen en el top 10% · prueba 4", int((top20.puesto_p4 <= corte).sum()))

# ============================================================ figura
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
x = tab.r_alc
ax.errorbar(x, tab.media_log_rr, xerr=[x - tab.r_lo, tab.r_hi - x], fmt="o", color="#1f4e79", ecolor="#9db4cc",
            capsize=2, ms=4)
for k, r in tab.iterrows():
    ax.annotate(r.alcaldia.replace(" de Morelos", "").replace("La Magdalena ", "M. "), (r.r_alc, r.media_log_rr),
                fontsize=6.5, xytext=(3, 3), textcoords="offset points")
ax.set_xscale("log")
ax.axhline(0, color="#999", lw=0.6)
ax.axvline(1, color="#999", lw=0.6, ls=":")
rho, pv_ = stats.spearmanr(tab.r_alc, tab.media_log_rr)
ax.set_xlabel("Denuncia relativa r_alc (FGJ vía pública / ENSU robo en calle; IC 95%)")
ax.set_ylabel("Media de log rr de sus AGEB (ranking principal)")
ax.set_title(f"ρ de Spearman = {rho:.2f} (p = {pv_:.2f}, n = 16)", fontsize=8.5)
ax = axs[1]
o = tab.sort_values("pct_top_principal")
yy = np.arange(len(o))
ax.barh(yy + 0.27, o.pct_top_principal, height=0.27, color="#7f3c8d", label="principal")
ax.barh(yy, o.pct_top_p1, height=0.27, color="#e74c3c", label="prueba 1: dummies de alcaldía")
ax.barh(yy - 0.27, o.pct_top_p4, height=0.27, color="#3498db", label="prueba 4: offset de denuncia")
ax.set_yticks(yy, [a.replace(" de Morelos", "").replace("La Magdalena ", "M. ") for a in o.alcaldia], fontsize=7)
ax.set_xlabel("% de las AGEB de la alcaldía en el top 10% de riesgo")
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.suptitle("¿El exceso por alcaldía sigue a la denuncia? ¿La lista cambia al corregirlo?", x=0.02, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_MOD / "fig17_alcaldia_denuncia.png")

# ============================================================ salidas
pd.DataFrame(FILAS).to_csv(OUT_MOD / "alcaldia_denuncia.csv", index=False, float_format="%.4f")
t_alc = tab.assign(r_alc_ic=lambda d: d.apply(lambda r: f"{r.r_alc:.2f} [{r.r_lo:.2f}, {r.r_hi:.2f}]", axis=1))[
    ["alcaldia", "n_ageb", "pct_top_principal", "pct_top_p1", "pct_top_p4", "pct_top_conteo", "media_log_rr",
     "media_log_rr_p1", "media_log_rr_p4", "efecto_p1", "r_alc_ic", "v3_por_mil", "vict_calle", "se_vict_calle",
     "vict_robo", "atestigua"]]
res = pd.DataFrame(FILAS)
res = res[res.prueba.isin(["0", "1", "4", "5"]) | res.metrica.str.startswith(("Q ", "razón", "alcaldías con IC", "Spearman"))]
(OUT_MOD / "alcaldia_denuncia.md").write_text(
    "# 04d · Alcaldía y denuncia en C (salida automática)\n\n"
    "Exploratorio posterior a T2b; no cambia veredictos. Prueba 1: Ê de M2 con 15 dummies de alcaldía (cota "
    "conservadora: absorbe toda diferencia entre alcaldías). Prueba 4: Ê de M2 con offset log(r_alc) (corrige solo "
    "la denuncia, bajo el supuesto de que hogares y negocios de una alcaldía denuncian en la misma proporción "
    "relativa). r_alc = (robo en vía pública FGJ por habitante, 2020–2024) / (% hogares víctimas de robo en calle o "
    "transporte, ENSU, 9 semestres), normalizado a media geométrica 1; IC 95% Monte Carlo con el EE de la ENSU.\n\n"
    "## Por alcaldía\n\n`pct_top_*`: % de las AGEB de la alcaldía en el top 10% (principal, prueba 1, prueba 4, "
    "conteo). `media_log_rr*`: media de log rr de sus AGEB. `efecto_p1`: media de log(Ê prueba 1 / Ê principal). "
    "`v3_por_mil`: robo en vía pública denunciado por mil habitantes al año. `vict_calle`, `vict_robo`, `atestigua`: "
    "ENSU, % (promedio 2020–2024).\n\n"
    + t_alc.round(3).to_markdown(index=False) + "\n\n## Correlaciones (Spearman, n = 16 y sin las 3 centrales, n = 13; solo indicativas, sin corrección por comparaciones múltiples)\n\n"
    + cor.round(3).to_markdown(index=False) + "\n\n## Top 20 del ranking principal\n\n"
    f"Puesto con la Ê de la prueba 1 y de la prueba 4 (de {len(g0):,}; top 10% = puesto ≤ {corte}).\n\n"
    + top20[["puesto", "alcaldia", "colonia", "O", "Ehat", "rr", "r_alc", "puesto_p1", "puesto_p4"]].round(2).to_markdown()
    + "\n\n## Resumen de comparaciones\n\n" + res.round(4).to_markdown(index=False) + "\n", encoding="utf-8")
print(t_alc.round(2).to_string())
print(top20[["alcaldia", "colonia", "rr", "puesto_p1", "puesto_p4"]].round(2).to_string())
print("listo →", OUT_MOD / "alcaldia_denuncia.csv")
