"""04e · ¿La diferencia de propensión a denunciar entre alcaldías, medida con la ENVIPE, cambia la lista de C o la
lectura de A/B? Análisis exploratorio posterior a T2b: no modifica veredictos, umbrales, datos de la app ni salidas
de 04_modelos.py. Hermano de 04b_sesgo_denuncia.py y 04d_alcaldia_denuncia.py.

Motivo: 04d no pudo medir la denuncia por alcaldía (la ENSU no pregunta por denuncia y su indicador indirecto quedó
dominado por la población flotante). La ENVIPE sí pregunta, para cada delito, en qué alcaldía ocurrió (BP1_3C) y si
se denunció ante el MP (BP1_20, BP1_21). Con eso se mide la denuncia relativa entre alcaldías por el lugar del hecho,
que es el mismo criterio con el que la FGJ registra las carpetas.

Marco: O_i = r_i · C_i (r = tasa de denuncia, C = robo real). Un factor constante en toda la CDMX no cambia ningún
ranking: solo da la magnitud ("robos reales ≈ k × carpetas", con la cifra negra de la ENVE en comercio). Lo que
puede cambiar la lista es la variación ENTRE zonas:
    r_i = r_CDMX,comercio (ENVE t3_6) × (d_a / d_CDMX) (ENVIPE) × (s_i / s_CDMX) (mezcla de tamaños, ENVE t3_8, 04b)

1. ENVIPE ediciones 2021–2026 armonizadas (delitos ocurridos en la CDMX). Denuncia por alcaldía de ocurrencia con
   EE por linealización de Taylor (estratos EST_DIS y UPM_DIS de cada edición; las UPM de la muestra se toman de
   THogar para contar también las que no aportan delitos). Prueba de heterogeneidad Q sobre el logit.
2. Contracción bayesiana empírica normal-normal del logit de d_a hacia la media de la CDMX (τ² por REML) y Monte
   Carlo de d_a: μ ~ N(μ̂, Var μ̂) y θ_a | μ ~ N(μ + (1 − B_a)(y_a − μ), (1 − B_a) s_a²).
3. Ê de M2 con offset log r_i (init_score de LightGBM, igual que 04d), reajustando el modelo con la misma validación
   espacial. Verificaciones: offset 0 reproduce el principal; offset constante no cambia nada. Se compara con la
   versión sin reajuste (Ê principal × r).
4. Lista de C contra la principal: Jaccard del top 10% (la lista de la app), cuántas del top 20 sobreviven, ρ de
   Spearman del rr, zonas que el conteo no ve retenidas y % de varianza de log rr explicada por la alcaldía. Monte
   Carlo con reajuste completo de M2 en cada sorteo. Referencias: ruido de V2b al 40% (04c) y offset ENSU (04d).
5. A: β de M0 con el offset (bootstrap de bloques, como 04b) y su dispersión en el Monte Carlo. B: IRR de los grupos
   de efectivo en M1 con el offset (estimación puntual).
6. Sensibilidades: periodo (ediciones 2021–2025 principal = delitos 2020–2024, como V2b; 2025–2026; 2021–2026), tipo
   de delito (todos; solo robos a la persona o su vivienda, BPCOD 04/05/06), denuncia vs carpeta iniciada (BP1_24),
   con y sin contracción, y alcaldía no especificada ('999').

Supuestos:
- La denuncia relativa entre alcaldías de los hogares (ENVIPE) se transfiere a los negocios: si en Coyoacán los
  hogares denuncian 1.3 veces lo que la media de la ciudad, sus negocios también. Es el supuesto clave y no es
  verificable con estos datos (la ENVE no baja de entidad). Por qué es razonable aquí: el canal que diferencia a las
  alcaldías (cercanía y trato del MP, confianza en la fiscalía local) es el mismo para hogares y negocios; lo que es
  propio de los negocios (tamaño) entra aparte con la ENVE.
- La ENVIPE ubica el delito donde ocurrió (BP1_3C), como la FGJ: no tiene la discordancia de constructo del indicador
  ENSU de 04d. Pero solo ve víctimas que residen en hogares de México: el robo a visitantes extranjeros o a negocios
  no entra.
- La denuncia de los hogares en 2020–2024 representa la de los negocios en el mismo periodo (ediciones 2021–2025,
  cuyo año de referencia es el anterior a la edición).
- El tamaño (ENVE) y la alcaldía (ENVIPE) actúan de forma multiplicativa e independiente.
- La contracción supone que los logit de denuncia de las alcaldías vienen de una normal común; τ² se trata como
  conocido (no se propaga su incertidumbre: la versión sin contracción sirve de cota).
- El Monte Carlo ignora la covarianza entre alcaldías que induce compartir UPM (un hogar puede sufrir delitos en
  varias alcaldías); es pequeña porque la mayoría de los delitos de una UPM ocurren en una sola alcaldía.

Uso: .venv/bin/python analisis/pipeline/04e_cifra_negra_alcaldia.py   (~15 min; el Monte Carlo reajusta M2 en cada sorteo)
Salidas: analisis/pipeline/outputs/modelos/cifra_negra_alcaldia.csv, cifra_negra_alcaldia.md, fig18_cifra_negra_alcaldia.png,
         data/interim/envipe_alcaldia.parquet (ENVIPE armonizada: un renglón por delito ocurrido en la CDMX)
Datos: microdatos ENVIPE descargados de
         https://www.inegi.org.mx/contenidos/programas/envipe/<AAAA>/datosabiertos/conjunto_de_datos_envipe_<AAAA>_csv.zip
       2021–2024 el 2026-10-14; 2025 y 2026 el 2026-10-03 (docs/_factibilidad/01-descargas.md).
"""
from modelado import lightgbm

lightgbm()  # en macOS puede relanzar el proceso: debe ir antes que todo lo demás

import importlib  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import optimize, special, stats  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import INTERIM, RAW, lee_panel  # noqa: E402
from modelado import (MODELOS, OUT_MOD, SEED, GRID_M2, bloques, cols_m0, cols_m1, datos, devianza,  # noqa: E402
                      jaccard, ranking_residual, top, cols_m2)

# Los procesos de joblib importan modelado/common por referencia: necesitan analisis/pipeline/ en la ruta.
os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + os.environ.get("PYTHONPATH", "")
m04b = importlib.import_module("04b_sesgo_denuncia")  # tasa por tamaño (ENVE t3_8), NB2 con offset y bootstrap

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
B_MC = int(os.environ.get("B_MC", 200))  # sorteos con reajuste completo de M2 (~1.5 s cada uno en paralelo, 14 núcleos)
ENC = RAW / "encuestas"
EDICIONES = list(range(2021, 2027))  # año de levantamiento; el año de referencia de los delitos es el anterior
PRINCIPAL = [2021, 2022, 2023, 2024, 2025]  # delitos 2020–2024 = periodo de V2b
ALCALDIAS = [f"{i:03d}" for i in range(2, 18)]  # 002 Azcapotzalco … 017 Venustiano Carranza (claves del MG2020)
# Robos a la persona o a su vivienda: los más parecidos al robo a negocio. Se excluyen el robo total de vehículo
# (01; se denuncia por el seguro) y el de autopartes (02; bajo monto, en la calle de la vivienda).
ROBOS = ["04", "05", "06"]
FILAS = []


def reg(prueba, metrica, valor, nota=""):
    FILAS.append({"prueba": prueba, "metrica": metrica, "valor": valor, "nota": nota})
    v = f"{valor:.3f}" if isinstance(valor, float) else valor
    print(f"  [{prueba}] {metrica}: {v}  {nota}")


# ============================================================ ENVIPE
def _tabla(z, patron, cols=None):
    """Lee una tabla del zip de la ENVIPE. Las ediciones cambian mayúsculas en nombres de archivo y de columna, y
    el fin de línea (\\r); latin1 lee todas (los campos que se usan son claves numéricas)."""
    n = [f for f in z.namelist() if re.search(patron, f, re.I)]
    assert len(n) == 1, (patron, n)
    t = pd.read_csv(io.BytesIO(z.read(n[0])), dtype=str, encoding="latin1",
                    usecols=(lambda c: c.strip().upper() in cols) if cols else None)
    t.columns = t.columns.str.strip().str.upper()
    return t.apply(lambda s: s.str.strip())


def _diccionario(z):
    t = _tabla(z, r"tmod_vic.*/diccionario_de_datos/.*\.csv$")
    return t.drop_duplicates("NEMONICO").set_index("NEMONICO").NOMBRE_CAMPO.str.lower()


def envipe():
    """Un renglón por delito ocurrido en la CDMX con respuesta válida a la denuncia (BP1_20 ∈ {1, 2}), con los
    datos de diseño. n_upm_est = UPM de la muestra en el estrato (de THogar), necesario para la varianza de dominio."""
    out, chequeo = [], []
    for ed in EDICIONES:
        z = zipfile.ZipFile(ENC / f"conjunto_de_datos_envipe_{ed}_csv.zip")
        dic = _diccionario(z)
        # Verificación de que los campos significan lo mismo en cada edición
        assert "ocurrencia" in dic["BP1_2C"] and "entidad" in dic["BP1_2C"], (ed, dic["BP1_2C"])
        assert "ocurrencia" in dic["BP1_3C"] and "municipio" in dic["BP1_3C"], (ed, dic["BP1_3C"])
        assert "denuncia" in dic["BP1_20"] and "denuncia" in dic["BP1_21"], ed
        assert "carpeta" in dic["BP1_24"], (ed, dic["BP1_24"])
        v = _tabla(z, r"tmod_vic.*/conjunto_de_datos/.*\.csv$",
                   {"BPCOD", "BP1_2C", "BP1_3C", "BP1_5", "BP1_20", "BP1_21", "BP1_24", "FAC_DEL", "EST_DIS",
                    "UPM_DIS"})
        h = _tabla(z, r"thogar.*/conjunto_de_datos/.*\.csv$", {"EST_DIS", "UPM_DIS"})
        n_upm = h.groupby("EST_DIS").UPM_DIS.nunique()
        assert set(zip(v.EST_DIS, v.UPM_DIS)) <= set(zip(h.EST_DIS, h.UPM_DIS)), ed
        c = v[v.BP1_2C == "09"].copy()
        n_cdmx = len(c)
        assert set(c.BP1_3C) <= set(ALCALDIAS) | {"999"}, (ed, set(c.BP1_3C) - set(ALCALDIAS))
        c = c[c.BP1_20.isin(["1", "2"])]
        chequeo.append({"edicion": ed, "delitos_cdmx": n_cdmx, "con_respuesta": len(c)})
        out.append(pd.DataFrame({
            "edicion": ed, "anio_delito": ed - 1, "cve_mun": c.BP1_3C.values, "bpcod": c.BPCOD.values,
            "lugar": c.BP1_5.values,
            # Denuncia ante el MP por la víctima o por otro integrante del hogar (BP1_21 solo se pregunta si BP1_20 = 2)
            "denuncia": ((c.BP1_20 == "1") | (c.BP1_21 == "1")).astype(int).values,
            # Carpeta iniciada: lo que registra la FGJ y lo que la ENVE llama "no cifra negra"; "no sabe" cuenta como no
            "carpeta": (((c.BP1_20 == "1") | (c.BP1_21 == "1")) & (c.BP1_24 == "1")).astype(int).values,
            "fac_del": c.FAC_DEL.astype(float).values,
            "estrato": f"{ed}-" + c.EST_DIS.values, "upm": f"{ed}-" + c.EST_DIS.values + "-" + c.UPM_DIS.values,
            "n_upm_est": c.EST_DIS.map(n_upm).values}))
    d = pd.concat(out, ignore_index=True)
    assert d.n_upm_est.notna().all() and (d.n_upm_est >= 2).all()
    return d, pd.DataFrame(chequeo)


def razon_dominio(d, dom, y="denuncia"):
    """Proporción ponderada (FAC_DEL) en un dominio y su EE por linealización de Taylor con estratos y UPM.
    z_k = w_k · 1[dominio] · (y_k − R) / X̂. Las UPM del estrato sin delitos en el dominio aportan z = 0 (de ahí
    n_upm_est). V = Σ_h n_h/(n_h − 1) Σ_j (z_hj − z̄_h)². Con varias ediciones, cada (edición, estrato) es un
    estrato: las muestras son independientes."""
    w, x = d.fac_del.values, dom.astype(float)
    X = (w * x).sum()
    R = (w * x * d[y].values).sum() / X
    z = w * x * (d[y].values - R) / X
    t = pd.DataFrame({"estrato": d.estrato.values, "upm": d.upm.values, "z": z})
    tu = t.groupby(["estrato", "upm"]).z.sum()
    ss = tu.groupby(level=0).agg(["sum", "count"])
    nh = d.groupby("estrato").n_upm_est.first().reindex(ss.index).values
    zbar = ss["sum"].values / nh
    dev2 = (tu - pd.Series(zbar, index=ss.index).reindex(tu.index.get_level_values(0)).values) ** 2
    s2 = dev2.groupby(level=0).sum().reindex(ss.index).values + (nh - ss["count"].values) * zbar ** 2
    return R, float(np.sqrt((nh / (nh - 1) * s2).sum())), int(dom.sum())


def por_alcaldia(d, y="denuncia"):
    filas = [{"cve_mun": a, **dict(zip(["d", "se", "n"], razon_dominio(d, (d.cve_mun == a).values, y)))}
             for a in ALCALDIAS]
    t = pd.DataFrame(filas).set_index("cve_mun")
    t["logit"] = special.logit(t.d)
    t["s_logit"] = t.se / (t.d * (1 - t.d))  # método delta
    return t


def contrae(t):
    """Normal-normal sobre el logit: y_a ~ N(θ_a, s_a²), θ_a ~ N(μ, τ²). τ² por REML; μ̂ ponderada."""
    y, s2 = t.logit.values, t.s_logit.values ** 2

    def reml(tau2):
        w = 1 / (s2 + tau2)
        mu = (w * y).sum() / w.sum()
        return 0.5 * (np.log(s2 + tau2).sum() + (w * (y - mu) ** 2).sum() + np.log(w.sum()))

    tau2 = optimize.minimize_scalar(reml, bounds=(0, 5), method="bounded").x
    tau2 = 0.0 if reml(0.0) <= reml(tau2) else tau2
    w = 1 / (s2 + tau2)
    mu, v_mu = (w * y).sum() / w.sum(), 1 / w.sum()
    B = s2 / (s2 + tau2)
    t = t.assign(B=B, theta=mu + (1 - B) * (y - mu), v_post=(1 - B) * s2 + B ** 2 * v_mu)
    t["d_c"] = special.expit(t.theta)
    t["d_c_lo"], t["d_c_hi"] = (special.expit(t.theta + k * np.sqrt(t.v_post)) for k in (-1.96, 1.96))
    Q = float(((y - (y / s2).sum() / (1 / s2).sum()) ** 2 / s2).sum())  # homogeneidad con la media ponderada por 1/s²
    return t, {"tau2": float(tau2), "mu": float(mu), "v_mu": float(v_mu), "Q": Q, "p_Q": float(stats.chi2.sf(Q, len(t) - 1))}


def sorteos(t, par, contraido, B, rng):
    """Matriz B × 16 de log d_a. Contraído: μ ~ N(μ̂, Var μ̂); θ_a | μ ~ N(μ + (1 − B_a)(y_a − μ), (1 − B_a) s_a²).
    Sin contracción: θ_a ~ N(y_a, s_a²) independientes."""
    y, s = t.logit.values, t.s_logit.values
    if contraido:
        mu = rng.normal(par["mu"], np.sqrt(par["v_mu"]), size=(B, 1))
        th = mu + (1 - t.B.values) * (y - mu) + rng.normal(size=(B, len(t))) * np.sqrt((1 - t.B.values) * s ** 2)
    else:
        th = y + rng.normal(size=(B, len(t))) * s
    return np.log(special.expit(th))


# ============================================================ M2 con offset (igual que 04d; 04d no se importa porque corre al cargarse)
def ajusta_m2_off(X, y, grupos, off, n_jobs=1):
    """modelado.ajusta_m2 con offset: log μ = off + c + f(X). LightGBM no hace boost_from_average cuando recibe
    init_score, así que el promedio se pone a mano (c = log Σy / Σe^off en entrenamiento); con off = 0 equivale al
    arranque por defecto y un offset constante queda absorbido por c. n_jobs = 1 para correr sorteos en paralelo."""
    lgb = lightgbm()
    rng = np.random.default_rng(SEED)
    ub = np.unique(grupos)
    val_b = rng.choice(ub, size=max(1, len(ub) // 4), replace=False)
    va = np.isin(grupos, val_b)
    base = dict(objective="poisson", learning_rate=0.03, n_estimators=2000, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=n_jobs)

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


def cv_off(X, y, fold, bloque, off):
    """Validación espacial de 04d con offset por fila: μ_oof = predict(X) · exp(off + c)."""
    mu = np.full(len(y), np.nan)
    for f in range(5):
        tr, te = fold != f, fold == f
        m, c = ajusta_m2_off(X[tr], y[tr], bloque[tr], off[tr])
        mu[te] = m.predict(X[te]) * np.exp(off[te] + c)
    return mu


def reajusta(offsets):
    """Una validación espacial completa de M2 por offset, en paralelo (un hilo por proceso)."""
    return Parallel(n_jobs=-1)(delayed(cv_off)(X2, p.y.values, FOLD, BLOQUE, np.asarray(o, float)) for o in offsets)


eta2 = lambda s, a: float(s.groupby(a).transform("mean").var() / s.var())  # noqa: E731


def metricas(g):
    """Comparación con la lista principal g0 (top 10% = lista de la app)."""
    t = top(g.score)
    t20 = set(g.score.sort_values(ascending=False).index[:20])
    return {"jaccard_top10": jaccard(T0, t), "top20_en_top20": len(T20 & t20), "top20_en_top10": len(T20 & t),
            "rho_rr": stats.spearmanr(g0.rr, g.rr.reindex(g0.index))[0],
            "invisibles_retenidas": len(INV & t) / len(INV), "jaccard_conteo_vs_riesgo": jaccard(C0, t),
            "eta2_alcaldia": eta2(np.log(g.rr), g.index.str[2:5])}


# ============================================================ insumos
p = datos("V2b")
c2 = cols_m2(p)
p["cve_mun"] = p.cvegeo.str[2:5]
X2 = p[c2].values.astype(float)
FOLD, BLOQUE = p.cvegeo.map(bloques().fold).values, p.cvegeo.map(bloques().bloque).values
g0 = pd.read_parquet(MODELOS / "riesgo_periodo.parquet").set_index("cvegeo")  # ranking principal (04_modelos)
pan = lee_panel()
nombre = pan[pan.anio == 2023].drop_duplicates("cve_mun").set_index("cve_mun").alcaldia
alc0 = g0.index.str[2:5]
T0, C0 = top(g0.score), top(g0.O)
T20 = set(g0.score.sort_values(ascending=False).index[:20])
INV = T0 - C0

print("== 1. ENVIPE: denuncia por alcaldía de ocurrencia")
ev, chequeo = envipe()
ev.to_parquet(INTERIM / "envipe_alcaldia.parquet", index=False)
for _, r in chequeo.iterrows():
    reg("1", f"edición {r.edicion} (delitos de {r.edicion - 1}) · delitos ocurridos en la CDMX / con respuesta de denuncia",
        int(r.con_respuesta), f"de {r.delitos_cdmx}")
for ed in EDICIONES:
    e = ev[ev.edicion == ed]
    R, se, n = razon_dominio(e, np.ones(len(e), bool))
    reg("1", f"denuncia CDMX · edición {ed}", R, f"EE {se:.4f}; n = {n}")
# Reproducción del cálculo previo (ediciones 2025 + 2026, todos los delitos): 8.9%, n = 8,868
e = ev[ev.edicion.isin([2025, 2026])]
R, se, n = razon_dominio(e, np.ones(len(e), bool))
reg("1", "reproducción del cálculo previo · denuncia CDMX ediciones 2025–2026", R, f"EE {se:.4f}; n = {n} (previo: 8.9%, 8,868)")

ev_p = ev[ev.edicion.isin(PRINCIPAL)]
d_cdmx, se_cdmx, n_cdmx = razon_dominio(ev_p, np.ones(len(ev_p), bool))
reg("1", "denuncia CDMX · principal (ediciones 2021–2025)", d_cdmx, f"EE {se_cdmx:.4f}; n = {n_cdmx}")
d_cdmx_c, se_c, _ = razon_dominio(ev_p, np.ones(len(ev_p), bool), "carpeta")
reg("1", "carpeta iniciada CDMX · principal", d_cdmx_c, f"EE {se_c:.4f}; cifra oculta de hogares = {1 - d_cdmx_c:.1%}")
m999 = (ev_p.cve_mun == "999").values
R9, se9, n9 = razon_dominio(ev_p, m999)
reg("1", "alcaldía no especificada ('999') · % de los delitos (ponderado)",
    float(ev_p.fac_del[m999].sum() / ev_p.fac_del.sum()), f"n = {n9}")
reg("1", "alcaldía no especificada ('999') · denuncia", R9, f"EE {se9:.4f}")
reg("1", "alcaldía no especificada ('999') · % en transporte público (BP1_5 = 6)",
    float(np.average(ev_p.lugar[m999] == "6", weights=ev_p.fac_del[m999])),
    f"resto de la CDMX: {np.average(ev_p.lugar[~m999] == '6', weights=ev_p.fac_del[~m999]):.3f}")
d_sin999, _, _ = razon_dominio(ev_p, ~m999)
reg("1", "denuncia CDMX sin '999'", d_sin999, "solo cambia la constante d_CDMX: no puede mover ningún ranking")

# Variantes de la ENVIPE: (nombre, ediciones, filtro de delito, variable)
VARIANTES = {
    "principal": (PRINCIPAL, None, "denuncia"),
    "robos 04/05/06": (PRINCIPAL, ROBOS, "denuncia"),
    "carpeta iniciada": (PRINCIPAL, None, "carpeta"),
    "ediciones 2025–2026": ([2025, 2026], None, "denuncia"),
    "ediciones 2021–2026": (EDICIONES, None, "denuncia"),
}
TAB, PAR = {}, {}
for nom, (eds, cods, yv) in VARIANTES.items():
    e = ev[ev.edicion.isin(eds)]
    if cods:
        e = e[e.bpcod.isin(cods)]
    t, par = contrae(por_alcaldia(e, yv))
    TAB[nom], PAR[nom] = t, par
    reg("2", f"{nom} · Q de heterogeneidad del logit (χ² 15 gl)", par["Q"], f"p = {par['p_Q']:.3g}")
    reg("2", f"{nom} · τ (REML, escala logit)", float(np.sqrt(par["tau2"])),
        f"contracción media B = {t.B.mean():.2f}; d̂ cruda {t.d.min():.3f}–{t.d.max():.3f}; contraída "
        f"{t.d_c.min():.3f}–{t.d_c.max():.3f}; n mínimo {int(t.n.min())}")
ta = TAB["principal"].join(nombre.rename("alcaldia"))
for k, r in ta.sort_values("d").iterrows():
    reg("2", f"denuncia · {r.alcaldia}", float(r.d),
        f"EE {r.se:.4f}; n = {int(r.n)}; contraída {r.d_c:.3f} [{r.d_c_lo:.3f}, {r.d_c_hi:.3f}]; B = {r.B:.2f}")
reg("2", "razón max/min de d_a · cruda", float(ta.d.max() / ta.d.min()))
reg("2", "razón max/min de d_a · contraída", float(ta.d_c.max() / ta.d_c.min()))
reg("2", "alcaldías con IC 95% crudo que excluye la media CDMX",
    int(((ta.d - 1.96 * ta.se > d_cdmx) | (ta.d + 1.96 * ta.se < d_cdmx)).sum()))
# 04d: el indicador ENSU (r_alc) frente a la denuncia medida directamente
r04d = pd.read_csv(OUT_MOD / "alcaldia_denuncia.csv")
r04d = r04d[r04d.metrica.str.startswith("r_alc · ")].assign(alcaldia=lambda x: x.metrica.str[8:]).set_index("alcaldia").valor
ta["r_alc_04d"] = ta.alcaldia.map(r04d)
rho, pv = stats.spearmanr(ta.d, ta.r_alc_04d)
reg("2", "Spearman d_a (ENVIPE) vs r_alc de 04d (ENSU)", float(rho), f"p = {pv:.3f}, n = 16")

# ============================================================ magnitud (ENVE)
print("== Magnitud con la ENVE")
t36 = pd.read_csv(ENC / "enve2024/conjunto_de_datos/enve2024_iii_denuncia_delito/t3_6.csv", encoding="utf-8")
t36.columns = ["nombre", "cifra"]
i = t36.index[t36.nombre == "Ciudad de México"][0]
assert t36.nombre[i + 1] == "Comercio"
CN_COM = float(t36.cifra[i + 1]) / 100
r_com = 1 - CN_COM
reg("ENVE", "cifra negra en comercio, CDMX (ENVE 2024, delitos de 2023)", CN_COM,
    f"→ ≈ {1 / r_com:.1f} delitos por carpeta")
y_tot = p.y.sum()
reg("ENVE", "carpetas V2b 2020–2024 (AGEB del modelo)", int(y_tot),
    f"→ orden de magnitud de robos reales ≈ {y_tot / r_com:,.0f} (cota superior: la cifra de la ENVE incluye robo hormiga y extorsión)")
CN_ROBO = 0.822  # ENVE 2024 t3_4, nacional: robo o asalto de bienes o dinero (no hay desglose por tipo para la CDMX)
reg("ENVE", "cifra negra de robo o asalto de bienes o dinero (nacional)", CN_ROBO,
    f"→ ≈ {1 / (1 - CN_ROBO):.1f} delitos por carpeta; robos reales V2b ≈ {y_tot / (1 - CN_ROBO):,.0f} (cota inferior)")

# ============================================================ offsets
r_tam = m04b.tasa_denuncia_tamano()  # Σ_k s_ik (1 − cifra negra_k), ENVE t3_8
s_rel = p.cvegeo.map(r_tam).values
den = pd.read_parquet(INTERIM / "denue_ageb.parquet")
den = den[den.corte == 2023]
s_cdmx = float(den.per_ocu.map(m04b.TAMANO).map({k: 1 - v / 100 for k, v in m04b.CIFRA_NEGRA_CDMX.items()}).mean())
log_tam = np.log(s_rel / s_cdmx)
reg("3", "factor de tamaño s_i / s_CDMX · P5–P95 entre AGEB", float(np.percentile(np.exp(log_tam), 5)),
    f"P95 = {np.percentile(np.exp(log_tam), 95):.3f}")


def off_alc(log_d):
    """log(d_a / d_CDMX) por fila del panel; log_d: Series por cve_mun (o arreglo en el orden de ALCALDIAS)."""
    log_d = pd.Series(np.asarray(log_d), index=ALCALDIAS)
    return p.cve_mun.map(log_d).values - np.log(d_cdmx)


off_c = off_alc(np.log(TAB["principal"].d_c))
off_principal = off_c + log_tam  # corrección completa: alcaldía (contraída) × tamaño
r_abs = r_com * np.exp(off_principal)
u = pd.Series(r_abs, index=p.cvegeo).groupby(level=0).first()
reg("3", "tasa de denuncia implícita r_i (comercio CDMX × alcaldía × tamaño) · mediana entre AGEB", float(u.median()),
    f"P5–P95 [{u.quantile(.05):.3f}, {u.quantile(.95):.3f}]: entre 1 de {1 / u.quantile(.95):.0f} y 1 de {1 / u.quantile(.05):.0f} robos llega a carpeta")

PUNTOS = {
    "verif. offset 0": np.zeros(len(p)),
    "verif. offset constante 0.7": np.full(len(p), 0.7),
    "solo tamaño (ENVE)": log_tam,
    "alcaldía cruda (sin contracción)": off_alc(np.log(TAB["principal"].d)),
    "alcaldía contraída": off_c,
    "alcaldía contraída × tamaño (corrección completa)": off_principal,
}
for nom in ["robos 04/05/06", "carpeta iniciada", "ediciones 2025–2026", "ediciones 2021–2026"]:
    PUNTOS[f"{nom} · alcaldía contraída"] = off_alc(np.log(TAB[nom].d_c))
    PUNTOS[f"{nom} · alcaldía cruda"] = off_alc(np.log(TAB[nom].d))

print(f"== 3. M2 con offset: {len(PUNTOS)} reajustes")
MU = dict(zip(PUNTOS, reajusta(PUNTOS.values())))
G = {k: ranking_residual(p, mu) for k, mu in MU.items()}
reg("3", "verif. offset 0 · jaccard top vs principal", jaccard(T0, top(G["verif. offset 0"].score)),
    f"máx |ΔÊ|/Ê = {float((G['verif. offset 0'].Ehat / g0.Ehat - 1).abs().max()):.2e} (un hilo por proceso)")
reg("3", "verif. offset constante 0.7 · máx |ΔÊ|/Ê vs offset 0",
    float(np.max(np.abs(MU["verif. offset constante 0.7"] / MU["verif. offset 0"] - 1))))
M = {}
for k, g in G.items():
    if k.startswith("verif."):
        continue
    M[k] = metricas(g)
    for mt, v in M[k].items():
        reg("3", f"{k} · {mt}", float(v))
# Sin reajuste: Ê principal × r renormalizado (equivale a dividir O entre r en el modelo, no en la contracción)
mu_nr = MU["verif. offset 0"] * np.exp(off_principal)
mu_nr *= MU["verif. offset 0"].sum() / mu_nr.sum()
M["sin reajuste (Ê × r)"] = metricas(ranking_residual(p, mu_nr))
for mt, v in M["sin reajuste (Ê × r)"].items():
    reg("3", f"sin reajuste (Ê principal × r) · {mt}", float(v))
reg("3", "sin reajuste vs reajuste · jaccard top", jaccard(top(ranking_residual(p, mu_nr).score),
                                                         top(G["alcaldía contraída × tamaño (corrección completa)"].score)))
reg("3", "% varianza de log rr explicada por la alcaldía · principal", eta2(np.log(g0.rr), alc0))

# ============================================================ 4. Monte Carlo con reajuste
print(f"== 4. Monte Carlo: {B_MC} sorteos × 3 escenarios, reajuste completo de M2")
rng = np.random.default_rng(SEED)
MC = []
ESC_MC = [("contraído", "principal", True), ("sin contracción", "principal", False),
          ("robos 04/05/06 · contraído", "robos 04/05/06", True)]
for nom, var, contraido in ESC_MC:
    ld = sorteos(TAB[var], PAR[var], contraido, B_MC, rng)
    offs = [off_alc(l) + log_tam for l in ld]
    for b, mu in enumerate(reajusta(offs)):
        MC.append({"escenario": nom, "sorteo": b, **metricas(ranking_residual(p, mu)),
                   "beta": float(m04b.ajusta(p[cols_m0(p)].values.astype(float), p.y.values, offs[b]).params[1])})
MC = pd.DataFrame(MC)
for nom, m in MC.groupby("escenario"):
    for mt in ["jaccard_top10", "top20_en_top20", "top20_en_top10", "rho_rr", "invisibles_retenidas",
               "eta2_alcaldia", "beta"]:
        lo, med, hi = np.percentile(m[mt], [2.5, 50, 97.5])
        reg("4", f"MC {nom} · {mt}", float(med), f"IC 95% [{lo:.3f}, {hi:.3f}]")

# Referencias: ruido (04c) y 04d
rc = pd.read_csv(OUT_MOD / "robustez_C.csv")
ruido = rc[rc.metrica.str.startswith("ruido puro")].assign(k=lambda x: x.metrica.str.split(" · ").str[-1]).set_index("k")
r4d = pd.read_csv(OUT_MOD / "alcaldia_denuncia.csv").set_index("metrica").valor
REF = {"ruido V2b al 40% (04c)": {"jaccard_top10": ruido.valor["jaccard_vs_principal"], "rho_rr": ruido.valor["rho_rr"],
                                   "invisibles_retenidas": ruido.valor["invisibles_retenidas"]},
       "offset ENSU (04d)": {"jaccard_top10": r4d["Ê con offset de denuncia · jaccard_vs_principal"],
                             "rho_rr": r4d["Ê con offset de denuncia · rho_rr"],
                             "invisibles_retenidas": r4d["Ê con offset de denuncia · invisibles_retenidas"],
                             "eta2_alcaldia": r4d["% varianza de log rr explicada por la alcaldía · Ê con offset"]},
       "efectos de alcaldía (04d)": {"jaccard_top10": r4d["Ê con alcaldía · jaccard_vs_principal"],
                                     "rho_rr": r4d["Ê con alcaldía · rho_rr"],
                                     "invisibles_retenidas": r4d["Ê con alcaldía · invisibles_retenidas"]}}
for k, v in REF.items():
    for mt, x in v.items():
        reg("ref", f"{k} · {mt}", float(x))

# ============================================================ ¿el exceso por alcaldía sigue a la denuncia?
print("== Patrón por alcaldía")
gc = G["alcaldía contraída × tamaño (corrección completa)"]
ta["media_log_rr"] = np.log(g0.rr).groupby(alc0).mean()
ta["media_log_rr_corr"] = np.log(gc.rr).groupby(gc.index.str[2:5]).mean()
ta["exceso_agregado"] = g0.groupby(alc0).O.sum() / g0.groupby(alc0).Ehat.sum()
ta["n_ageb"] = pd.Series(alc0).value_counts()
ta["pct_top_principal"] = pd.Series([c[2:5] for c in T0]).value_counts().reindex(ta.index, fill_value=0) / ta.n_ageb * 100
ta["pct_top_corr"] = pd.Series([c[2:5] for c in top(gc.score)]).value_counts().reindex(ta.index, fill_value=0) / ta.n_ageb * 100
for col in ["media_log_rr", "exceso_agregado", "pct_top_principal"]:
    rho, pv = stats.spearmanr(ta.d, ta[col])
    reg("patrón", f"Spearman d_a cruda vs {col} (principal)", float(rho), f"p = {pv:.3f}, n = 16")

# ============================================================ 5. A y B
print("== 5. A (β de M0) y B (IRR de M1) con el offset")
c0, c1 = cols_m0(p), cols_m1(p)
BETA = {}
for nom, off in [("sin corrección (principal)", None), ("alcaldía contraída", off_c),
                 ("alcaldía contraída × tamaño", off_principal), ("alcaldía cruda", PUNTOS["alcaldía cruda (sin contracción)"])]:
    b, boot = m04b.beta_boot(p, c0, off)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    BETA[nom] = (b, lo, hi)
    reg("5", f"β de M0 · {nom}", float(b), f"IC 95% [{lo:.3f}, {hi:.3f}]; A {'confirmada' if hi < 1 else 'rechazada'} con el criterio ex ante")
IRR = {}
for nom, off in [("sin corrección", None), ("alcaldía contraída × tamaño", off_principal)]:
    r = m04b.ajusta(p[c1].values.astype(float), p.y.values, off)
    for gname in ["s_efectivo_barrio", "s_efectivo_formal"]:
        IRR[(nom, gname[2:])] = float(np.exp(0.1 * r.params[1 + c1.index(gname)]))
        reg("5", f"IRR por +10 pp de {gname[2:]} (M1, puntual) · {nom}", IRR[(nom, gname[2:])])
pa = p.drop_duplicates("cvegeo")
barrio = pa.groupby("cve_mun").apply(lambda d: np.average(d.s_efectivo_barrio, weights=d.E))
rho, pv = stats.spearmanr(ta.d, barrio.reindex(ta.index))
reg("5", "Spearman d_a vs % de comercio de barrio en la alcaldía", float(rho), f"p = {pv:.3f}, n = 16")

# ============================================================ figura
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
ax = axs[0]
o = ta.sort_values("d")
yy = np.arange(len(o))
ax.errorbar(o.d * 100, yy + 0.15, xerr=1.96 * o.se * 100, fmt="o", color="#999", ecolor="#ccc", ms=3.5, capsize=2,
            label="cruda (IC 95%, Taylor)")
ax.errorbar(o.d_c * 100, yy - 0.15, xerr=[(o.d_c - o.d_c_lo) * 100, (o.d_c_hi - o.d_c) * 100], fmt="o",
            color="#1f4e79", ecolor="#9db4cc", ms=3.5, capsize=2, label="contraída hacia la CDMX")
ax.axvline(d_cdmx * 100, color="#b22222", ls="--", lw=0.8)
ax.set_yticks(yy, [a.replace(" de Morelos", "").replace("La Magdalena ", "M. ") for a in o.alcaldia], fontsize=7)
ax.set_xlabel("% de delitos denunciados ante el MP (ENVIPE 2021–2025)")
ax.set_title(f"Q = {PAR['principal']['Q']:.1f}, p = {PAR['principal']['p_Q']:.2f}", fontsize=8.5)
ax.legend(fontsize=6.5, frameon=False, loc="lower right")
ax = axs[1]
for nom, col in [("contraído", "#1f4e79"), ("sin contracción", "#e67e22"), ("robos 04/05/06 · contraído", "#8e44ad")]:
    ax.hist(MC[MC.escenario == nom].jaccard_top10, bins=np.linspace(0.3, 1, 36), alpha=0.6, color=col,
            label=f"Monte Carlo ENVIPE · {nom}")
for k, (col, ls) in {"ruido V2b al 40% (04c)": ("#555", ":"), "offset ENSU (04d)": ("#b22222", "--"),
                     "efectos de alcaldía (04d)": ("#2e8b57", "-.")}.items():
    ax.axvline(REF[k]["jaccard_top10"], color=col, ls=ls, lw=1, label=k)
ax.set_xlabel("Jaccard del top 10% (lista de la app) con la principal")
ax.set_ylabel("Sorteos")
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
ax = axs[2]
ax.scatter(ta.d * 100, ta.media_log_rr, color="#7f3c8d", s=14)
for _, r in ta.iterrows():
    ax.annotate(r.alcaldia.replace(" de Morelos", "").replace("La Magdalena ", "M. "), (r.d * 100, r.media_log_rr),
                fontsize=6, xytext=(3, 2), textcoords="offset points")
rho, pv = stats.spearmanr(ta.d, ta.media_log_rr)
ax.axhline(0, color="#999", lw=0.6)
ax.set_xlabel("% denunciado (ENVIPE, cruda)")
ax.set_ylabel("Media de log rr de sus AGEB (principal)")
ax.set_title(f"ρ de Spearman = {rho:.2f} (p = {pv:.2f}, n = 16)", fontsize=8.5)
fig.suptitle("¿La denuncia distinta entre alcaldías (ENVIPE) cambia la lista de prioridades?", x=0.02, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_MOD / "fig18_cifra_negra_alcaldia.png")

# ============================================================ salidas
res = pd.DataFrame(FILAS)
res.to_csv(OUT_MOD / "cifra_negra_alcaldia.csv", index=False, float_format="%.4f")
mc_res = MC.groupby("escenario").agg(**{f"{mt}_{q}": (mt, (lambda s, q=q: np.percentile(s, q)))
                                         for mt in ["jaccard_top10", "top20_en_top20", "top20_en_top10", "rho_rr",
                                                    "invisibles_retenidas", "eta2_alcaldia", "beta"]
                                         for q in (2.5, 50, 97.5)})
mcc = MC[MC.escenario == "contraído"]
mcs = MC[MC.escenario == "sin contracción"]
mcr = MC[MC.escenario == "robos 04/05/06 · contraído"]
q = lambda s: f"{np.median(s):.2f} [{np.percentile(s, 2.5):.2f}, {np.percentile(s, 97.5):.2f}]"  # noqa: E731
qi = lambda s: f"{np.median(s):.0f} [{np.percentile(s, 2.5):.0f}, {np.percentile(s, 97.5):.0f}]"  # noqa: E731
cmp = pd.DataFrame([
    *[{"comparación con la lista principal": f"ENVIPE · MC {nom} (mediana [IC 95%])", "Jaccard top 10%": q(m.jaccard_top10),
       "top 20 que sigue en el top 20": qi(m.top20_en_top20), "top 20 que sigue en el top 10%": qi(m.top20_en_top10),
       "ρ del rr": q(m.rho_rr), "zonas que el conteo no ve retenidas": q(m.invisibles_retenidas),
       "% var. log rr por alcaldía": q(m.eta2_alcaldia * 100)} for nom, m in ((n, MC[MC.escenario == n]) for n, _, _ in ESC_MC)],
    *[{"comparación con la lista principal": f"ENVIPE · {k} (puntual)", "Jaccard top 10%": f"{v['jaccard_top10']:.2f}",
       "top 20 que sigue en el top 20": f"{v['top20_en_top20']}", "top 20 que sigue en el top 10%": f"{v['top20_en_top10']}",
       "ρ del rr": f"{v['rho_rr']:.2f}", "zonas que el conteo no ve retenidas": f"{v['invisibles_retenidas']:.2f}",
       "% var. log rr por alcaldía": f"{v['eta2_alcaldia'] * 100:.1f}"} for k, v in M.items()],
    {"comparación con la lista principal": "*principal (sin corrección)*", "Jaccard top 10%": "1", "top 20 que sigue en el top 20": "20",
     "top 20 que sigue en el top 10%": "20", "ρ del rr": "1", "zonas que el conteo no ve retenidas": "1",
     "% var. log rr por alcaldía": f"{eta2(np.log(g0.rr), alc0) * 100:.1f}"},
    *[{"comparación con la lista principal": f"*referencia: {k}*", "Jaccard top 10%": f"{v['jaccard_top10']:.2f}",
       "top 20 que sigue en el top 20": "—", "top 20 que sigue en el top 10%": "20" if "04d" in k else "—",
       "ρ del rr": f"{v['rho_rr']:.2f}", "zonas que el conteo no ve retenidas": f"{v['invisibles_retenidas']:.2f}",
       "% var. log rr por alcaldía": f"{v['eta2_alcaldia'] * 100:.1f}" if "eta2_alcaldia" in v else "—"} for k, v in REF.items()],
])
t_alc = ta.assign(
    d_pct=lambda x: (x.d * 100).round(1), ee_pp=lambda x: (x.se * 100).round(1),
    d_contraida=lambda x: x.apply(lambda r: f"{r.d_c * 100:.1f} [{r.d_c_lo * 100:.1f}, {r.d_c_hi * 100:.1f}]", axis=1),
    r_rel=lambda x: (x.d_c / d_cdmx).round(2),
    robos_pct=lambda x: (TAB["robos 04/05/06"].d * 100).round(1), robos_ee=lambda x: (TAB["robos 04/05/06"].se * 100).round(1),
    robos_contraida=lambda x: (TAB["robos 04/05/06"].d_c * 100).round(1))[
    ["alcaldia", "n", "d_pct", "ee_pp", "d_contraida", "B", "r_rel", "robos_pct", "robos_ee", "robos_contraida", "r_alc_04d", "media_log_rr", "media_log_rr_corr",
     "pct_top_principal", "pct_top_corr"]].sort_values("d_pct")
sens = pd.DataFrame([{"variante ENVIPE": k, "n": int(TAB[k].n.sum()), "Q (15 gl)": round(PAR[k]["Q"], 1),
                      "p": round(PAR[k]["p_Q"], 3), "τ logit": round(np.sqrt(PAR[k]["tau2"]), 3),
                      "d̂ cruda mín–máx (%)": f"{TAB[k].d.min() * 100:.1f}–{TAB[k].d.max() * 100:.1f}",
                      "contraída mín–máx (%)": f"{TAB[k].d_c.min() * 100:.1f}–{TAB[k].d_c.max() * 100:.1f}"}
                     for k in VARIANTES])
bc = BETA["alcaldía contraída × tamaño"]
mc_j, mc_t20 = np.median(mcc.jaccard_top10), np.median(mcc.top20_en_top10)
p_ruido = float((mcc.jaccard_top10 > REF["ruido V2b al 40% (04c)"]["jaccard_top10"]).mean())
p_ruido_s = float((mcs.jaccard_top10 > REF["ruido V2b al 40% (04c)"]["jaccard_top10"]).mean())
pq = PAR["principal"]
min_med = float(MC.groupby("escenario").jaccard_top10.median().min())
j_ruido, j_dum = REF["ruido V2b al 40% (04c)"]["jaccard_top10"], REF["efectos de alcaldía (04d)"]["jaccard_top10"]
(OUT_MOD / "cifra_negra_alcaldia.md").write_text(
    "# 04e · Cifra negra diferencial entre alcaldías (ENVIPE) y la lista de C (salida automática)\n\n"
    "Exploratorio posterior a T2b; no cambia veredictos, umbrales, datos de la app ni salidas de 04_modelos.py.\n\n"
    "## Entonces qué\n\n"
    f"- **La cifra negra es enorme{', pero casi pareja entre alcaldías' if pq['p_Q'] >= 0.10 else ' y varía entre alcaldías'}.** "
    f"En la CDMX los hogares denuncian ante el MP "
    f"{d_cdmx:.1%} de los delitos (ENVIPE 2021–2025, delitos de 2020–2024). Por alcaldía la tasa va de "
    f"{ta.d.min():.1%} a {ta.d.max():.1%}; la hipótesis de que todas son iguales "
    f"{'no se rechaza' if pq['p_Q'] >= 0.10 else 'se rechaza solo marginalmente' if pq['p_Q'] >= 0.05 else 'se rechaza'} "
    f"(Q = {pq['Q']:.1f}, 15 gl, p = {pq['p_Q']:.2f}). Al separar señal de ruido (contracción), el rango queda en "
    f"{ta.d_c.min():.1%}–{ta.d_c.max():.1%}.\n"
    f"- **Corregir por esa diferencia {'casi no' if mc_j >= 0.7 else 'sí'} mueve la lista de prioridades.** Con la "
    f"corrección (alcaldía contraída × tamaño de empresa, reajustando el modelo en {B_MC} sorteos), el top 10% de la "
    f"app comparte una mediana de Jaccard {mc_j:.2f} con la lista principal y {mc_t20:.0f} de las 20 primeras zonas "
    f"siguen en el top 10%. El ruido de submuestrear V2b al 40% da un Jaccard de "
    f"{REF['ruido V2b al 40% (04c)']['jaccard_top10']:.2f}: en {p_ruido:.0%} de los sorteos la lista corregida se parece "
    "más a la principal que una réplica de ese ruido. Sin "
    f"contracción (cota alta), Jaccard mediano {np.median(mcs.jaccard_top10):.2f}; con solo robos a la persona o su "
    f"vivienda (la variante más heterogénea entre alcaldías), {np.median(mcr.jaccard_top10):.2f}. El escenario que más "
    f"mueve la lista (Jaccard mediano {min_med:.2f}) queda {'por encima' if min_med > j_ruido else 'por debajo'} del "
    f"ruido de muestreo ({j_ruido:.2f}) y {'por encima' if min_med > j_dum else 'por debajo'} de los efectos fijos de "
    f"alcaldía de 04d ({j_dum:.2f}), que absorben toda diferencia entre alcaldías. Las 20 primeras zonas siguen en el "
    f"top 10% en {(MC.top20_en_top10 == 20).mean():.0%} de los sorteos de los tres escenarios (mínimo "
    f"{MC.top20_en_top10.min():.0f} de 20).\n"
    f"- **A no cambia:** β = {bc[0]:.3f} [{bc[1]:.3f}, {bc[2]:.3f}] con la corrección (principal: "
    f"{BETA['sin corrección (principal)'][0]:.3f}). **B tampoco:** IRR por +10 pp de comercio de barrio "
    f"{IRR[('sin corrección', 'efectivo_barrio')]:.3f} → {IRR[('alcaldía contraída × tamaño', 'efectivo_barrio')]:.3f}; "
    f"de giros formales de efectivo {IRR[('sin corrección', 'efectivo_formal')]:.3f} → "
    f"{IRR[('alcaldía contraída × tamaño', 'efectivo_formal')]:.3f} (M1, puntual). La menor denuncia del comercio de "
    "barrio no es un efecto de alcaldía.\n"
    f"- **Magnitud:** con la cifra negra de la ENVE en comercio ({CN_COM:.1%}), cada carpeta de robo a negocio "
    f"representa del orden de {1 / r_com:.0f} delitos; con la de robo o asalto (nacional, {CN_ROBO:.1%}), del orden "
    f"de {1 / (1 - CN_ROBO):.0f}. Es un factor común a toda la ciudad: sirve para la narrativa, no cambia ningún "
    "ranking.\n\n"
    "## Denuncia por alcaldía de ocurrencia (ediciones 2021–2025)\n\n"
    "`d_pct` y `ee_pp`: % de delitos denunciados ante el MP (por la víctima u otro integrante del hogar) y su EE por "
    "linealización de Taylor (estratos y UPM de cada edición). `d_contraida`: contracción normal-normal del logit "
    "(REML) con IC 95% posterior; `B` = peso de la media CDMX. `r_rel` = d contraída / d CDMX (factor del offset). "
    "`robos_*`: lo mismo con solo robos a la persona o su vivienda (BPCOD 04/05/06). `r_alc_04d`: indicador ENSU de 04d. `media_log_rr*` y `pct_top_*`: principal y con la corrección completa.\n\n"
    + t_alc.round(3).to_markdown(index=False) +
    f"\n\nCDMX: {d_cdmx:.2%} (EE {se_cdmx:.2%}, n = {n_cdmx:,}); carpeta iniciada {d_cdmx_c:.2%}. τ (logit) = "
    f"{np.sqrt(pq['tau2']):.3f}.\n\n## Sensibilidades de la ENVIPE\n\n" + sens.to_markdown(index=False) +
    "\n\n## Efecto en la lista de C\n\n" + cmp.to_markdown(index=False) +
    "\n\n## β de A con la corrección (bootstrap de bloques, 1,000 réplicas)\n\n"
    + pd.DataFrame([{"escenario": k, "β": round(v[0], 3), "IC 95%": f"[{v[1]:.3f}, {v[2]:.3f}]"} for k, v in BETA.items()]).to_markdown(index=False)
    + "\n\n## Todas las métricas\n\n" + res.round(4).to_markdown(index=False) + "\n", encoding="utf-8")
print(t_alc.to_string())
print(cmp.to_string())
print("listo →", OUT_MOD / "cifra_negra_alcaldia.csv")
