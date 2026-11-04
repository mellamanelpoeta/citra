"""Maquinaria de modelado (T2b). Promovida de analisis/lab/lab_comun.py sin cambios de método; analisis/lab/ se conserva intacto
como registro de los experimentos de especificación. Diseño: docs/5.Diseno-Experimental.md §4–§7.

- datos(): panel listo para modelar (AGEB × año) con la composición de la lista de giros elegida.
- NB2 (M0/M1) con statsmodels; M2 LightGBM Poisson con grid pequeño y validación interna por bloques.
- Validación espacial por bloques (k-means → 30 bloques → 5 folds) y métricas (devianza, MAE, capture rate).
- Bootstrap de bloques espaciales para el IC de β.
- Contracción Poisson-gamma (bayes empírico) → rr y P(θ > 1).
- panel_h3(): el mismo panel sobre malla hexagonal H3 resolución 8 (MAUP).
"""
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, special, stats
from sklearn.cluster import KMeans

from common import CANDIDATOS_EFECTIVO, CORTE_FIJO, EFECTIVO, INTERIM, LISTA_TENTATIVA, ROOT, SECTORES, GRANDES, \
    clasifica, lee_panel

warnings.filterwarnings("ignore")
SEED = 42
OUT_MOD = ROOT / "analisis/pipeline/outputs/modelos"
OUT_MOD.mkdir(parents=True, exist_ok=True)
MODELOS = ROOT / "data/processed/modelos"
MODELOS.mkdir(parents=True, exist_ok=True)

Y = {"V1": "y_v1_negocio", "V2": "y_v2_sin_farderos", "V2b": "y_v2b", "V3": "y_v3_via_publica"}
# Periodos por variante (decisión S1/S4, docs/5.Diseno-Experimental.md §3.1)
ANIOS_VAR = {"V1": list(range(2019, 2025)), "V2": list(range(2020, 2025)), "V2b": list(range(2020, 2025)),
             "V3": list(range(2019, 2025))}
PRINCIPAL = "V2b"
LISTAS = {"a": {"efectivo": LISTA_TENTATIVA}, "b": EFECTIVO, "c": {}}  # (b) es la lista final
REF_SECTOR = "serv_prof_fin"  # categoría de referencia de la partición (se omite en el diseño)
CONTROLES = ["p_grandes", "log_pob", "no_residencial", "log_dist"]


# ------------------------------------------------------------------ LightGBM en macOS
def lightgbm():
    """Importa LightGBM. En macOS la rueda de PyPI enlaza libomp de Homebrew; si no está, se usa la copia que
    trae scikit-learn (sin instalar nada fuera del .venv) relanzando el proceso con DYLD_FALLBACK_LIBRARY_PATH."""
    try:
        import lightgbm as lgb
        return lgb
    except OSError:
        import sklearn
        libs = str(Path(sklearn.__file__).parent / ".dylibs")
        if sys.platform != "darwin" or os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") == libs:
            raise
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "DYLD_FALLBACK_LIBRARY_PATH": libs})


# ------------------------------------------------------------------ datos
@lru_cache(maxsize=None)
def _denue():
    e = pd.read_parquet(INTERIM / "denue_ageb.parquet")
    e["cand"] = clasifica(e.codigo_act, CANDIDATOS_EFECTIVO, resto="")
    e["sector"] = clasifica(e.codigo_act, SECTORES)
    return e


def _grupo(e, lista):
    g = e.sector.copy()
    for nombre, cands in LISTAS[lista].items():
        g[e.cand.isin(cands)] = nombre
    return g


@lru_cache(maxsize=None)
def composicion(lista):
    """Partición sectorial por (corte, cvegeo) con los grupos de efectivo de la lista. Proporciones que suman 1."""
    e = _denue()
    tab = pd.crosstab([e.corte, e.cvegeo], _grupo(e, lista))
    return tab.div(tab.sum(axis=1), axis=0).add_prefix("s_")


def datos(y=PRINCIPAL, e="fijo", lista="b", anios=None):
    """Panel listo para modelar: AGEB con E_fijo > 0 (supuesto declarado: 12 AGEB sin establecimientos se
    excluyen), columnas y, logE, composición y controles."""
    anios = anios or ANIOS_VAR[y]
    p = lee_panel()
    p = p[p.anio.isin(anios) & (p.E_fijo > 0)].copy()
    if e == "nov":
        p = p[p.E_nov > 0]
    p["y"] = p[Y[y]].astype(float)
    p["logE"] = np.log(p[f"E_{e}"])
    p["E"] = p[f"E_{e}"]
    p["p_grandes"] = p[f"p_grandes_{e}"]
    comp = composicion(lista)
    corte = p.anio if e == "nov" else pd.Series(CORTE_FIJO, index=p.index)
    c = comp.reindex(pd.MultiIndex.from_arrays([corte.values, p.cvegeo.values])).fillna(0)
    c.index = p.index
    p = pd.concat([p, c], axis=1)
    # Supuesto: la distancia entra en log (rendimientos decrecientes); +50 m evita log(0) en AGEB con estación.
    p["log_dist"] = np.log(p.dist_estacion_km + 0.05)
    return p.reset_index(drop=True)


def cols_comp(p):
    return [c for c in p.columns if c.startswith("s_") and c != f"s_{REF_SECTOR}"]


def cols_m0(p):
    return ["logE"] + _fe(p)


def cols_m1(p):
    return ["logE"] + cols_comp(p) + [c for c in CONTROLES if c in p] + _fe(p)


def cols_m2(p):
    """M2 usa las mismas variables que M1; el año entra como numérico (los árboles lo cortan como efecto fijo)."""
    return ["logE"] + cols_comp(p) + [c for c in CONTROLES if c in p] + ["anio"]


def _fe(p):
    for a in sorted(p.anio.unique())[1:]:
        p[f"fe_{a}"] = (p.anio == a).astype(float)
    return [f"fe_{a}" for a in sorted(p.anio.unique())[1:]]


# ------------------------------------------------------------------ bloques espaciales
def asigna_bloques(xy, k=30, n_folds=5):
    """k-means sobre centroides → k bloques contiguos, repartidos en folds equilibrados por número de unidades."""
    a = pd.DataFrame(xy, columns=["x", "y"])
    a["bloque"] = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(a[["x", "y"]])
    tam = a.bloque.value_counts().sort_values(ascending=False)
    carga, asign = np.zeros(n_folds), {}
    for b_, n in tam.items():
        f = int(np.argmin(carga))
        asign[b_] = f
        carga[f] += n
    a["fold"] = a.bloque.map(asign)
    return a[["bloque", "fold"]]


@lru_cache(maxsize=None)
def bloques():
    """cvegeo → (bloque, fold) para las AGEB con E_fijo > 0 (idéntico a analisis/lab/lab_comun.py)."""
    p = lee_panel()
    a = p[(p.anio == CORTE_FIJO) & (p.E_fijo > 0)][["cvegeo", "x", "y"]].reset_index(drop=True)
    b = asigna_bloques(a[["x", "y"]].values)
    b.index = a.cvegeo
    return b


# ------------------------------------------------------------------ modelos
def ajusta_nb(X, y):
    """NB2 por máxima verosimilitud (alpha estimado). Arranca desde el Poisson GLM."""
    Xc = sm.add_constant(X, has_constant="add")
    start = sm.GLM(y, Xc, family=sm.families.Poisson()).fit().params
    try:
        m = sm.NegativeBinomial(y, Xc, loglike_method="nb2").fit(
            start_params=np.append(start, 0.5), method="newton", maxiter=200, disp=0)
    except np.linalg.LinAlgError:  # Hessiano singular (p. ej., un remuestreo bootstrap sin variación): BFGS
        m = None
    if m is None or not m.mle_retvals.get("converged", True) or not np.isfinite(m.params).all():
        m = sm.NegativeBinomial(y, Xc, loglike_method="nb2").fit(
            start_params=np.append(start, 0.5), method="bfgs", maxiter=2000, disp=0)
    return m


def predice(m, X):
    return np.asarray(m.predict(sm.add_constant(X, has_constant="add")))


GRID_M2 = [{"num_leaves": nl, "min_child_samples": mc} for nl in (7, 15, 31) for mc in (20, 60)]


def ajusta_m2(X, y, grupos):
    """LightGBM Poisson. Grid pequeño (6 combinaciones) elegido con validación interna por bloques: se aparta
    1/4 de los bloques de entrenamiento y se usa early stopping sobre ellos. Se reentrena con todo el
    entrenamiento y el número de árboles elegido."""
    lgb = lightgbm()
    rng = np.random.default_rng(SEED)
    ub = np.unique(grupos)
    val_b = rng.choice(ub, size=max(1, len(ub) // 4), replace=False)
    va = np.isin(grupos, val_b)
    base = dict(objective="poisson", learning_rate=0.03, n_estimators=2000, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)
    mejor = None
    for g in GRID_M2:
        m = lgb.LGBMRegressor(**base, **g)
        m.fit(X[~va], y[~va], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        d = devianza(y[va], m.predict(X[va]))
        if mejor is None or d < mejor[0]:
            mejor = (d, g, max(m.best_iteration_, 50))
    _, g, n = mejor
    m = lgb.LGBMRegressor(**{**base, "n_estimators": n}, **g)
    m.fit(X, y)
    m.config_ = {**g, "n_estimators": n}
    return m


def cv_espacial(p, cols, modelo="nb", fold=None, bloque=None):
    """Predicciones fuera de muestra por fold espacial. Devuelve (mu_oof, devianza media por fold)."""
    fold = p.cvegeo.map(bloques().fold).values if fold is None else fold
    bloque = p.cvegeo.map(bloques().bloque).values if bloque is None else bloque
    mu = np.full(len(p), np.nan)
    dev = {}
    for f in range(5):
        tr, te = fold != f, fold == f
        X, y = p[cols].values.astype(float), p.y.values
        if modelo == "nb":
            m = ajusta_nb(X[tr], y[tr])
            mu[te] = predice(m, X[te])
        else:
            m = ajusta_m2(X[tr], y[tr], bloque[tr])
            mu[te] = m.predict(X[te])
        dev[f] = devianza(y[te], mu[te])
    return mu, dev


def devianza(y, mu):
    """Devianza Poisson media."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (t - (y - mu))))


def capture(y, score, q=0.10):
    """Fracción de los delitos que cae en las unidades del top q según `score`."""
    y, score = np.asarray(y, float), np.asarray(score, float)
    n = int(round(len(y) * q))
    return float(y[np.argsort(-score)[:n]].sum() / y.sum())


def capture_anual(p, mu, q=0.10):
    """Capture rate del top q calculado dentro de cada año y promediado."""
    d = p[["anio", "y"]].assign(mu=mu)
    return float(np.mean([capture(g.y, g.mu, q) for _, g in d.groupby("anio")]))


def bootstrap_beta(p, cols, bloque, B=500, idx=1):
    """IC 95% percentil de un coeficiente por bootstrap de bloques espaciales: se remuestrean bloques completos
    (con todas sus unidades y años) para respetar la autocorrelación espacial."""
    rng = np.random.default_rng(SEED)
    grupos = {b_: g.index.values for b_, g in p.groupby(bloque)}
    ids = np.array(list(grupos))
    X, y = p[cols].values.astype(float), p.y.values
    out = []
    for _ in range(B):
        i = np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)])
        out.append(ajusta_nb(X[i], y[i]).params[idx])
    out = np.array(out)
    assert np.isfinite(out).mean() > 0.99, f"{(~np.isfinite(out)).sum()} réplicas bootstrap no convergieron"
    return out[np.isfinite(out)]


# ------------------------------------------------------------------ contracción Poisson-gamma
def contraccion(O, E):
    """θ ~ Gamma(a, b); O | θ ~ Poisson(θ·E). (a, b) por MV marginal (binomial negativa).
    Devuelve dict con a, b, rr (media posterior) y p_exc = P(θ > 1 | O)."""
    O, E = np.asarray(O, float), np.asarray(E, float)

    def nll(t):
        a, b = np.exp(t)
        return -np.sum(special.gammaln(a + O) - special.gammaln(a) - special.gammaln(O + 1)
                       + a * np.log(b / (b + E)) + O * np.log(E / (b + E)))

    r = optimize.minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    a, b = np.exp(r.x)
    return {"a": a, "b": b, "rr": (a + O) / (b + E), "p_exc": stats.gamma.sf(1, a + O, scale=1 / (b + E))}


def ranking_residual(p, mu, unidad="cvegeo"):
    """Suma del periodo por unidad de O y Ê (fuera de muestra) → contracción."""
    g = p.assign(mu=mu).groupby(unidad).agg(O=("y", "sum"), Ehat=("mu", "sum"))
    s = contraccion(g.O, g.Ehat)
    g["rr"], g["p_exc"] = s["rr"], s["p_exc"]
    g["score"] = g.p_exc + 1e-6 * g.rr  # desempate por riesgo contraído
    return g


def riesgo_anual(p, mu, unidad="cvegeo"):
    """Contracción por año (a, b propios de cada año). DataFrame unidad × año con O, Ehat, rr, p_exc."""
    out = []
    for a, g in p.assign(mu=mu).groupby("anio"):
        s = contraccion(g.y, g.mu)
        out.append(pd.DataFrame({unidad: g[unidad].values, "anio": a, "O": g.y.values, "Ehat": g.mu.values,
                                 "rr": s["rr"], "p_exc": s["p_exc"]}))
    return pd.concat(out, ignore_index=True)


def top(s, q=0.10):
    return set(s.sort_values(ascending=False).index[: int(round(len(s) * q))])


def jaccard(a, b):
    return len(a & b) / len(a | b)


def persistencia(p, mu, unidad="cvegeo"):
    """ρ de Spearman del riesgo contraído entre años consecutivos (contracción por año). Devuelve (media, por par)."""
    r = riesgo_anual(p, mu, unidad).pivot(index=unidad, columns="anio", values="rr")
    anios = list(r.columns)
    pares = {f"{a}-{b}": stats.spearmanr(r[a], r[b], nan_policy="omit")[0] for a, b in zip(anios, anios[1:])}
    return float(np.mean(list(pares.values()))), pares


# ------------------------------------------------------------------ malla H3 (MAUP)
H3_RES = 8


@lru_cache(maxsize=None)
def _h3_puntos():
    """Celda H3 de cada delito (AGEB urbana) y de cada establecimiento del corte fijo."""
    import h3
    d = pd.read_parquet(INTERIM / "delitos_ageb.parquet")
    d = d[d.tipo_asig == "URBANA"].copy()
    d["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(d.latitud, d.longitud)]
    e = _denue()
    e = e[e.corte == CORTE_FIJO].copy()
    e["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(e.latitud, e.longitud)]
    return d, e


@lru_cache(maxsize=None)
def _h3_geo(celdas):
    """Polígonos H3 (LCC del MG2020) y pesos de interpolación areal desde las AGEB."""
    import geopandas as gpd
    import h3
    from shapely.geometry import Polygon
    ag = gpd.read_parquet(INTERIM / "ageb_geom.parquet")
    crs = ag.crs
    hx = gpd.GeoDataFrame({"h3": list(celdas)},
                          geometry=[Polygon([(lo, la) for la, lo in h3.cell_to_boundary(c)]) for c in celdas],
                          crs="EPSG:4326").to_crs(crs)
    ov = gpd.overlay(ag[["cvegeo", "geometry"]], hx, how="intersection", keep_geom_type=True)
    ov["a"] = ov.area
    ov["w_ageb"] = ov.a / ov.cvegeo.map(ag.set_index("cvegeo").area)   # fracción de la AGEB en la celda
    ov["w_hex"] = ov.a / ov.h3.map(hx.set_index("h3").area)            # fracción de la celda cubierta por la AGEB
    cen = hx.geometry.centroid
    return ov[["cvegeo", "h3", "w_ageb", "w_hex"]], pd.DataFrame({"h3": hx.h3, "cx": cen.x, "cy": cen.y})


def panel_h3(y=PRINCIPAL, lista="b"):
    """Panel H3 res 8 × año con las mismas variables que el de AGEB.
    Delitos y establecimientos se reasignan desde sus coordenadas (no desde la AGEB). Población (extensiva) y
    distancia a estación (intensiva) se interpolan por área desde las AGEB: supuesto de distribución uniforme
    dentro de cada AGEB. Universo: celdas con E > 0 dentro de AGEB urbanas."""
    anios = ANIOS_VAR[y]
    d, e = _h3_puntos()
    # misma definición de la variable objetivo que 02_features.py
    neg, far = d.objetivo == "negocio", d.farderos == 1
    defs = {"V1": neg, "V2": neg & ~far, "V2b": neg & ~far & (d.grupo != "negocio_sv_dentro"),
            "V3": d.objetivo == "via_publica"}
    dy = d[defs[y] & d.anio_hecho.isin(anios)]
    E = e.groupby("h3").size().rename("E")
    celdas = E.index
    idx = pd.MultiIndex.from_product([celdas, anios], names=["h3", "anio"])
    yy = dy.groupby(["h3", "anio_hecho"]).size().rename_axis(["h3", "anio"]).reindex(idx, fill_value=0)
    p = yy.rename("y").astype(float).reset_index()
    p["E"] = p.h3.map(E)
    p["logE"] = np.log(p.E)
    grp = _grupo(e, lista)
    tab = pd.crosstab(e.h3, grp)
    comp = tab.div(tab.sum(axis=1), axis=0).add_prefix("s_")
    p = p.join(comp, on="h3")
    p["p_grandes"] = p.h3.map(e.per_ocu.isin(GRANDES).groupby(e.h3).mean())
    ov, cen = _h3_geo(tuple(sorted(celdas)))
    pan = lee_panel()
    at = pan[pan.anio == anios[0]][["cvegeo", "pobtot"]]
    pob = ov.merge(at, on="cvegeo").assign(v=lambda x: x.pobtot * x.w_ageb).groupby("h3").v.sum()
    p["pobtot"] = p.h3.map(pob).fillna(0)
    p["log_pob"] = np.log1p(p.pobtot)
    # Sin `no_residencial`: en H3 solo 2 celdas tienen población 0 y el indicador vuelve singular el ajuste.
    dist = (ov.merge(pan[["cvegeo", "anio", "dist_estacion_km"]], on="cvegeo")
            .assign(v=lambda x: x.dist_estacion_km * x.w_hex, w=lambda x: x.w_hex)
            .groupby(["h3", "anio"])[["v", "w"]].sum())
    p["dist_estacion_km"] = (dist.v / dist.w).reindex(pd.MultiIndex.from_frame(p[["h3", "anio"]])).values
    p["log_dist"] = np.log(p.dist_estacion_km + 0.05)
    p = p.merge(cen, on="h3")
    # Celdas en el borde urbano sin intersección con AGEB (puntos en el límite): se descartan.
    p = p.dropna(subset=["dist_estacion_km"]).reset_index(drop=True)
    for c in [c for c in comp.columns if c not in p]:
        p[c] = 0.0
    return p.rename(columns={"h3": "cvegeo"})  # 'cvegeo' = id de la unidad, para reutilizar funciones
