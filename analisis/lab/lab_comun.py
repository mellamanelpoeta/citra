"""Maquinaria común del lab (reutilizable en T2b): datos, composición por lista de giros, NB2,
validación espacial por bloques, métricas y contracción Poisson-gamma. Protocolo en analisis/lab/README.md."""
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, special, stats
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common import (CANDIDATOS_EFECTIVO, CORTE_FIJO, EFECTIVO, INTERIM, LISTA_TENTATIVA, SECTORES,  # noqa: E402
                    clasifica, lee_panel)

warnings.filterwarnings("ignore")
RES = Path(__file__).resolve().parent / "resultados"
RES.mkdir(exist_ok=True)
SEED = 42

# Variables objetivo y valores por defecto del protocolo
Y = {"V1": "y_v1_negocio", "V2": "y_v2_sin_farderos", "V2b": "y_v2b", "V3": "y_v3_via_publica"}
DEFAULT = {"y": "V2b", "e": "fijo", "lista": "b", "anios": list(range(2020, 2025))}

# Listas de giros "intensivos en efectivo" (uniones de candidatos de analisis/pipeline/common.py)
LISTAS = {
    "a": {"efectivo": LISTA_TENTATIVA},
    "b": EFECTIVO,  # la lista final del pipeline (era la variante (b) del protocolo)
    "c": {},
}
REF_SECTOR = "serv_prof_fin"  # categoría de referencia de la partición (se omite en el diseño)


# ------------------------------------------------------------------ datos
@lru_cache(maxsize=None)
def _denue():
    e = pd.read_parquet(INTERIM / "denue_ageb.parquet")
    e["cand"] = clasifica(e.codigo_act, CANDIDATOS_EFECTIVO, resto="")
    e["sector"] = clasifica(e.codigo_act, SECTORES)
    return e


@lru_cache(maxsize=None)
def composicion(lista):
    """Partición sectorial por (corte, cvegeo) con los grupos de efectivo de la lista. Proporciones que suman 1."""
    e = _denue()
    grupo = e.sector.copy()
    for g, cands in LISTAS[lista].items():
        grupo[e.cand.isin(cands)] = g
    tab = pd.crosstab([e.corte, e.cvegeo], grupo)
    return tab.div(tab.sum(axis=1), axis=0).add_prefix("s_")


def datos(y="V2b", e="fijo", lista="b", anios=None):
    """Panel listo para modelar: AGEB con E_fijo > 0, columnas y, logE, composición y controles."""
    anios = anios or DEFAULT["anios"]
    p = lee_panel()
    p = p[p.anio.isin(anios) & (p.E_fijo > 0)].copy()
    if e == "nov":
        p = p[p.E_nov > 0]
    p["y"] = p[Y[y]].astype(float)
    p["logE"] = np.log(p[f"E_{e}"])
    p["p_grandes"] = p[f"p_grandes_{e}"]
    comp = composicion(lista)
    corte = p.anio if e == "nov" else pd.Series(CORTE_FIJO, index=p.index)
    c = comp.reindex(pd.MultiIndex.from_arrays([corte.values, p.cvegeo.values])).fillna(0)
    c.index = p.index
    p = pd.concat([p, c], axis=1)
    p["log_dist"] = np.log(p.dist_estacion_km + 0.05)
    return p.reset_index(drop=True)


def cols_m0(p):
    return ["logE"] + _fe(p)


def cols_m1(p):
    comp = [c for c in p.columns if c.startswith("s_") and c != f"s_{REF_SECTOR}"]
    return ["logE"] + comp + ["p_grandes", "log_pob", "no_residencial", "log_dist"] + _fe(p)


def _fe(p):
    for a in sorted(p.anio.unique())[1:]:
        p[f"fe_{a}"] = (p.anio == a).astype(float)
    return [f"fe_{a}" for a in sorted(p.anio.unique())[1:]]


# ------------------------------------------------------------------ bloques espaciales
@lru_cache(maxsize=None)
def bloques(k=30, n_folds=5):
    """cvegeo → (bloque, fold). k-means sobre centroides; bloques repartidos en folds equilibrados por # de AGEB."""
    p = lee_panel()
    a = p[(p.anio == CORTE_FIJO) & (p.E_fijo > 0)][["cvegeo", "x", "y"]].reset_index(drop=True)
    a["bloque"] = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(a[["x", "y"]])
    tam = a.bloque.value_counts().sort_values(ascending=False)
    carga, asign = np.zeros(n_folds), {}
    for b_, n in tam.items():
        f = int(np.argmin(carga))
        asign[b_] = f
        carga[f] += n
    a["fold"] = a.bloque.map(asign)
    return a.set_index("cvegeo")[["bloque", "fold"]]


# ------------------------------------------------------------------ modelos
def ajusta_nb(X, y):
    """NB2 por máxima verosimilitud (alpha estimado). Arranca desde el Poisson GLM."""
    Xc = sm.add_constant(X, has_constant="add")
    start = sm.GLM(y, Xc, family=sm.families.Poisson()).fit().params
    m = sm.NegativeBinomial(y, Xc, loglike_method="nb2").fit(
        start_params=np.append(start, 0.5), method="newton", maxiter=200, disp=0)
    if not m.mle_retvals.get("converged", True):
        m = sm.NegativeBinomial(y, Xc, loglike_method="nb2").fit(
            start_params=np.append(start, 0.5), method="bfgs", maxiter=2000, disp=0)
    return m


def predice(m, X):
    return np.asarray(m.predict(sm.add_constant(X, has_constant="add")))


def cv_espacial(p, cols):
    """Predicciones fuera de muestra por fold espacial. Devuelve (mu_oof, devianza media por fold)."""
    fold = p.cvegeo.map(bloques().fold).values
    mu = np.full(len(p), np.nan)
    dev = {}
    for f in range(5):
        tr, te = fold != f, fold == f
        m = ajusta_nb(p.loc[tr, cols].values, p.loc[tr, "y"].values)
        mu[te] = predice(m, p.loc[te, cols].values)
        dev[f] = devianza(p.loc[te, "y"].values, mu[te])
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


def ranking_residual(p, mu):
    """Suma del periodo por AGEB de O y Ê (fuera de muestra) → contracción. DataFrame indexado por cvegeo."""
    g = p.assign(mu=mu).groupby("cvegeo").agg(O=("y", "sum"), Ehat=("mu", "sum"), ag=("n_autoserv_grande_fijo", "first"))
    s = contraccion(g.O, g.Ehat)
    g["rr"], g["p_exc"] = s["rr"], s["p_exc"]
    g["score"] = g.p_exc + 1e-6 * g.rr  # desempate por riesgo contraído
    return g


def top(s, q=0.10):
    return set(s.sort_values(ascending=False).index[: int(round(len(s) * q))])


def jaccard(a, b):
    return len(a & b) / len(a | b)


def persistencia(p, mu):
    """ρ de Spearman medio del riesgo contraído entre años consecutivos (contracción por año)."""
    d = p.assign(mu=mu)
    rr = {}
    for a, g in d.groupby("anio"):
        rr[a] = pd.Series(contraccion(g.y, g.mu)["rr"], index=g.cvegeo.values)
    anios = sorted(rr)
    return float(np.mean([stats.spearmanr(rr[a], rr[b].reindex(rr[a].index))[0] for a, b in zip(anios, anios[1:])]))
