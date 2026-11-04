"""S7 · Piezas comunes de s7_variable_cadenas.py y s7_backtest_2025.py (protocolo en analisis/lab/README.md §S7).

- panel(var): panel AGEB × año 2020–2024 de V2b, V1 o V1c (definición de S6) con el control de cadenas.
- cols(p, ctl): columnas de M0/M1/M2 con o sin cadenas. Decisión del 2026-10-14: las cadenas entran
  como **control** (corrección de la denuncia), en M0, M1 y M2.
- por_violencia(var, p): el mismo panel con solo robo con violencia o sin violencia (como 04b/04c).
- backtest(p, c2, t): Ê de M2 sin fuga temporal (entrenado solo con años ≤ t, validación espacial) para los años
  ≤ t y para t + 1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from modelado import lightgbm  # noqa: E402

lightgbm()  # en macOS puede relanzar el proceso: debe ir antes que todo lo demás

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from common import CORTE_FIJO, GRANDES, INTERIM  # noqa: E402
from modelado import (SEED, _fe, ajusta_m2, bloques, cols_m1, cols_m2, contraccion, datos,  # noqa: E402
                      ajusta_nb)

RES = Path(__file__).resolve().parent / "resultados"
RES.mkdir(exist_ok=True)
ANIOS = list(range(2020, 2025))
VARIANTES = [("V2b", 0), ("V2b", 1), ("V1", 1), ("V1c", 1)]  # (variable, control de cadenas)
JARDINES = "0901000011260"
CASOS = {JARDINES: "Jardines del Pedregal (origen de T8)", "0901500010748": "Centro, Cuauhtémoc (demo 1)",
         "0900200011042": "Pueblo San Miguel Amantla (demo 2)", "0901700010278": "20 de Noviembre (demo 2b)",
         "0900700012390": "Dr. Alfonso Ortiz Tirado (demo 3, aviso ⚠)"}


def etiqueta(var, ctl):
    return f"{var}({ctl})"


# ------------------------------------------------------------ V1c: robo a clientes dentro de negocios (S6)
_d = pd.read_parquet(INTERIM / "delitos_ageb.parquet")
_d = _d[(_d.tipo_asig == "URBANA") & _d.anio_hecho.isin(ANIOS)]
LUGARES = ("EN NEGOCIO", "EN RESTAURANT", "EN HOTEL", "EN CINE")
_cli = _d[(_d.grupo == "transeunte_en_lugar") & _d.delito_norm.str.contains("|".join(LUGARES))]
assert _cli.delito_norm.str.contains("CON VIOLENCIA").all()  # todo el robo a clientes es con violencia


def _cuenta(m):
    return _d[m].groupby(["cvegeo", "anio_hecho"]).size().rename_axis(["cvegeo", "anio"])


CLIENTES = _cuenta(_d.index.isin(_cli.index))

# ------------------------------------------------------------ control de cadenas: DENUE 2023-11 (S6)
_e = pd.read_parquet(INTERIM / "denue_ageb.parquet")
_e = _e[_e.corte == CORTE_FIJO]
_MASCARAS = {
    "c_autoserv": lambda e: e.codigo_act.str.startswith("4621"),
    "c_autoserv_grande": lambda e: e.codigo_act.str.startswith("4621") & e.per_ocu.isin(GRANDES),
    "c_departamental": lambda e: e.codigo_act.str.startswith("4622"),
    "c_conveniencia": lambda e: e.codigo_act == "462112",
}
CADENAS = list(_MASCARAS)


def cadenas(e, unidad):
    """log1p de los conteos de cadenas por unidad espacial."""
    return pd.DataFrame({k: np.log1p(e[f(e)].groupby(e[unidad]).size()) for k, f in _MASCARAS.items()})


_CAD_AGEB = cadenas(_e, "cvegeo")


def panel(var):
    p = datos(y="V2b" if var == "V2b" else "V1", anios=ANIOS)
    if var == "V1c":
        p["y"] += pd.MultiIndex.from_arrays([p.cvegeo, p.anio]).map(CLIENTES).fillna(0).values
    p = p.join(_CAD_AGEB, on="cvegeo")
    p[CADENAS] = p[CADENAS].fillna(0)
    p["autoserv_grande"] = p.n_autoserv_grande_fijo > 0
    return p


def cols(p, ctl):
    """(M0, M1, M2). logE va primero en M0 y M1: el bootstrap toma β del índice 1 (tras la constante)."""
    cad = CADENAS if ctl else []
    return ["logE"] + cad + _fe(p), cols_m1(p) + cad, cols_m2(p) + cad


# ------------------------------------------------------------ robo con y sin violencia (04b/04c)
GRUPOS_CV = {"V2b": ["negocio_cv", "negocio_cv_otros"], "V1": ["negocio_cv", "negocio_cv_otros", "negocio_cv_farderos"]}
GRUPOS_SV = {"V2b": ["negocio_sv"], "V1": ["negocio_sv", "negocio_sv_dentro", "negocio_sv_farderos"]}


def por_violencia(var, p):
    base = "V2b" if var == "V2b" else "V1"
    out = {}
    for k, grupos in [("cv", GRUPOS_CV[base]), ("sv", GRUPOS_SV[base])]:
        n = _cuenta(_d.grupo.isin(grupos))
        q = p.copy()
        q["y"] = pd.MultiIndex.from_arrays([q.cvegeo, q.anio]).map(n).fillna(0).values.astype(float)
        if var == "V1c" and k == "cv":
            q["y"] += pd.MultiIndex.from_arrays([q.cvegeo, q.anio]).map(CLIENTES).fillna(0).values
        out[k] = q
    assert (out["cv"].y + out["sv"].y == p.y).all(), f"cv + sv no reproduce {var}"
    return out


# ------------------------------------------------------------ bootstrap de bloques (paralelo)
def boot_params(p, c, bloque, idx, B=1000):
    """Coeficientes NB2 en las posiciones `idx` (contando la constante) con B réplicas de bootstrap de bloques
    espaciales. Mismo esquema y misma secuencia aleatoria que modelado.bootstrap_beta (semilla 42)."""
    rng = np.random.default_rng(SEED)
    grupos = {b_: g.index.values for b_, g in p.groupby(bloque)}
    ids = np.array(list(grupos))
    X, y = p[c].values.astype(float), p.y.values
    muestras = [np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)]) for _ in range(B)]

    def uno(i):
        import warnings
        warnings.filterwarnings("ignore")  # los procesos de joblib no heredan el filtro de modelado.py
        return ajusta_nb(X[i], y[i]).params[idx]

    out = np.array(Parallel(n_jobs=-1)(delayed(uno)(i) for i in muestras))
    assert np.isfinite(out).all(axis=1).mean() > 0.99, "réplicas bootstrap sin converger"
    return out[np.isfinite(out).all(axis=1)]


# ------------------------------------------------------------ backtest sin fuga temporal
def backtest(p, c2, t):
    """Ê de M2 para los años 2020…t+1. Cada fold espacial se predice con un modelo entrenado en los otros 4 folds y
    solo con años ≤ t. El año entra numérico: t + 1 cae del lado de t en todos los cortes (hereda su nivel, como
    S3 y 04c). Devuelve el panel de años ≤ t + 1 con la columna `mu`."""
    q = p[p.anio <= t + 1].reset_index(drop=True)
    fold, bloque = q.cvegeo.map(bloques().fold).values, q.cvegeo.map(bloques().bloque).values
    X, y = q[c2].values.astype(float), q.y.values
    pre = (q.anio <= t).values
    mu = np.full(len(q), np.nan)
    for f in range(5):
        tr, te = pre & (fold != f), fold == f
        mu[te] = ajusta_m2(X[tr], y[tr], bloque[tr]).predict(X[te])
    return q.assign(mu=mu)


def pesos(anios, t, opcion):
    """(a) igual peso; (b) solo t−1 y t; (c) vida media de 1.5 años, año más reciente con peso 1."""
    anios = np.asarray(anios)
    if opcion == "a":
        return np.ones(len(anios))
    if opcion == "b":
        return (anios >= t - 1).astype(float)
    return 0.5 ** ((t - anios) / 1.5)


def lista(q, t, opcion, O=None):
    """Riesgo anticipado: contracción sobre Σ w·O y Σ w·Ê de los años ≤ t."""
    m = (q.anio <= t).values
    w = pesos(q.anio.values[m], t, opcion)
    O = q.y.values if O is None else O
    g = pd.DataFrame({"cvegeo": q.cvegeo.values[m], "O": w * O[m], "E": w * q.mu.values[m]}).groupby("cvegeo").sum()
    s = contraccion(g.O, g.E)
    return pd.DataFrame({"rr": s["rr"], "p_exc": s["p_exc"], "score": s["p_exc"] + 1e-6 * s["rr"]}, index=g.index)
