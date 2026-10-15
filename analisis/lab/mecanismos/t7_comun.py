"""Utilidades comunes de T7 (protocolo: analisis/lab/mecanismos/README.md). Importa la maquinaria de analisis/pipeline/modelado.py."""
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1] / "pipeline"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import modelado as md  # noqa: E402
from common import RAW  # noqa: E402

RES = AQUI / "resultados"
RES.mkdir(exist_ok=True)
SEED = 42
B = 1000


class Log:
    """print a pantalla y a un archivo de log."""

    def __init__(self, nombre):
        self.f = open(RES / f"log_{nombre}.txt", "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def guarda_json(d, nombre=None):
    """Un archivo por parte (veredictos_<parte>.json) para que scripts en paralelo no se pisen; `une_json()` los
    junta en veredictos.json."""
    for k, v in d.items():
        (RES / f"veredictos_{k}.json").write_text(json.dumps(v, indent=2, ensure_ascii=False, default=float))


def une_json():
    out = {f.stem.replace("veredictos_", ""): json.loads(f.read_text()) for f in sorted(RES.glob("veredictos_*.json"))}
    (RES / "veredictos.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    return out


def unidad_h3(p):
    """Copia de unidad_h3() de 04_modelos.py: bloques k-means sobre centroides únicos de celda."""
    u = p.drop_duplicates("cvegeo")[["cvegeo", "cx", "cy"]].reset_index(drop=True)
    b = md.asigna_bloques(u[["cx", "cy"]].values)
    b.index = u.cvegeo
    return p.cvegeo.map(b.fold).values, p.cvegeo.map(b.bloque).values


def temporal_mu(p, cols, modelo, bloque):
    """Copia literal de la lógica de temporal() de 04_modelos.py (2020–2023 → 2024; el efecto fijo de 2024 hereda
    el de 2023), devolviendo también las predicciones de 2024."""
    q = p.copy()
    c = [x for x in cols if x != "fe_2024"]
    if "fe_2023" in q:
        q.loc[q.anio == 2024, "fe_2023"] = 1.0
    tr, te = (q.anio <= 2023).values, (q.anio == 2024).values
    X, y = q[c].values.astype(float), q.y.values
    if modelo == "nb":
        mu = md.predice(md.ajusta_nb(X[tr], y[tr]), X[te])
    else:
        mu = md.ajusta_m2(X[tr], y[tr], bloque[tr]).predict(X[te])
    return mu, {"dev": md.devianza(y[te], mu), "mae": float(np.mean(np.abs(y[te] - mu))),
                "capture": md.capture(y[te], mu)}


def boot_params(p, cols, bloque, B=B, fn=None):
    """Bootstrap de bloques espaciales (mismo esquema y semilla que modelado.bootstrap_beta), guardando el vector
    completo de parámetros (const, cols..., alpha). Devuelve (matriz de réplicas finitas, nº descartadas)."""
    rng = np.random.default_rng(SEED)
    grupos = {b_: g.index.values for b_, g in p.groupby(bloque)}
    ids = np.array(list(grupos))
    X, y = p[cols].values.astype(float), p.y.values
    out = []
    for k in range(B):
        i = np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)])
        out.append(np.asarray(md.ajusta_nb(X[i], y[i]).params))
        if fn and (k + 1) % 100 == 0:
            fn(f"   réplica {k + 1}/{B}")
    out = np.array(out)
    ok = np.isfinite(out).all(axis=1)
    return out[ok], int((~ok).sum())


def ic(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------------ índice socioeconómico (parte 3)
INDICADORES = ["escolaridad", "sin_salud", "internet", "automovil", "ocup_cuarto"]


def censo_ageb():
    """Cinco indicadores del Censo 2020 por AGEB urbana (filas de total de AGEB: MZA == 0 y AGEB != 0000)."""
    c = pd.read_csv(RAW / "censo2020/resageburb/RESAGEBURB_09CSV20.csv", dtype=str, encoding="utf-8-sig")
    c = c[(c.MZA.astype(int) == 0) & (c.AGEB != "0000")].copy()
    c["cvegeo"] = c.ENTIDAD.str.zfill(2) + c.MUN.str.zfill(3) + c.LOC.str.zfill(4) + c.AGEB.str.zfill(4)
    assert not c.cvegeo.duplicated().any()
    num = lambda s: pd.to_numeric(s.replace({"*": np.nan, "N/D": np.nan}), errors="coerce")  # noqa: E731
    v = {k: num(c[k]) for k in ["GRAPROES", "PSINDER", "POBTOT", "VPH_INTER", "VPH_AUTOM", "VIVPAR_HAB", "VIVPARH_CV", "PRO_OCUP_C"]}
    # DESVIACIÓN DEL PROTOCOLO (documentada): los VPH_* del Censo 2020 se refieren a las viviendas particulares
    # habitadas CON características (VIVPARH_CV), no a VIVPAR_HAB; con VIVPAR_HAB 233 AGEB daban proporciones > 1.
    den = v["VIVPARH_CV"]
    d = pd.DataFrame({"cvegeo": c.cvegeo.values, "pobtot": v["POBTOT"].values,
                      "escolaridad": v["GRAPROES"].values,
                      "sin_salud": (v["PSINDER"] / v["POBTOT"]).values,
                      "internet": (v["VPH_INTER"] / den).values,
                      "automovil": (v["VPH_AUTOM"] / den).values,
                      "ocup_cuarto": v["PRO_OCUP_C"].values})
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.set_index("cvegeo")


def indice_social():
    """ACP sobre los cinco indicadores estandarizados (AGEB con población y datos completos). Más alto = mejor
    (carga de escolaridad positiva). Devuelve (DataFrame con indicadores e índice, dict con cargas y varianza)."""
    d = censo_ageb()
    n_total = len(d)
    d = d[d.pobtot > 0]
    n_pob = len(d)
    comp = d[INDICADORES].dropna()
    z = (comp - comp.mean()) / comp.std(ddof=0)
    val, vec = np.linalg.eigh(np.cov(z.values.T, ddof=0))
    orden = np.argsort(val)[::-1]
    val, vec = val[orden], vec[:, orden]
    w = vec[:, 0]
    if w[INDICADORES.index("escolaridad")] < 0:
        w = -w
    d = d.copy()
    d["indice"] = np.nan
    d.loc[comp.index, "indice"] = z.values @ w
    info = {"n_ageb_censo": n_total, "n_con_poblacion": n_pob, "n_completas": len(comp),
            "cargas": dict(zip(INDICADORES, w.round(4).tolist())),
            "var_explicada_pc1": float(val[0] / val.sum()), "var_explicada_pc2": float(val[1] / val.sum())}
    return d, info
