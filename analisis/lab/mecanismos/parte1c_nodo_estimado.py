"""T7 · Parte 1c (análisis posterior, 2026-10-14): el nodo del quiebre se estima en vez de
fijarlo en el p75. Se hace después de ver la parte 1b, y así se declara.

(a) Perfil de verosimilitud: logE + (logE − k)+ con k en una rejilla de cuantiles de log E por unidad (p20–p90, paso
    2.5 pp); k̂ = el de máxima verosimilitud. M0 y M1.
(b) Solo en H3 res. 8: bootstrap de bloques espaciales (1000 réplicas, semilla 42) que **reestima k en cada réplica**,
    así el IC de las pendientes incluye la incertidumbre de dónde cae el nodo. Da también el IC del nodo.
(c) Solo en H3 res. 8: datos de la gráfica del colchón para la app (`app/src/data/colchon.json`): robos por km² contra
    negocios por km² en 12 grupos de celdas (cuantiles de log E) y la recta con quiebre de M0 en k̂.

Uso: .venv/bin/python analisis/lab/mecanismos/parte1c_nodo_estimado.py h3     (y `h3_7`, `h3_9` = solo perfil)
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from t7_comun import md, RES, B, SEED, Log, unidad_h3, ic  # noqa: I001

import h3  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GRID_Q = np.round(np.arange(0.20, 0.9001, 0.025), 3)
APP = Path(__file__).resolve().parents[3] / "app" / "src" / "data" / "colchon.json"


def matriz(logE, base, k):
    return np.column_stack([logE, np.maximum(logE - k, 0.0), base])


def perfil(logE, base, y, ks):
    """(llf, β bajo, β alto) para cada nodo de la rejilla."""
    out = []
    for k in ks:
        par = md.ajusta_nb(matriz(logE, base, k), y)
        b = np.asarray(par.params)
        out.append((par.llf, b[1], b[1] + b[2]))
    return np.array(out)


_G = {}


def _init(logE, base, y, ks):
    _G.update(logE=logE, base=base, y=y, ks=ks)


def _replica(i):
    r = perfil(_G["logE"][i], _G["base"][i], _G["y"][i], _G["ks"])
    j = int(np.nanargmax(r[:, 0]))
    return j, r[j, 1], r[j, 2]


if __name__ == "__main__":
    UNIDAD = sys.argv[1]
    log = Log(f"parte1c_{UNIDAD}")
    md.H3_RES = int(UNIDAD.split("_")[1]) if "_" in UNIDAD else 8  # antes del primer panel_h3 (caché por proceso)
    res = md.H3_RES
    area = h3.average_hexagon_area(res, unit="km^2")
    p = md.panel_h3(md.PRINCIPAL)
    _, bloque = unidad_h3(p)
    u = p.drop_duplicates("cvegeo").logE
    ks = np.quantile(u, GRID_Q)
    log(f"Unidad {UNIDAD}: {p.cvegeo.nunique()} celdas de {area:.3f} km²; rejilla de {len(ks)} nodos "
        f"(E de {np.exp(ks[0]):.0f} a {np.exp(ks[-1]):.0f})")

    filas, nodo = [], {}
    for mod in ["M0", "M1"]:
        base_cols = [c for c in (md.cols_m0(p) if mod == "M0" else md.cols_m1(p)) if c != "logE"]
        logE, base, y = p.logE.values, p[base_cols].values.astype(float), p.y.values
        r = perfil(logE, base, y, ks)
        j = int(np.argmax(r[:, 0]))
        en_ic = r[:, 0] >= r[j, 0] - 1.92  # IC 95% por razón de verosimilitud (χ²₁/2)
        for q, k, (llf, bb, ba), dentro in zip(GRID_Q, ks, r, en_ic):
            filas.append({"unidad": UNIDAD, "modelo": mod, "q": q, "E": np.exp(k), "negocios_km2": np.exp(k) / area,
                          "llf": llf, "beta_bajo": bb, "beta_alto": ba, "max": q == GRID_Q[j], "ic_perfil": dentro})
        lo_q, hi_q = GRID_Q[en_ic].min(), GRID_Q[en_ic].max()
        log(f"   {mod}: k̂ = p{GRID_Q[j] * 100:.1f} (E = {np.exp(ks[j]):.0f}; ~{np.exp(ks[j]) / area:.0f}/km²); "
            f"β {r[j, 1]:.3f} → {r[j, 2]:.3f}; IC perfil del nodo p{lo_q * 100:.1f}–p{hi_q * 100:.1f} "
            f"(~{np.exp(np.quantile(u, lo_q)) / area:.0f}–{np.exp(np.quantile(u, hi_q)) / area:.0f}/km²); "
            f"Δllf contra p75 = {r[j, 0] - r[GRID_Q == 0.75, 0][0]:.2f}")
        nodo[mod] = {"j": j, "beta_bajo": r[j, 1], "beta_alto": r[j, 2]}

        if UNIDAD != "h3":
            continue
        # (b) Bootstrap de bloques con reestimación del nodo (mismo esquema y semilla que t7_comun.boot_params).
        rng = np.random.default_rng(SEED)
        grupos = {b_: g.index.values for b_, g in p.groupby(bloque)}
        ids = np.array(list(grupos))
        idx = [np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)]) for _ in range(B)]
        with ProcessPoolExecutor(max_workers=12, initializer=_init, initargs=(logE, base, y, ks)) as ex:
            bt = np.array(list(ex.map(_replica, idx, chunksize=10)))
        bt = bt[np.isfinite(bt).all(axis=1)]
        jj = bt[:, 0].astype(int)
        for nombre, punto, rep in [("β bajo", r[j, 1], bt[:, 1]), ("β alto", r[j, 2], bt[:, 2]),
                                   ("Δ = β alto − β bajo", r[j, 2] - r[j, 1], bt[:, 2] - bt[:, 1])]:
            lo, hi = ic(rep)
            filas.append({"unidad": UNIDAD, "modelo": mod, "q": np.nan, "coeficiente": nombre, "valor": punto,
                          "ic_lo": lo, "ic_hi": hi, "replicas": len(bt),
                          "p_boot": float(np.mean(rep >= 0)) if nombre.startswith("Δ") else np.nan})
            log(f"   {mod} bootstrap (nodo reestimado) {nombre}: {punto:.3f} [{lo:.3f}, {hi:.3f}]")
        qlo, qhi = np.percentile(GRID_Q[jj], [2.5, 97.5])
        log(f"   {mod} nodo en las réplicas: p{qlo * 100:.1f}–p{qhi * 100:.1f} "
            f"(~{np.exp(np.quantile(u, qlo)) / area:.0f}–{np.exp(np.quantile(u, qhi)) / area:.0f}/km²); "
            f"réplicas válidas {len(bt)}")
        nodo[mod].update(ic_delta=ic(bt[:, 2] - bt[:, 1]), ic_bajo=ic(bt[:, 1]), ic_alto=ic(bt[:, 2]),
                         ic_nodo=(float(np.exp(np.quantile(u, qlo)) / area), float(np.exp(np.quantile(u, qhi)) / area)))

    pd.DataFrame(filas).to_csv(RES / f"parte1c_nodo_{UNIDAD}.csv", index=False)
    if UNIDAD != "h3":
        log("listo")
        sys.exit()

    # (c) Gráfica: datos crudos por grupos de celdas + recta con quiebre de M0 en su k̂ (M0 no tiene controles, así
    # que la recta y los puntos están en la misma escala).
    c = p.groupby("cvegeo").agg(E=("E", "first"), y=("y", "sum"), anios=("anio", "nunique")).reset_index()
    c["y_anual"] = c.y / c.anios
    c["grupo"] = pd.qcut(np.log(c.E), 12, labels=False)
    rng = np.random.default_rng(SEED)
    puntos = []
    for _, g in c.groupby("grupo"):
        bs = [g.y_anual.values[rng.integers(0, len(g), len(g))].mean() / area for _ in range(1000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        puntos.append({"negocios_km2": g.E.mean() / area, "robos_km2": g.y_anual.mean() / area, "ic": [lo, hi],
                       "celdas": len(g)})

    base_cols = [x for x in md.cols_m0(p) if x != "logE"]
    k = ks[nodo["M0"]["j"]]
    m0 = md.ajusta_nb(matriz(p.logE.values, p[base_cols].values.astype(float), k), p.y.values)
    b = np.asarray(m0.params)
    fe = np.r_[0.0, b[3:3 + len(base_cols)]]  # 2020 = referencia
    nivel = np.log(np.mean(np.exp(fe)))  # promedio anual de 2020–2024
    xs = np.linspace(np.log(c.E.min()), np.log(c.E.max()), 60)

    def recta(x, quiebre=True):
        return float(np.exp(b[0] + nivel + b[1] * x + (b[2] * max(x - k, 0.0) if quiebre else 0.0)) / area)

    ultimo = np.log(puntos[-1]["negocios_km2"] * area)  # el grupo más denso, no la celda extrema
    salida = {
        "_fuente": "analisis/lab/mecanismos/parte1c_nodo_estimado.py (T7 parte 1c). No editar a mano.",
        "unidad": f"Hexágonos H3 res. {res} ({area:.2f} km²)",
        "celdas": int(len(c)),
        "periodo": f"{p.anio.min()}–{p.anio.max()}",
        "nodo_km2": float(np.exp(k) / area),
        "nodo_ic_km2": nodo["M0"].get("ic_nodo"),
        "percentil_nodo": float(GRID_Q[nodo["M0"]["j"]]),
        "percentiles_km2": {f"p{q}": float(np.quantile(c.E, q / 100) / area) for q in (25, 50, 75)},
        "m0": {k_: nodo["M0"][k_] for k_ in ("beta_bajo", "beta_alto", "ic_delta")},
        "m1": {k_: nodo["M1"][k_] for k_ in ("beta_bajo", "beta_alto", "ic_bajo", "ic_alto", "ic_delta", "ic_nodo")},
        "brecha_max": 1 - recta(ultimo) / recta(ultimo, quiebre=False),
        "puntos": puntos,
        "recta": [{"negocios_km2": float(np.exp(x) / area), "robos_km2": recta(x)} for x in xs],
        "sin_quiebre": [{"negocios_km2": float(np.exp(x) / area), "robos_km2": recta(x, quiebre=False)}
                        for x in xs if x >= k],
    }
    APP.write_text(json.dumps(salida, indent=1, ensure_ascii=False, default=float))
    log(f"Gráfica → {APP}; brecha en el extremo denso = {salida['brecha_max']:.1%}")
    log("listo")
