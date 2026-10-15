"""T7 · Parte 1b (análisis posterior, 2026-10-14): ¿la relación es casi proporcional en T1–T3
y se aplana a partir de T4? Protocolo de la parte 1 (analisis/lab/mecanismos/README.md), con un solo nodo en el p75 de log E.

(a) Quiebre: logE + (logE − k75)+ en toda la muestra → β T1–T3, β T4 y Δ = β T4 − β T1–T3, con el bootstrap de bloques.
(b) Verificación: el mismo modelo sin quiebre, ajustado por separado en T1–T3 y en T4.

Uso: .venv/bin/python analisis/lab/mecanismos/parte1b_quiebre_t4.py ageb   (y `h3` = res. 8; `h3_7`, `h3_9` = sensibilidad de
     resolución, agregada el 2026-10-14 para no elegir la escala después de ver el resultado)
"""
import sys

from t7_comun import md, RES, B, Log, unidad_h3, boot_params, ic  # noqa: I001

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

UNIDAD = sys.argv[1]
log = Log(f"parte1b_{UNIDAD}")
if UNIDAD == "ageb":
    p = md.datos(md.PRINCIPAL)
    bloque = p.cvegeo.map(md.bloques().bloque).values
else:
    md.H3_RES = int(UNIDAD.split("_")[1]) if "_" in UNIDAD else 8  # antes del primer panel_h3 (caché por proceso)
    p = md.panel_h3(md.PRINCIPAL)
    _, bloque = unidad_h3(p)
k = np.quantile(p.drop_duplicates("cvegeo").logE, 0.75)
p["sp_t4"] = np.maximum(p.logE - k, 0.0)
alto = (p.logE > k).values
log(f"Unidad {UNIDAD}: {p.cvegeo.nunique()} unidades; nodo p75 log E = {k:.3f} → E = {np.exp(k):.1f}; "
    f"T4 = {p[alto].cvegeo.nunique()} unidades")

filas = []
for mod in ["M0", "M1"]:
    base = [c for c in (md.cols_m0(p) if mod == "M0" else md.cols_m1(p)) if c != "logE"]

    cols = ["logE", "sp_t4"] + base
    par = np.asarray(md.ajusta_nb(p[cols].values.astype(float), p.y.values).params)
    bt, malas = boot_params(p, cols, bloque, B=B, fn=log)
    for nombre, punto, rep in [("β T1–T3", par[1], bt[:, 1]), ("β T4", par[1] + par[2], bt[:, 1] + bt[:, 2]),
                               ("Δ = β T4 − β T1–T3", par[2], bt[:, 2])]:
        lo, hi = ic(rep)
        filas.append({"unidad": UNIDAD, "modelo": mod, "metodo": "quiebre", "coeficiente": nombre, "valor": punto,
                      "ic_lo": lo, "ic_hi": hi, "p_boot": float(np.mean(rep >= 0)) if nombre.startswith("Δ") else
                      np.nan, "replicas": len(bt), "descartadas": malas})
        log(f"   {mod} quiebre {nombre}: {punto:.3f} [{lo:.3f}, {hi:.3f}]")

    cols = ["logE"] + base
    for nombre, m in [("β T1–T3", ~alto), ("β T4", alto)]:
        q = p[m].reset_index(drop=True)
        q = q.loc[:, q.nunique() > 1]  # efectos fijos o dummies constantes en la submuestra
        c = [x for x in cols if x in q]
        par = np.asarray(md.ajusta_nb(q[c].values.astype(float), q.y.values).params)
        bt, malas = boot_params(q, c, bloque[m], B=B, fn=log)
        lo, hi = ic(bt[:, 1])
        filas.append({"unidad": UNIDAD, "modelo": mod, "metodo": "submuestra", "coeficiente": nombre,
                      "valor": par[1], "ic_lo": lo, "ic_hi": hi, "p_boot": np.nan, "replicas": len(bt),
                      "descartadas": malas})
        log(f"   {mod} submuestra {nombre}: {par[1]:.3f} [{lo:.3f}, {hi:.3f}] (descartadas {malas})")

pd.DataFrame(filas).to_csv(RES / f"parte1b_quiebre_{UNIDAD}.csv", index=False)
log("listo")
