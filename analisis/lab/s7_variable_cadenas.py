"""S7 parte 1 · V2b(0), V2b(1), V1(1) y V1c(1) con la maquinaria completa de 04_modelos (analisis/lab/README.md §S7).

(1) = control de cadenas (autoservicios, autoservicios grandes, departamentales, minisúpers del DENUE 2023-11),
como control en M0, M1 y M2 (decisión del 2026-10-14). Periodo 2020–2024 para las cuatro variantes.

Por variante: A (β de M0 en AGEB y en H3), B (ΔD M0 → M1, IRR por grupo de giros), C (Jaccard conteo vs riesgo,
ρ interanual), robustez (solo robo con violencia, denuncia por tamaño, efectos de alcaldía), validez (% de
autoservicio grande en el top 10%, calibración O/Ê por decil de Ê) y casos (Jardines del Pedregal y zonas de demo).
El backtest (parte 2 y criterio 2 de la regla) está en s7_backtest_2025.py; la regla, en s7_decidir.py.

Uso: .venv/bin/python analisis/lab/s7_variable_cadenas.py
Salidas: analisis/lab/resultados/s7_variable.csv, s7_irr.csv, s7_calibracion.csv, s7_casos.csv, s7_rankings.parquet
"""
from s7_comun import (CADENAS, CASOS, JARDINES, RES, VARIANTES, boot_params, cadenas, cols, etiqueta,  # noqa: I001
                      panel, por_violencia)

import importlib

import numpy as np
import pandas as pd
from scipy import stats

from common import lee_panel
from modelado import (_h3_puntos, ajusta_nb, asigna_bloques, bloques, cols_comp, cv_espacial, jaccard, panel_h3,
                      persistencia, ranking_residual, top)

s04b = importlib.import_module("04b_sesgo_denuncia")
B = 1000
UMBRAL = {"A_ic_sup": 1.0, "B_dD": 0.05, "B_folds": 4, "C_jaccard": 0.5, "C_rho": 0.3, "M2_sel": 0.05}
CORTE_AVISO = 0.25  # aviso ⚠ de la app: puesto con solo violencia fuera del top 25%


def h3(var):
    """Panel H3 res. 8, 2020–2024, con cadenas por celda y, para V1c, el robo a clientes reasignado por coordenada."""
    h = panel_h3("V2b" if var == "V2b" else "V1")
    h = h[h.anio.isin(range(2020, 2025))].reset_index(drop=True)
    d, e = _h3_puntos()
    if var == "V1c":
        cli = d[d.index.isin(CLIENTES_IDX)]
        n = cli.groupby(["h3", "anio_hecho"]).size()
        h["y"] += pd.MultiIndex.from_arrays([h.cvegeo, h.anio]).map(n).fillna(0).values
    h = h.join(cadenas(e, "h3"), on="cvegeo")
    h[CADENAS] = h[CADENAS].fillna(0)
    return h


def bloques_h3(h):
    u = h.drop_duplicates("cvegeo")[["cvegeo", "cx", "cy"]].reset_index(drop=True)
    b = asigna_bloques(u[["cx", "cy"]].values)
    b.index = u.cvegeo
    return h.cvegeo.map(b.fold).values, h.cvegeo.map(b.bloque).values


def ic(x):
    return np.percentile(x, [2.5, 97.5])


# Índices (en delitos_ageb) del robo a clientes: para reasignarlo a H3 por coordenada
import s7_comun  # noqa: E402

CLIENTES_IDX = s7_comun._cli.index
pan = lee_panel()
nombres = pan[pan.anio == 2023].set_index("cvegeo")[["alcaldia", "colonia"]]
r_tam = s04b.tasa_denuncia_tamano()

filas, irr, calib, casos, rankings = [], [], [], [], {}
for var, ctl in VARIANTES:
    et = etiqueta(var, ctl)
    print(f"== {et}", flush=True)
    p = panel(var)
    fold, bloque = p.cvegeo.map(bloques().fold).values, p.cvegeo.map(bloques().bloque).values
    c0, c1, c2 = cols(p, ctl)
    X0, X1, y = p[c0].values.astype(float), p[c1].values.astype(float), p.y.values
    f = {"variante": et, "variable": var, "control": ctl, "carpetas": int(y.sum())}

    # ---------------- A: β de M0 (AGEB y H3)
    m0 = ajusta_nb(X0, y)
    lo, hi = ic(boot_params(p, c0, bloque, [1], B)[:, 0])
    f.update(beta=m0.params[1], beta_lo=lo, beta_hi=hi, A="confirmada" if hi < UMBRAL["A_ic_sup"] else "rechazada")
    h = h3(var)
    _, bh = bloques_h3(h)
    c0h = cols(h, ctl)[0]
    mh = ajusta_nb(h[c0h].values.astype(float), h.y.values)
    lo, hi = ic(boot_params(h, c0h, bh, [1], B)[:, 0])
    f.update(beta_h3=mh.params[1], beta_h3_lo=lo, beta_h3_hi=hi, n_h3=h.cvegeo.nunique(),
             A_h3="confirmada" if hi < UMBRAL["A_ic_sup"] else "rechazada")

    # ---------------- B: ΔD M0 → M1 por fold e IRR de la composición
    mu0, dev0 = cv_espacial(p, c0)
    mu1, dev1 = cv_espacial(p, c1)
    mejora = {k: (dev0[k] - dev1[k]) / dev0[k] for k in dev0}
    dD, n_mej = float(np.mean(list(mejora.values()))), int(sum(m > 0 for m in mejora.values()))
    f.update(dD=dD, folds_mejora=n_mej,
             B="confirmada" if dD >= UMBRAL["B_dD"] and n_mej >= UMBRAL["B_folds"] else "rechazada")
    m1 = ajusta_nb(X1, y)
    comp = cols_comp(p)
    k = [c1.index(c) + 1 for c in comp]
    bt = boot_params(p, c1, bloque, k, B)
    for j, c in enumerate(comp):
        a, b_ = ic(bt[:, j])
        irr.append({"variante": et, "grupo": c[2:], "irr10": np.exp(0.1 * m1.params[k[j]]),
                    "irr10_lo": np.exp(0.1 * a), "irr10_hi": np.exp(0.1 * b_)})

    # ---------------- C: Ê con M2, contracción del periodo, persistencia
    mu2, dev2 = cv_espacial(p, c2, modelo="m2")
    mejora_m2 = (np.mean(list(dev1.values())) - np.mean(list(dev2.values()))) / np.mean(list(dev1.values()))
    g = ranking_residual(p, mu2)
    T, C = top(g.score), top(g.O)
    rho, _ = persistencia(p, mu2)
    f.update(dev_M1=np.mean(list(dev1.values())), dev_M2=np.mean(list(dev2.values())), mejora_m2_sobre_m1=mejora_m2,
             regla4_elige="M2" if mejora_m2 >= UMBRAL["M2_sel"] else "M1",
             jaccard_conteo_riesgo=jaccard(C, T), rho_interanual=rho, invisibles=len(T - C),
             C="confirmada" if jaccard(C, T) <= UMBRAL["C_jaccard"] and rho >= UMBRAL["C_rho"] else "rechazada")

    # ---------------- Validez: % de autoservicio grande en el top y calibración por decil de Ê
    ag = p.groupby("cvegeo").autoserv_grande.first()
    f.update(pct_autoserv_top10=100 * float(ag.reindex(list(T)).mean()), pct_autoserv_base=100 * float(ag.mean()))
    gd = g.assign(decil=pd.qcut(g.Ehat, 10, labels=False) + 1).groupby("decil")[["O", "Ehat"]].sum()
    gd["O_sobre_E"] = gd.O / gd.Ehat
    calib += [{"variante": et, "decil": d_, **r} for d_, r in gd.iterrows()]
    f.update(calib_min=gd.O_sobre_E.min(), calib_max=gd.O_sobre_E.max())

    # ---------------- Robustez
    # (i) C con solo robo con violencia (04c): lista propia de violencia frente a la de la variante
    tipos = por_violencia(var, p)
    g_cv = ranking_residual(tipos["cv"], cv_espacial(tipos["cv"], c2, modelo="m2")[0])
    puesto_cv = g_cv.score.rank(ascending=False, method="first")
    comun = g.index.intersection(g_cv.index)
    f.update(cv_carpetas=int(tipos["cv"].y.sum()), cv_jaccard_vs_lista=jaccard(T, top(g_cv.score)),
             cv_rho_rr=stats.spearmanr(g.rr[comun], g_cv.rr[comun])[0],
             cv_invisibles_retenidas=len((T - C) & top(g_cv.score)) / len(T - C),
             cv_jaccard_conteo_riesgo=jaccard(top(g_cv.O), top(g_cv.score)),
             aviso_sin_violencia=int((puesto_cv.reindex(list(T)) > CORTE_AVISO * len(g)).sum()))
    # (ii) denuncia por tamaño de empresa (04b): β de M0 con offset log r
    off = np.log(p.cvegeo.map(r_tam).values)
    b_off, boot_off = s04b.beta_boot(p, c0, off=off)
    lo, hi = ic(boot_off)
    f.update(beta_denuncia_tamano=b_off, beta_denuncia_tamano_lo=lo, beta_denuncia_tamano_hi=hi)
    # (iii) Ê con efectos de alcaldía (04d): 15 dummies en M2
    alc = pd.get_dummies(p.cvegeo.str[2:5], prefix="alc", drop_first=True).astype(float)
    q = pd.concat([p, alc], axis=1)
    g_alc = ranking_residual(q, cv_espacial(q, c2 + list(alc.columns), modelo="m2")[0])
    f.update(alc_jaccard_vs_lista=jaccard(T, top(g_alc.score)), alc_rho_rr=stats.spearmanr(g.rr, g_alc.rr[g.index])[0],
             alc_invisibles_retenidas=len((T - C) & top(g_alc.score)) / len(T - C))

    # ---------------- Casos y rankings
    puesto = g.score.rank(ascending=False, method="first")
    puesto_O = g.O.rank(ascending=False, method="first")
    for cve, nom in CASOS.items():
        casos.append({"variante": et, "cvegeo": cve, "zona": nom, "O": g.O[cve], "Ehat": g.Ehat[cve], "rr": g.rr[cve],
                      "p_exc": g.p_exc[cve], "puesto": int(puesto[cve]), "puesto_conteo": int(puesto_O[cve]),
                      "puesto_solo_violencia": int(puesto_cv[cve]), "en_top10": cve in T})
    rankings[et] = g.assign(puesto=puesto, puesto_conteo=puesto_O, puesto_cv=puesto_cv)
    f["rank_jardines"] = int(puesto[JARDINES])
    filas.append(f)
    print({k_: round(v, 3) if isinstance(v, float) else v for k_, v in f.items()}, flush=True)

# Jaccard de cada lista con la de producción
T_ref = top(rankings["V2b(0)"].score)
for f in filas:
    f["jaccard_vs_V2b0"] = jaccard(top(rankings[f["variante"]].score), T_ref)
    f["rho_rr_vs_V2b0"] = stats.spearmanr(rankings[f["variante"]].rr, rankings["V2b(0)"].rr)[0]

pd.DataFrame(filas).to_csv(RES / "s7_variable.csv", index=False, float_format="%.4f")
pd.DataFrame(irr).to_csv(RES / "s7_irr.csv", index=False, float_format="%.4f")
pd.DataFrame(calib).to_csv(RES / "s7_calibracion.csv", index=False, float_format="%.4f")
pd.DataFrame(casos).to_csv(RES / "s7_casos.csv", index=False, float_format="%.4f")
pd.concat({k: v for k, v in rankings.items()}, names=["variante", "cvegeo"]).reset_index().join(
    nombres, on="cvegeo").to_parquet(RES / "s7_rankings.parquet", index=False)
print(pd.DataFrame(filas).set_index("variante").T.to_string())
