"""T7 · Partes 1 y 2: β por tramos de E (spline lineal) y densidad (+ log área). Protocolo: analisis/lab/mecanismos/README.md.

Uso: .venv/bin/python analisis/lab/mecanismos/parte1_2_beta.py ageb   (y luego `h3`; pueden correr en paralelo)
     .venv/bin/python analisis/lab/mecanismos/parte1_2_beta.py figura (une las salidas y dibuja; aplica las reglas)
"""
import sys

from t7_comun import md, RES, B, Log, guarda_json, unidad_h3, boot_params, ic  # noqa: I001

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

UNIDAD = sys.argv[1]
TRAMOS = ["T1 (inferior)", "T2", "T3", "T4 (superior)"]


def base_spline(p):
    """logE + (logE − k)+ con k = cuartiles de log E sobre unidades únicas (fijos en el bootstrap)."""
    u = p.drop_duplicates("cvegeo").logE
    nodos = np.quantile(u, [0.25, 0.5, 0.75])
    for j, k in enumerate(nodos, 1):
        p[f"sp{j}"] = np.maximum(p.logE - k, 0.0)
    return ["logE", "sp1", "sp2", "sp3"], nodos


def corre(p, cols, bloque, etiqueta, log, filas, extra=None):
    """Ajusta NB en toda la muestra + bootstrap; devuelve punto y réplicas de los coeficientes de interés."""
    m = md.ajusta_nb(p[cols].values.astype(float), p.y.values)
    log(f"== {etiqueta}: ajuste completo, bootstrap {B} réplicas")
    bt, malas = boot_params(p, cols, bloque, B=B, fn=log)
    log(f"   réplicas descartadas: {malas}")
    return np.asarray(m.params), bt, malas


if UNIDAD in ("ageb", "h3"):
    log = Log(f"parte1_2_{UNIDAD}")
    if UNIDAD == "ageb":
        p = md.datos(md.PRINCIPAL)
        bloque = p.cvegeo.map(md.bloques().bloque).values
    else:
        p = md.panel_h3(md.PRINCIPAL)
        _, bloque = unidad_h3(p)
    log(f"Unidad {UNIDAD}: {p.cvegeo.nunique()} unidades, {len(p)} unidad-año")
    sp, nodos = base_spline(p)
    log("Nodos (log E):", np.round(nodos, 3).tolist(), "→ E =", np.round(np.exp(nodos), 1).tolist())
    filas = []
    for mod in ["M0", "M1"]:
        base = md.cols_m0(p) if mod == "M0" else md.cols_m1(p)
        cols = sp + [c for c in base if c != "logE"]
        par, bt, malas = corre(p, cols, bloque, f"{mod} spline {UNIDAD}", log, filas)
        for j, t in enumerate(TRAMOS):
            punto = par[1:2 + j].sum()
            rep = bt[:, 1:2 + j].sum(axis=1)
            lo, hi = ic(rep)
            filas.append({"unidad": UNIDAD, "modelo": mod, "tramo": t, "beta": punto, "ic_lo": lo, "ic_hi": hi,
                          "E_desde": float(np.exp([-np.inf, *nodos][j])), "E_hasta": float(np.exp([*nodos, np.inf][j])),
                          "replicas": len(bt), "descartadas": malas})
            log(f"   {mod} {t}: β = {punto:.3f} [{lo:.3f}, {hi:.3f}]")
    pd.DataFrame(filas).to_csv(RES / f"parte1_tramos_{UNIDAD}.csv", index=False)

    f2 = []
    if UNIDAD == "ageb":
        p["log_area"] = np.log(p.area_km2)
        for mod in ["M0", "M1"]:
            base = md.cols_m0(p) if mod == "M0" else md.cols_m1(p)
            cols = ["logE", "log_area"] + [c for c in base if c != "logE"]
            par, bt, malas = corre(p, cols, bloque, f"{mod} + log_area", log, f2)
            for nombre, punto, rep in [("beta (densidad, área fija)", par[1], bt[:, 1]),
                                       ("gamma (log área)", par[2], bt[:, 2]),
                                       ("beta + gamma (área, densidad fija)", par[1] + par[2], bt[:, 1] + bt[:, 2])]:
                lo, hi = ic(rep)
                f2.append({"unidad": "ageb", "modelo": f"{mod} + log_area", "coeficiente": nombre, "valor": punto,
                           "ic_lo": lo, "ic_hi": hi, "replicas": len(bt), "descartadas": malas})
                log(f"   {mod}+área {nombre}: {punto:.3f} [{lo:.3f}, {hi:.3f}]")
    else:
        cols = md.cols_m1(p)
        par, bt, malas = corre(p, cols, bloque, "M1 global H3", log, f2)
        lo, hi = ic(bt[:, 1])
        f2.append({"unidad": "h3", "modelo": "M1", "coeficiente": "beta global (área constante = densidad)",
                   "valor": par[1], "ic_lo": lo, "ic_hi": hi, "replicas": len(bt), "descartadas": malas})
        log(f"   M1 H3 β global: {par[1]:.3f} [{lo:.3f}, {hi:.3f}]")
    pd.DataFrame(f2).to_csv(RES / f"parte2_densidad_{UNIDAD}.csv", index=False)
    log("listo")

elif UNIDAD == "figura":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    log = Log("parte1_2_reglas")
    t = pd.concat([pd.read_csv(RES / f"parte1_tramos_{u}.csv") for u in ("ageb", "h3")], ignore_index=True)
    d = pd.concat([pd.read_csv(RES / f"parte2_densidad_{u}.csv") for u in ("ageb", "h3")], ignore_index=True)
    t.to_csv(RES / "parte1_tramos.csv", index=False)
    d.to_csv(RES / "parte2_densidad.csv", index=False)
    s = t.set_index(["unidad", "modelo", "tramo"])
    sup = {u: s.loc[(u, "M1", "T4 (superior)")] for u in ("ageb", "h3")}
    inf = {u: s.loc[(u, "M0", "T1 (inferior)")] for u in ("ageb", "h3")}
    colchon = bool(sup["ageb"].ic_hi < 1 and sup["h3"].ic_hi < 1)
    piso = bool(inf["ageb"].ic_hi < 1 and inf["h3"].ic_lo <= 1 <= inf["h3"].ic_hi)
    a = d.set_index(["modelo", "coeficiente"])
    bm1 = a.loc[("M1 + log_area", "beta (densidad, área fija)")]
    gm1 = a.loc[("M1 + log_area", "gamma (log área)")]
    conc = bool(bm1.ic_hi < 1 and gm1.ic_lo > 0)
    log(f"A-colchón: M1 T4 AGEB IC sup {sup['ageb'].ic_hi:.3f}, H3 IC sup {sup['h3'].ic_hi:.3f} → "
        f"{'pasa' if colchon else 'NO pasa'}")
    log(f"A-piso: M0 T1 AGEB IC sup {inf['ageb'].ic_hi:.3f}; H3 IC [{inf['h3'].ic_lo:.3f}, {inf['h3'].ic_hi:.3f}] → "
        f"{'pasa' if piso else 'NO pasa'}")
    log(f"A-concentración: β {bm1.valor:.3f} IC sup {bm1.ic_hi:.3f}; γ {gm1.valor:.3f} IC inf {gm1.ic_lo:.3f} → "
        f"{'pasa' if conc else 'NO pasa'}")
    log(t.round(3).to_string(index=False))
    log(d.round(3).to_string(index=False))
    guarda_json({"parte1": {"tramos": t.to_dict("records"), "A_colchon": colchon, "A_piso": piso},
                 "parte2": {"densidad": d.to_dict("records"), "A_concentracion": conc}}, "veredictos.json")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, mod in zip(axs, ["M0", "M1"]):
        for k, (u, col) in enumerate([("ageb", "#1f4e79"), ("h3", "#e67e22")]):
            g = t[(t.unidad == u) & (t.modelo == mod)]
            x = np.arange(4) + (k - 0.5) * 0.18
            ax.errorbar(x, g.beta, yerr=[g.beta - g.ic_lo, g.ic_hi - g.beta], fmt="o", color=col, capsize=3,
                        label="AGEB" if u == "ageb" else "H3 res. 8")
        ax.axhline(1, color="#c0392b", ls="--", lw=1)
        ax.set_xticks(range(4), ["Q1\n(pocos negocios)", "Q2", "Q3", "Q4\n(muchos negocios)"])
        ax.set_title(f"{mod}: β por cuartil de E (IC 95%, bootstrap de bloques)", loc="left", fontsize=9)
    axs[0].set_ylabel("β del tramo")
    axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig_T7_parte1_beta_tramos.png")
    log("listo")
