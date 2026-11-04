"""S1 · Variable objetivo principal (regla en analisis/lab/README.md).

Métrica de decisión: % del top 10% por P(θ > 1) de M1 ocupado por AGEB con autoservicio grande.
Además, como ROBUSTEZ (no entra en la regla): β de M0, ΔD M0 → M1, Jaccard conteo vs residual y persistencia.
"""
import numpy as np
import pandas as pd

from lab_comun import (DEFAULT, RES, ajusta_nb, cols_m0, cols_m1, cv_espacial, datos, jaccard, persistencia,
                       ranking_residual, top)

filas = []
for v in ["V1", "V2", "V2b", "V3"]:
    for lista in ["b", "a"]:  # (b) es el valor por defecto; (a) comprueba que S1 no dependa de la lista
        p = datos(y=v, lista=lista)
        c0, c1 = cols_m0(p), cols_m1(p)
        mu1, dev1 = cv_espacial(p, c1)
        g = ranking_residual(p, mu1)
        t_res = top(g.score)
        fila = {"variable": v, "lista": lista,
                "pct_top10_autoserv_grande": np.mean([g.ag[i] > 0 for i in t_res]) * 100,
                "pct_base_autoserv_grande": (g.ag > 0).mean() * 100,
                "max_pct_ceros_anual": p.groupby("anio").y.apply(lambda s: (s == 0).mean() * 100).max()}
        if lista == DEFAULT["lista"]:
            mu0, dev0 = cv_espacial(p, c0)
            m0 = ajusta_nb(p[c0].values, p.y.values)
            mejora = {f: (dev0[f] - dev1[f]) / dev0[f] for f in dev0}
            fila |= {"rob_beta_M0": m0.params[1], "rob_beta_M0_ee": m0.bse[1],
                     "rob_dev_M0": np.mean(list(dev0.values())), "rob_dev_M1": np.mean(list(dev1.values())),
                     "rob_deltaD_pct": np.mean(list(mejora.values())) * 100,
                     "rob_folds_mejora": sum(m > 0 for m in mejora.values()),
                     "rob_jaccard_conteo_vs_residual": jaccard(top(g.O), t_res),
                     "rob_persistencia_rho": persistencia(p, mu1)}
        filas.append(fila)
        print(fila)

pd.DataFrame(filas).to_csv(RES / "s1.csv", index=False)
