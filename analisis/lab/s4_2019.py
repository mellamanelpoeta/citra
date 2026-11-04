"""S4 · ¿Entra 2019 en V1 y V3? (regla en analisis/lab/README.md)

(1) β de M0 con 2019–2024 vs IC 95% (bootstrap de bloques espaciales, 200 réplicas) con 2020–2024.
(2) Spearman entre rankings P(θ > 1) del periodo 2020–2024, con modelos M1 entrenados con y sin 2019.
"""
import numpy as np
import pandas as pd
from scipy import stats

from lab_comun import (RES, SEED, ajusta_nb, bloques, cols_m0, cols_m1, cv_espacial, datos, ranking_residual)

B = 200
rng = np.random.default_rng(SEED)
blq = bloques().bloque
filas = []
for v in ["V1", "V3"]:
    con = datos(y=v, anios=list(range(2019, 2025)))
    sin = datos(y=v, anios=list(range(2020, 2025)))

    # (1) β M0 y bootstrap de bloques espaciales (sobre la muestra sin 2019)
    b_con = ajusta_nb(con[cols_m0(con)].values, con.y.values).params[1]
    b_sin = ajusta_nb(sin[cols_m0(sin)].values, sin.y.values).params[1]
    c0 = cols_m0(sin)
    sin["bloque"] = sin.cvegeo.map(blq)
    grupos = {b_: g.index.values for b_, g in sin.groupby("bloque")}
    ids = np.array(list(grupos))
    boot = []
    for _ in range(B):
        idx = np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)])
        s = sin.loc[idx]
        boot.append(ajusta_nb(s[c0].values, s.y.values).params[1])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # (2) rankings 2020–2024 con modelos entrenados con y sin 2019 (predicción fuera de muestra espacial)
    mu_con, _ = cv_espacial(con, cols_m1(con))
    mu_sin, _ = cv_espacial(sin, cols_m1(sin))
    m20 = (con.anio >= 2020).values
    r_con = ranking_residual(con[m20].reset_index(drop=True), mu_con[m20])
    r_sin = ranking_residual(sin.drop(columns="bloque"), mu_sin)
    rho = stats.spearmanr(r_con.score, r_sin.score.reindex(r_con.index))[0]

    fila = {"variable": v, "beta_con_2019": b_con, "beta_sin_2019": b_sin, "ic95_sin_lo": lo, "ic95_sin_hi": hi,
            "beta_con_dentro_ic": bool(lo <= b_con <= hi), "spearman_rankings": rho}
    filas.append(fila)
    print(fila)
pd.DataFrame(filas).to_csv(RES / "s4.csv", index=False)
