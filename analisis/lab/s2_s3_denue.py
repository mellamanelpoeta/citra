"""S2 (corte DENUE en validación espacial) y S3 (validación temporal 2024). Reglas en analisis/lab/README.md."""
import numpy as np
import pandas as pd

from lab_comun import RES, DEFAULT, ajusta_nb, capture, cols_m1, cv_espacial, datos, devianza, predice

v = DEFAULT["y"]

# ---------------------------------------------------------------- S2: espacial, 2020–2024
filas = []
for e in ["fijo", "nov"]:
    p = datos(y=v, e=e)
    mu, dev = cv_espacial(p, cols_m1(p))
    fila = {"e": e, "n": len(p), "dev_media_folds": np.mean(list(dev.values()))}
    fila |= {f"dev_{a}": devianza(p.y[p.anio == a], mu[p.anio == a]) for a in sorted(p.anio.unique())}
    filas.append(fila)
    print(fila)
pd.DataFrame(filas).to_csv(RES / "s2.csv", index=False)

# ---------------------------------------------------------------- S3: temporal, entrenar 2020–2023 → 2024
filas = []
for e in ["fijo", "nov"]:  # (a) E 2023-11 para 2024 = "fijo"; (b) E 2024-11 = "nov" (entrenando también con nov)
    p = datos(y=v, e=e)
    # El efecto fijo de 2024 no es estimable al predecir el año siguiente. Supuesto: persistencia del nivel,
    # 2024 hereda el efecto de 2023 (igual en ambas variantes, así que la comparación es justa).
    c = [x for x in cols_m1(p) if x != "fe_2024"]
    p.loc[p.anio == 2024, "fe_2023"] = 1.0
    tr, te = p.anio <= 2023, p.anio == 2024
    m = ajusta_nb(p.loc[tr, c].values, p.loc[tr, "y"].values)
    mu = predice(m, p.loc[te, c].values)
    y = p.loc[te, "y"].values
    fila = {"variante": "a (E 2023-11)" if e == "fijo" else "b (E 2024-11)", "n_2024": int(te.sum()),
            "dev_2024": devianza(y, mu), "mae_2024": float(np.mean(np.abs(y - mu))), "capture_top10_2024": capture(y, mu)}
    filas.append(fila)
    print(fila)
pd.DataFrame(filas).to_csv(RES / "s3.csv", index=False)
