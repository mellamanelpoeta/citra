"""Lista de giros intensivos en efectivo: (a) tentativa, (b) dos grupos, (c) sin grupo (regla en analisis/lab/README.md).

Se evalúa sobre la variable que gana S1 (se lee de resultados/s1.csv aplicando la regla de S1).
Métrica: devianza Poisson fuera de muestra de M1, por fold espacial. Robustez: ΔD M0 → M1 con cada lista.
"""
import pandas as pd

from lab_comun import RES, cols_m0, cols_m1, cv_espacial, datos
from decidir import regla_s1

v = regla_s1(pd.read_csv(RES / "s1.csv"))["eleccion"]
print("variable (S1):", v)
p0 = datos(y=v, lista="c")
_, dev0 = cv_espacial(p0, cols_m0(p0))
filas = []
for lista in ["a", "b", "c"]:
    p = datos(y=v, lista=lista)
    _, dev = cv_espacial(p, cols_m1(p))
    for f in dev:
        filas.append({"variable": v, "lista": lista, "fold": f, "dev_M1": dev[f], "dev_M0": dev0[f]})
r = pd.DataFrame(filas)
r.to_csv(RES / "lista_giros.csv", index=False)
print(r.pivot(index="fold", columns="lista", values="dev_M1").round(4))
print(r.groupby("lista").dev_M1.mean().round(4))
