"""08b · Descriptivo de rotación para la app (página Contexto): no prueba hipótesis ni cambia veredictos de 08.

1. Dos ritmos del DENUE: entre censos (2019-11 → 2023-11) casi no se actualiza; al integrar el Censo Económico 2024
   (2024-11) aparece el recambio de cinco años. Por eso el E de 2023-11 son, sobre todo, los negocios del censo 2019.
2. Aunque los negocios rotan, el tamaño comercial de cada AGEB casi no cambia (correlación N 2019 vs N 2024).
3. Salidas, entradas, rotación (salidas + entradas) y saldo por quintil de riesgo relativo previo (rr_pre 2016–2018).
   Tasas crudas, sin controles; las pruebas con controles son D2a/D2b de 08.

Entrada: data/processed/modelos/rotacion_ageb.parquet (08_rotacion.py).
Salidas: analisis/pipeline/outputs/dinamica/rotacion_quintiles.csv, rotacion_ritmos.csv.
Uso: .venv/bin/python analisis/pipeline/08b_rotacion_quintiles.py   (segundos)
"""
import numpy as np
import pandas as pd

from common import PROCESSED, ROOT

OUT = ROOT / "analisis/pipeline/outputs/dinamica"

r = pd.read_parquet(PROCESSED / "modelos/rotacion_ageb.parquet")
N19, N24 = r.N_2019.sum(), r.N_2024.sum()
# N 2023-11 por identidad (sin cambios de AGEB, que son 1.6% de los que siguen)
N23 = N19 - r.S23.sum() + r.En23.sum()

ritmos = pd.DataFrame([
    ("salen entre censos (2019-11 → 2023-11), % de 2019", r.S23.sum() / N19 * 100),
    ("entran entre censos (2019-11 → 2023-11), % de 2019", r.En23.sum() / N19 * 100),
    ("negocios de 2023-11 que ya estaban en 2019-11, %", (N19 - r.S23.sum()) / N23 * 100),
    ("salen en cinco años (2019-11 → 2024-11), % de 2019", r.S.sum() / N19 * 100),
    ("entran en cinco años (2019-11 → 2024-11), % de 2019", r.En.sum() / N19 * 100),
    ("saldo de la ciudad 2019 → 2024, %", (N24 / N19 - 1) * 100),
    ("correlación de log N por AGEB, 2019 vs 2024", np.corrcoef(np.log1p(r.N_2019), np.log1p(r.N_2024))[0, 1]),
    ("mediana del cambio absoluto de N por AGEB, %",
     np.median(np.abs(r.N_2024 / r.N_2019.where(r.N_2019 > 0) - 1).dropna()) * 100),
], columns=["metrica", "valor"])
ritmos.to_csv(OUT / "rotacion_ritmos.csv", index=False, float_format="%.3f")
print(ritmos.round(3).to_string(index=False))

r["q"] = pd.qcut(r.log_rr_pre.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
g = r.groupby("q", observed=True)
t = pd.DataFrame({"agebs": g.size(), "rr_pre_mediana": np.exp(g.log_rr_pre.median()),
                  "pct_salen": g.S.sum() / g.N_2019.sum() * 100,
                  "pct_entran": g.En.sum() / g.N_2019.sum() * 100})
t["pct_rotacion"] = t.pct_salen + t.pct_entran
t["pct_saldo"] = (g.N_2024.sum() / g.N_2019.sum() - 1) * 100
t.to_csv(OUT / "rotacion_quintiles.csv", float_format="%.2f")
print(t.round(2).to_string())
