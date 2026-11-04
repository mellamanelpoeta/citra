"""S7 parte 2 · Backtest: con datos hasta t, ¿qué lista anticipa mejor el exceso de t + 1? (analisis/lab/README.md §S7)

Pares 2022 → 2023 y 2023 → 2024. Ê de M2 reentrenado solo con años ≤ t (validación espacial). Opciones de lista:
(a) periodo agregado con igual peso, (b) últimos 2 años, (c) pesos con vida media de 1.5 años.
Métricas en t + 1: ρ(rr anticipado, rr observado), % del top 10% con P(θ > 1) > 0.5, capture del exceso
Σ(O − Ê)⁺ y, por continuidad con 04c, % del top con O > Ê. Aparte: ρ con O adelgazado al volumen de V2b (S6).

Uso: .venv/bin/python analisis/lab/s7_backtest_2025.py
Salida: analisis/lab/resultados/s7_backtest.csv
"""
from s7_comun import RES, VARIANTES, backtest, cols, contraccion, etiqueta, lista, panel  # noqa: I001

import numpy as np
import pandas as pd
from scipy import stats

from modelado import SEED, top

PARES = [2022, 2023]
B_THIN = 50


def evalua(q, t, L):
    """Métricas en t + 1 de la lista L (DataFrame con rr y score por AGEB)."""
    n = q[q.anio == t + 1].set_index("cvegeo")
    s = contraccion(n.y, n.mu)
    n["rr"], n["p_exc"] = s["rr"], s["p_exc"]
    T = list(top(L.score))
    exc = (n.y - n.mu).clip(lower=0)
    return {"rho": stats.spearmanr(L.rr, n.rr.reindex(L.index))[0],
            "pct_top_p_exc_05": 100 * float((n.p_exc.reindex(T) > 0.5).mean()),
            "pct_top_O_mayor_E": 100 * float((n.y.reindex(T) > n.mu.reindex(T)).mean()),
            "capture_exceso": 100 * float(exc.reindex(T).sum() / exc.sum())}


n_v2b = panel("V2b").y.sum()
filas = []
for var, ctl in VARIANTES:
    p = panel(var)
    _, _, c2 = cols(p, ctl)
    for t in PARES:
        q = backtest(p, c2, t)
        for op in "abc":
            fila = {"variante": etiqueta(var, ctl), "par": f"{t}→{t + 1}", "opcion": op, **evalua(q, t, lista(q, t, op))}
            if op == "a":
                # control de volumen (S6): O adelgazado binomialmente al total de V2b y Ê × π, sin reajustar el modelo
                pi = min(1.0, n_v2b / p.y.sum())
                rng = np.random.default_rng(SEED)
                rhos = []
                for _ in range(B_THIN if pi < 1 else 1):
                    O = rng.binomial(q.y.values.astype(int), pi).astype(float) if pi < 1 else q.y.values
                    qt = q.assign(y=O, mu=q.mu * pi)
                    rhos.append(evalua(qt, t, lista(qt, t, "a"))["rho"])
                fila["rho_adelgazado_mediana"] = float(np.median(rhos))
                fila["rho_adelgazado_p05"] = float(np.percentile(rhos, 5))
            filas.append(fila)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in fila.items()}, flush=True)

r = pd.DataFrame(filas)
r.to_csv(RES / "s7_backtest.csv", index=False, float_format="%.4f")
print(r.pivot_table(index=["variante", "opcion"], columns="par", values="rho").round(3))
