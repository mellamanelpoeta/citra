"""T7 · Parte 0c: diagnóstico de V5 escrito DESPUÉS de ver que el conteo de P(θ>1) > 0.9 es degenerado bajo el azar
(si la varianza estimada → 0, la previa colapsa en su media y P(θ>1) vale 0 o 1 para todas las AGEB a la vez).

Estadístico bien definido: con la previa (a, b) ESTIMADA EN LOS DATOS REALES fija, ¿cuántas AGEB tendrían
P(θ>1 | O*) > 0.9 si O* ~ Poisson(c·Ê)? Mide cuántas "alertas" produce el puro ruido Poisson con la misma contracción
que usa la app. No cambia el veredicto registrado de V5.
"""
from t7_comun import md, RES, SEED, Log, guarda_json  # noqa: I001

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from common import ROOT  # noqa: E402

log = Log("parte0c")
g = pd.read_parquet(ROOT / "data/processed/modelos/riesgo_periodo.parquet")
cE = g.O.sum() / g.Ehat.sum() * g.Ehat.values
real = md.contraccion(g.O.values, cE)
a, b = real["a"], real["b"]
n_real = int((real["p_exc"] > 0.9).sum())
rng = np.random.default_rng(SEED)
n_sim = np.array([int((stats.gamma.sf(1, a + o, scale=1 / (b + cE)) > 0.9).sum())
                  for o in (rng.poisson(cE) for _ in range(500))])
p99 = float(np.percentile(n_sim, 99))
# fracción de simulaciones de parte0/parte0b con n ≥ n_real (estadístico tal como está escrito)
s0 = pd.read_csv(RES / "parte0_V5_simulaciones.csv").n_p09
s1 = pd.read_csv(RES / "parte0b_V5_simulaciones_reescaladas.csv").n_p09
out = {"a": a, "b": b, "n_real": n_real, "n_sim_media": float(n_sim.mean()), "n_sim_p99": p99,
       "n_sim_max": int(n_sim.max()), "pasa": bool(n_real > p99),
       "frac_sims_n_ge_real_escrito": float((s0 >= n_real).mean()),
       "frac_sims_n_ge_real_reescalado": float((s1 >= n_real).mean())}
log(out)
guarda_json({"parte0c": out})
