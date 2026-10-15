"""04g · La lista como anticipación del año siguiente (2026-10-14; contrato v0.5).

Hasta v0.4 la lista se armaba con el periodo 2020–2024 sumado con igual peso, y la app mostraba ese acumulado. Desde v0.5
la lista dice lo que se espera para el año siguiente, con más peso a lo reciente, y se muestra cómo habría
funcionado un año atrás. Este script produce las dos cosas; la prueba registrada de C (04_modelos) no cambia.

1. Lista 2025 (anticipada). Con O y Ê (M2 fuera de muestra, de 04_modelos) de 2020–2024, cada año pesa
   w = 0.5^((2024 − año) / 1.5): vida media de 1.5 años, la fijada en S7 antes de correrlo (analisis/lab/README.md §S7).
   La contracción Poisson-gamma se hace sobre Σw·O y Σw·Ê. Esperado 2025 por su actividad = Ê de 2024 (E congelado
   en el DENUE 2023-11 y el efecto de año hereda el último, como S3); anticipado 2025 = rr × esperado 2025.
2. Prueba 2024 ("si lo hubiéramos hecho en 2024"). M2 reentrenado solo con 2020–2023 y validación espacial por
   bloques: Ê de 2020–2024 fuera de muestra en espacio y, para 2024, también en tiempo. Misma receta con t = 2023
   (lista por riesgo y lista por conteo ponderado) y se compara con lo que pasó en 2024.
3. Aviso ⚠ (v0.3) con la misma ponderación: ranking con solo robo con violencia.

Supuestos: (i) persistencia, el exceso de una zona cambia despacio (S7: 70% de la lista sigue en exceso al año
siguiente); (ii) la vida media de 1.5 años no se ajustó al resultado: S7 la registró antes de correr y en el backtest
anticipa igual o mejor que el igual peso (ρ 0.40 frente a 0.38); (iii) E congelado (DENUE 2023-11).

Uso: .venv/bin/python analisis/pipeline/04g_anticipacion.py   (~2 min)
Salidas: data/processed/modelos/anticipado.parquet, anticipado_cv.parquet, validacion_2024.parquet,
         validacion_2024_anual.parquet;
         analisis/pipeline/outputs/modelos/anticipacion.csv
"""
from modelado import lightgbm

lightgbm()  # en macOS puede relanzar el proceso: debe ir antes que todo lo demás

import importlib  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modelado import MODELOS, OUT_MOD, PRINCIPAL, ajusta_m2, bloques, cols_m2, contraccion, cv_espacial, datos, top  # noqa: E402

VIDA_MEDIA = 1.5
TOP_Q = 0.10


def pesos(anios, t):
    return 0.5 ** ((t - np.asarray(anios)) / VIDA_MEDIA)


def anticipa(q, mu, t):
    """Riesgo anticipado para t + 1 con los años ≤ t: contracción sobre Σw·O y Σw·Ê. Devuelve una fila por AGEB con
    O_w, E_w, ritmo (robos por año ponderado), rr, p_exc y score."""
    m = (q.anio <= t).values
    w = pesos(q.anio.values[m], t)
    g = pd.DataFrame({"cvegeo": q.cvegeo.values[m], "O_w": w * q.y.values[m], "E_w": w * mu[m], "w": w}).groupby("cvegeo").sum()
    s = contraccion(g.O_w, g.E_w)
    g["rr"], g["p_exc"] = s["rr"], s["p_exc"]
    g["score"] = g.p_exc + 1e-6 * g.rr
    g["ritmo"] = g.O_w / g.w
    return g


p = datos(PRINCIPAL)
anual = pd.read_parquet(MODELOS / "riesgo_anual.parquet")
mu_prod = p[["cvegeo", "anio"]].merge(anual[["cvegeo", "anio", "Ehat"]], on=["cvegeo", "anio"], how="left").Ehat.values
assert np.isfinite(mu_prod).all(), "Ê de 04_modelos desalineado con el panel"
T = int(p.anio.max())

# ============================================================ 1. Lista 2025
g = anticipa(p, mu_prod, T)
g["E_next"] = anual[anual.anio == T].set_index("cvegeo").Ehat.reindex(g.index)
g["anticipados"] = g.rr * g.E_next
g.reset_index().to_parquet(MODELOS / "anticipado.parquet", index=False)
n_top = int(round(len(g) * TOP_Q))
lista = top(g.score)
print(f"lista {T + 1}: {len(lista)} AGEB; anticipados en la lista {g.anticipados[list(lista)].sum():.0f} frente a "
      f"{g.E_next[list(lista)].sum():.0f} esperados por su actividad")

# ============================================================ 3. Aviso: solo robo con violencia, misma ponderación
s04b = importlib.import_module("04b_sesgo_denuncia")
cv = s04b.panel_por_tipo(p)["cv"]
mu_cv, _ = cv_espacial(cv, cols_m2(cv), modelo="m2")
anticipa(cv, mu_cv, T)[["score"]].reset_index().to_parquet(MODELOS / "anticipado_cv.parquet", index=False)

# ============================================================ 2. Prueba 2024
t = T - 1
c2 = cols_m2(p)
fold, bloque = p.cvegeo.map(bloques().fold).values, p.cvegeo.map(bloques().bloque).values
X, y = p[c2].values.astype(float), p.y.values
pre = (p.anio <= t).values
mu_bt = np.full(len(p), np.nan)
for f in range(5):
    # Cada fold se predice (todos los años) con un modelo que no vio ni esas AGEB ni el año T. M2 lleva el año
    # numérico: T cae del lado de t en todos los cortes (hereda su nivel, como S3 y 04c).
    mu_bt[fold == f] = ajusta_m2(X[pre & (fold != f)], y[pre & (fold != f)], bloque[pre & (fold != f)]).predict(X[fold == f])
v = anticipa(p, mu_bt, t)
d = p.loc[~pre, ["cvegeo", "y"]].assign(E=mu_bt[~pre]).set_index("cvegeo").reindex(v.index)
s = contraccion(d.y, d.E)
v["O_sig"], v["E_sig"], v["rr_sig"], v["p_exc_sig"] = d.y, d.E, s["rr"], s["p_exc"]
v["anticipados"] = v.rr * v.E_sig
v["rank_riesgo"] = v.score.rank(ascending=False, method="first").astype(int)
v["rank_conteo"] = pd.Series(np.arange(1, len(v) + 1), index=v.O_w.sort_values(ascending=False).index).reindex(v.index)
v["en_riesgo"], v["en_conteo"] = v.rank_riesgo <= n_top, v.rank_conteo <= n_top
v.reset_index().to_parquet(MODELOS / "validacion_2024.parquet", index=False)
# Serie por año del mismo modelo (sin el año evaluado), para que la app muestre la prueba sin mezclar el modelo que sí lo vio.
p[["cvegeo", "anio", "y"]].assign(Ehat=mu_bt).to_parquet(MODELOS / "validacion_2024_anual.parquet", index=False)

exc = (v.O_sig - v.E_sig).clip(lower=0)
filas = []
for nom, L in [("lista por riesgo", v.en_riesgo), ("lista por conteo", v.en_conteo),
               ("riesgo, no conteo", v.en_riesgo & ~v.en_conteo), ("conteo, no riesgo", v.en_conteo & ~v.en_riesgo),
               ("todas", pd.Series(True, index=v.index))]:
    x = v[L]
    filas.append({"lista": nom, "n": len(x), "pct_O_mayor_E": (x.O_sig > x.E_sig).mean(),
                  "pct_por_encima": (x.p_exc_sig >= 0.9).mean(), "robos": x.O_sig.sum(), "esperados": x.E_sig.sum(),
                  "anticipados": x.anticipados.sum(), "pct_exceso_ciudad": exc[L].sum() / exc.sum()})
r = pd.DataFrame(filas)
r.to_csv(OUT_MOD / "anticipacion.csv", index=False, float_format="%.4f")
print(f"prueba {T}: datos {int(p.anio.min())}–{t} → {T}")
print(r.round(3).to_string(index=False))
print(f"Jaccard lista {T + 1} (ponderada) vs lista del periodo con igual peso: "
      f"{len(lista & top(pd.read_parquet(MODELOS / 'riesgo_periodo.parquet').set_index('cvegeo').score)) / n_top:.0%} en común")
