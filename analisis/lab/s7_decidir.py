"""S7 · Aplica las reglas ex ante de analisis/lab/README.md §S7 a s7_variable.csv y s7_backtest.csv.

Uso: .venv/bin/python analisis/lab/s7_decidir.py   (después de s7_variable_cadenas.py y s7_backtest_2025.py)
Salida: analisis/lab/resultados/s7_reglas.md
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent / "resultados"
v = pd.read_csv(RES / "s7_variable.csv").set_index("variante")
bt = pd.read_csv(RES / "s7_backtest.csv")

# ------------------------------------------------------------ parte 1
a = bt[bt.opcion == "a"].groupby("variante")[["rho", "pct_top_p_exc_05", "rho_adelgazado_mediana"]].mean()
v = v.join(a.add_prefix("bt_"))
base = v.loc["V2b(0)"]
cand = v.loc[["V1(1)", "V1c(1)"]].copy()
cand["r1_validez"] = cand.pct_autoserv_top10 <= base.pct_autoserv_top10 + 3
cand["r2_backtest"] = (cand.bt_rho >= base.bt_rho + 0.03) | (cand.bt_pct_top_p_exc_05 >= base.bt_pct_top_p_exc_05 + 3)
cand["r3_calibracion"] = (cand.calib_min >= 0.85) & (cand.calib_max <= 1.15)
cand["pasa"] = cand.r1_validez & cand.r2_backtest & cand.r3_calibracion
pasan = cand[cand.pasa]
if pasan.empty:
    rec1 = "Mantener V2b como variable principal (ninguna candidata cumple las tres condiciones)."
elif len(pasan) == 2:
    # "V1c gana en (1) y (2)": menos autoservicio en el top y mayor ρ medio del backtest (a)
    gana = (cand.loc["V1c(1)", "pct_autoserv_top10"] < cand.loc["V1(1)", "pct_autoserv_top10"]) and \
           (cand.loc["V1c(1)", "bt_rho"] > cand.loc["V1(1)", "bt_rho"])
    rec1 = f"Cambiar a {'V1c' if gana else 'V1'} con control de cadenas (pasan las dos; " \
           f"{'V1c gana en (1) y (2)' if gana else 'se prefiere V1 por simplicidad'})."
else:
    rec1 = f"Cambiar a {pasan.index[0][:-3]} con control de cadenas."

t1 = pd.concat([base.to_frame().T, cand])[
    ["carpetas", "pct_autoserv_top10", "bt_rho", "bt_pct_top_p_exc_05", "calib_min", "calib_max",
     "r1_validez", "r2_backtest", "r3_calibracion", "pasa", "bt_rho_adelgazado_mediana"]]

# ------------------------------------------------------------ parte 2
filas = []
for var, g in bt.groupby("variante", sort=False):
    r = g.pivot(index="opcion", columns="par", values="rho")
    supera = {op: bool(((r.loc[op] - r.loc["a"]) >= 0.03).all()) for op in "bc"}
    ganan = [op for op, s in supera.items() if s]
    elige = "a" if not ganan else max(ganan, key=lambda op: r.loc[op].mean())
    filas.append({"variante": var, **{f"rho_{op}_{par}": r.loc[op, par] for op in "abc" for par in r.columns},
                  "b_supera_a": supera["b"], "c_supera_a": supera["c"], "opcion_elegida": elige})
t2 = pd.DataFrame(filas).set_index("variante")
tbt = bt.pivot_table(index=["variante", "opcion"], columns="par",
                     values=["rho", "pct_top_p_exc_05", "pct_top_O_mayor_E", "capture_exceso"]).round(3)
tbt.columns = [f"{m} {p}" for m, p in tbt.columns]

md = f"""# S7 · Aplicación de las reglas ex ante (salida automática de `s7_decidir.py`)

## Parte 1: ¿cambiar la variable principal?

Condiciones (README §S7): (1) % de autoservicio grande en el top 10% ≤ V2b(0) + 3 pp; (2) backtest (a), media de los dos
pares: ρ ≥ V2b(0) + 0.03 **o** % del top con P(θ>1) > 0.5 ≥ V2b(0) + 3 pp; (3) O/Ê de cada decil de Ê en [0.85, 1.15].
La última columna (ρ adelgazado al volumen de V2b) se reporta y no entra en la regla.

{t1.to_markdown(floatfmt=".3f")}

**Recomendación de la regla:** {rec1}

## Parte 2: ¿cómo se arma la lista para el año siguiente?

Regla: se queda (a) salvo que (b) o (c) la superen en ≥ 0.03 de ρ en ambos pares.

{t2.to_markdown(floatfmt=".3f")}

### Detalle del backtest

{tbt.to_markdown()}
"""
(RES / "s7_reglas.md").write_text(md, encoding="utf-8")
print(md)
