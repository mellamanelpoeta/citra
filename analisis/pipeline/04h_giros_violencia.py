"""04h · Razón de tasas por giro, con y sin violencia, para todos los sectores de la composición de M1. Análisis
exploratorio posterior (2026-10-14, para la gráfica de giro de la app): no modifica veredictos ni
salidas de 04_modelos.py ni de 04c.

04c (bloque "¿0.78 es menos robo o menos denuncia?") solo guardó los dos grupos de efectivo. Aquí se repite el mismo
bootstrap (mismos datos, columnas, semilla y B) pidiendo todos los coeficientes s_*, así que los dos grupos de efectivo
reproducen las cifras de robustez_C.csv. Todo el robo (V2b) va como referencia, con el mismo esquema.

Supuesto de lectura: cada razón es el cambio del robo si 10 de cada 100 negocios de la zona pasan de servicios
profesionales y financieros (sector de referencia, = 1 por construcción) a ese grupo, con lo demás fijo. Solo se lee
donde +10 de cada 100 cabe en los datos (rango P10–P90 entre AGEB ≥ 10 puntos); la columna `rango_p10_p90` lo dice.

Uso: .venv/bin/python analisis/pipeline/04h_giros_violencia.py
Salida: analisis/pipeline/outputs/modelos/giros_violencia.csv
"""
import importlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import lee_panel
from modelado import OUT_MOD, SEED, bloques, cols_comp, cols_m1, datos

s04b = importlib.import_module("04b_sesgo_denuncia")
B = 500  # como 04c


def boot_coefs(p, cols, nombres):
    """Copia de 04c.boot_coefs (importar 04c lo ejecuta completo)."""
    X, y = p[cols].values.astype(float), p.y.values
    grupos = {b_: g.index.values for b_, g in p.groupby(p.cvegeo.map(bloques().bloque))}
    ids = np.array(list(grupos))
    rng = np.random.default_rng(SEED)
    muestras = [np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)]) for _ in range(B)]
    k = [cols.index(n) + 1 for n in nombres]
    out = np.array(Parallel(n_jobs=-1)(delayed(lambda i: s04b.ajusta(X[i], y[i]).params[k])(i) for i in muestras))
    punto = s04b.ajusta(X, y).params[k]
    return {n: (punto[j], *np.nanpercentile(out[:, j], [2.5, 97.5])) for j, n in enumerate(nombres)}


if __name__ == "__main__":
    p = datos("V2b")
    tipos = s04b.panel_por_tipo(p)
    pan = lee_panel()
    at = pan[pan.anio == 2023]

    filas = []
    for k, nom, q in [("total", "todo el robo", p), ("cv", "con violencia", tipos["cv"]), ("sv", "sin violencia", tipos["sv"])]:
        cols = cols_m1(q)
        comp = cols_comp(q)
        print(f"== {nom}: {len(comp)} grupos")
        for n, (pt, lo, hi) in boot_coefs(q, cols, comp).items():
            s = at[f"ps_{n[2:]}_fijo"]
            filas.append({"tipo": k, "grupo": n[2:], "irr10": np.exp(0.1 * pt), "ic_lo": np.exp(0.1 * lo),
                          "ic_hi": np.exp(0.1 * hi), "rango_p10_p90": s.quantile(0.9) - s.quantile(0.1)})
            print(f"  {n[2:]:<18} {np.exp(0.1 * pt):.3f} [{np.exp(0.1 * lo):.3f}, {np.exp(0.1 * hi):.3f}]")

    pd.DataFrame(filas).to_csv(OUT_MOD / "giros_violencia.csv", index=False, float_format="%.4f")
    print(f"→ {OUT_MOD / 'giros_violencia.csv'}")
