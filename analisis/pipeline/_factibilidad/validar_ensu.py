"""Valida el cálculo de ensu_alcaldia.py contra los tabulados oficiales de INEGI (dic-2024).

Para comparar manzanas con manzanas se usa la definición de INEGI:
- BP1_1 y BP1_4_3: el "No sabe / no responde" (9) SÍ entra al denominador.
- BP1_2_xx: el denominador es la población que usa el lugar (excluye "No aplica" = 3).
- Victimización del hogar (tabulado 7.3): robo (BP1_6_1..5) + extorsión (BP1_6_6)
  + fraude bancario (BP1_6_7) + fraude al consumidor (BP1_6_8), con FAC_VIV.

Uso:  .venv/bin/python analisis/pipeline/_factibilidad/validar_ensu.py
"""
from __future__ import annotations

import os
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import ensu_alcaldia as ea  # noqa: E402
from _xlsx_min import read  # noqa: E402

TAB = os.path.join(ea.RAW, "ensu_docs", "tabulados")
EST = os.path.join(TAB, "ensu_tabulados_basicos_diciembre_2024_estimacion")
ERR = os.path.join(TAB, "ensu_tabulados_basicos_diciembre_2024_error")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().strip().lower()
    return s


NOMBRES = {norm(v): k for k, v in ea.ALCALDIAS.items()}


def tabla_bloques(path: str, hoja: str, col: str = "E") -> dict:
    """{(cve_mun, etiqueta_normalizada): valor} para bloques 'Alcaldía' seguidos de filas de categoría."""
    out, actual, en_cdmx = {}, None, False
    for _, r in read(path, hoja):
        a = (r.get("A") or "").strip()
        if a == "Ciudad de México":
            en_cdmx = True
            continue
        if not en_cdmx or not a:
            continue
        if norm(a) in NOMBRES:
            actual = NOMBRES[norm(a)]
            if r.get(col):  # tabulado de una sola fila por alcaldía (7.3)
                out[(actual, "_fila")] = float(r[col])
            continue
        if a.startswith("Región"):
            continue
        if actual and r.get(col):
            out[(actual, norm(a)[:20])] = float(r[col])
        if len(out) and actual and norm(a) not in NOMBRES and not r.get(col) and not a.startswith("Región"):
            break  # salió del bloque de la CDMX
    return out


def main():
    f = [x for x in ea.archivos() if x[0] == 2024 and x[1] == 4][0]
    cb = ea.cargar_trimestre(*f)
    of = ea.indicadores(cb, denom_oficial=True).set_index("cve_mun")
    num = ea.num
    # victimización total del hogar (definición del tabulado 7.3 desde 2S-2024)
    r = pd.concat([num(cb[f"BP1_6_{i}"]) for i in range(1, 9)], axis=1)
    vict = []
    for cve in ea.ALCALDIAS:
        d = cb["cve_mun"] == cve
        p, se, _ = ea.prop_y_se(cb, (r == 1).any(axis=1).astype(float), pd.Series(True, index=cb.index),
                                cb["w_viv"], d)
        vict.append((cve, p, se))
    vict = pd.DataFrame(vict, columns=["cve_mun", "pct_vict_total", "se_vict_total"]).set_index("cve_mun")

    t17 = tabla_bloques(os.path.join(EST, "1.Bas_percepcion_seguridad_dic_2024_est.xlsx"), "1.7")
    e17 = tabla_bloques(os.path.join(ERR, "1.Bas_percepcion_seguridad_dic_2024_err.xlsx"), "1.7")
    t19 = tabla_bloques(os.path.join(EST, "1.Bas_percepcion_seguridad_dic_2024_est.xlsx"), "1.9")
    t33 = tabla_bloques(os.path.join(EST, "3.Bas_atestiguacion_dic_2024_est.xlsx"), "3.3")
    t73 = tabla_bloques(os.path.join(EST, "7.Bas_delitos_al_hogar_dic_2024_est.xlsx"), "7.3")

    filas = []
    for cve, nom in ea.ALCALDIAS.items():
        o = of.loc[cve]
        filas.append({
            "alcaldia": nom,
            "inseg_ciudad_ofi": t17.get((cve, "inseguro")), "inseg_ciudad_calc": o.pct_inseg_ciudad,
            "se_ofi": e17.get((cve, "inseguro")), "se_calc": o.se_inseg_ciudad,
            "mercado_ofi": t19.get((cve, "mercado")), "mercado_calc": o.pct_inseg_mercado,
            "cc_ofi": t19.get((cve, "centro comercial")), "cc_calc": o.pct_inseg_centro_comercial,
            "calles_ofi": t19.get((cve, "calles que habitualm")), "calles_calc": o.pct_inseg_calle,
            "robos_ofi": t33.get((cve, "robos o asaltos")), "robos_calc": o.pct_atestigua_robos,
            "vict_ofi": t73.get((cve, "_fila")), "vict_calc": vict.loc[cve, "pct_vict_total"],
        })
    v = pd.DataFrame(filas)
    pd.set_option("display.width", 250)
    print(v.round(2).to_string(index=False))
    pares = [("inseg_ciudad", "inseg_ciudad_ofi", "inseg_ciudad_calc"), ("se", "se_ofi", "se_calc"),
             ("mercado", "mercado_ofi", "mercado_calc"), ("centro_comercial", "cc_ofi", "cc_calc"),
             ("calles", "calles_ofi", "calles_calc"), ("robos", "robos_ofi", "robos_calc"),
             ("vict_hogar", "vict_ofi", "vict_calc")]
    print("\nMáxima diferencia absoluta (pp) calculado − oficial, 16 alcaldías, dic-2024:")
    for n, a, b in pares:
        dif = (v[b] - v[a]).abs()
        print(f"  {n:18s} max={dif.max():.4f}  n_comparadas={dif.notna().sum()}")


def main_2019():
    """Segunda verificación (era DBF, primer trimestre con alcaldías publicado en tabulados): dic-2019."""
    f = [x for x in ea.archivos() if x[0] == 2019 and x[1] == 4][0]
    of = ea.indicadores(ea.cargar_trimestre(*f), denom_oficial=True).set_index("cve_mun")
    t = tabla_bloques(os.path.join(TAB, "ensu_tabulados_basicos_diciembre_2019_estimaciones",
                                   "I_percepcion_dic_2019_est.xlsx"), "1.4")
    dif = pd.Series({c: of.loc[c, "pct_inseg_ciudad"] - t.get((c, "inseguro"), np.nan) for c in ea.ALCALDIAS})
    print(f"\nDic-2019, % inseguridad en la ciudad (BP1_1): max |calc − oficial| = {dif.abs().max():.4f} pp,"
          f" n_comparadas={dif.notna().sum()}")
    t = tabla_bloques(os.path.join(TAB, "ensu_tabulados_basicos_diciembre_2019_estimaciones",
                                   "X_delitos_contra_integrantes_del_hogar_dic_2019_est.xlsx"), "10.3")
    dif = pd.Series({c: of.loc[c, "pct_hog_vict_robo"] - t.get((c, "_fila"), np.nan) for c in ea.ALCALDIAS})
    print(f"Dic-2019, % hogares víctimas de robo (BP1_6_1..5, FAC_VIV): max |calc − oficial| ="
          f" {dif.abs().max():.4f} pp, n_comparadas={dif.notna().sum()}")


if __name__ == "__main__":
    main()
    main_2019()
