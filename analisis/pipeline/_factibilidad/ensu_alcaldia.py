"""ENSU (INEGI) por alcaldía de la CDMX: tabla armonizada alcaldía × año × trimestre.

Lee los microdatos trimestrales (tabla ENSU_CB = cuestionario del informante
seleccionado de 18+ años) en data/raw/_samples/encuestas/ensu<AAAA>/<mes>_<csv|dbf>/
y escribe data/interim/ensu_alcaldia.parquet.

Uso:  .venv/bin/python analisis/pipeline/_factibilidad/ensu_alcaldia.py [--validar]

Supuestos (detalle en docs/_factibilidad/03-ensu.md):
- Solo trimestres con la CDMX desagregada en 16 alcaldías (CD 71–86, desde sep-2019).
  En mar/jun-2019 la CDMX venía en 4 regiones (CD 15–18) y se excluye.
- Porcentajes ponderados en escala 0–100. Denominador = respuestas válidas:
  se excluye "No sabe / no responde" (9) y, en los lugares (BP1_2_xx), "No aplica" (3).
  Ojo: INEGI sí deja el 9 en el denominador en sus tabulados (ver --validar).
- Personas: FAC_SEL. Hogares (victimización BP1_6_x, solo jun y dic): FAC_VIV
  (en la tabla CB desde jun-2022; antes se une desde ENSU_VIV).
- Errores estándar: linealización de Taylor (conglomerado último), estratos EST_DIS,
  UPM_DIS como unidad primaria, alcaldía como dominio dentro de la CDMX.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import pyogrio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "data", "raw", "_samples", "encuestas")
OUT = os.path.join(ROOT, "data", "interim", "ensu_alcaldia.parquet")

MES = {"marzo": 1, "junio": 2, "septiembre": 3, "diciembre": 4}

# CD (ciudad de interés ENSU) = CVE_MUN + 69 (verificado en todos los trimestres sep-2019 → dic-2025)
ALCALDIAS = {
    "002": "Azcapotzalco", "003": "Coyoacán", "004": "Cuajimalpa de Morelos",
    "005": "Gustavo A. Madero", "006": "Iztacalco", "007": "Iztapalapa",
    "008": "La Magdalena Contreras", "009": "Milpa Alta", "010": "Álvaro Obregón",
    "011": "Tláhuac", "012": "Tlalpan", "013": "Xochimilco", "014": "Benito Juárez",
    "015": "Cuauhtémoc", "016": "Miguel Hidalgo", "017": "Venustiano Carranza",
}

# indicador -> (variable, código "sí", códigos válidos para el denominador)
PERSONA = {
    "inseg_ciudad": ("BP1_1", 2, (1, 2)),              # ¿vivir en (CIUDAD) es seguro/inseguro?
    "inseg_calle": ("BP1_2_03", 2, (1, 2)),            # calles que habitualmente usa / la calle (2025)
    "inseg_mercado": ("BP1_2_05", 2, (1, 2)),
    "inseg_centro_comercial": ("BP1_2_06", 2, (1, 2)),
    "inseg_cajero": ("BP1_2_08", 2, (1, 2)),            # cajero automático en vía pública
    "inseg_transporte": ("BP1_2_09", 2, (1, 2)),        # transporte público
    "atestigua_robos": ("BP1_4_3", 1, (1, 2)),          # robos o asaltos en los alrededores de la vivienda
}
ROBOS_HOGAR = ["BP1_6_1", "BP1_6_2", "BP1_6_3", "BP1_6_4", "BP1_6_5"]  # definición INEGI de "robo"


def archivos():
    out = []
    for f in sorted(glob.glob(f"{RAW}/ensu20*/*_*/ENSU_CB_*")):
        m = re.search(r"ensu(\d{4})/(\w+?)_(csv|dbf)/", f)
        if m is None or "sec4" in f:
            continue
        out.append((int(m.group(1)), MES[m.group(2)], f))
    return out


def leer(f: str) -> pd.DataFrame:
    if f.endswith(".dbf"):
        df = pyogrio.read_dataframe(f, read_geometry=False, encoding="latin1")
    else:
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    return df


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"": None, "-": None, "None": None}),
                         errors="coerce")


def cargar_trimestre(anio: int, trim: int, f: str) -> pd.DataFrame | None:
    cb = leer(f)
    cb = cb[num(cb["CVE_ENT"]) == 9].copy()
    cb["CD_N"] = num(cb["CD"])
    if not cb["CD_N"].between(71, 86).all():
        return None  # CDMX en regiones (mar/jun-2019)
    cb["cve_mun"] = num(cb["CVE_MUN"]).astype(int).map(lambda x: f"{x:03d}")
    assert ((cb["CD_N"] - 69).astype(int).map(lambda x: f"{x:03d}") == cb["cve_mun"]).all()

    if "FAC_VIV" not in cb.columns:  # se une desde la tabla de vivienda
        viv = leer(f.replace("ENSU_CB_", "ENSU_VIV_").replace("sec1_2_3_", ""))
        key = ["ID_VIV"] if "ID_VIV" in cb.columns and "ID_VIV" in viv.columns else ["UPM", "VIV_SEL"]
        viv = viv[key + ["FAC_VIV"]]
        assert not viv.duplicated(key).any()
        cb = cb.merge(viv, on=key, how="left", validate="1:1")
        assert cb["FAC_VIV"].notna().all()

    cb["w_sel"] = num(cb["FAC_SEL"])
    cb["w_viv"] = num(cb["FAC_VIV"])
    cb["anio"], cb["trimestre"] = anio, trim
    # bloque de victimización del hogar: BP1_6_1 con códigos 1/2/9 (en mar/sep BP1_6_xx son fuentes de información)
    b61 = num(cb["BP1_6_1"]) if "BP1_6_1" in cb.columns else pd.Series(np.nan, index=cb.index)
    vals = set(b61.dropna().unique())
    cb["_vict_block"] = bool(vals) and vals <= {1, 2, 9} and 2 in vals and not (anio == 2020 and trim == 3)
    return cb


def prop_y_se(d: pd.DataFrame, y: pd.Series, valido: pd.Series, w: pd.Series, dominio: pd.Series):
    """Proporción ponderada en el dominio y su EE por linealización (conglomerado último)."""
    m = dominio & valido
    W = w[m].sum()
    if W <= 0:
        return np.nan, np.nan, 0
    p = (w[m] * y[m]).sum() / W
    z = np.where(m, w * (y.fillna(0) - p) / W, 0.0)
    t = pd.DataFrame({"h": d["EST_DIS"].astype(str).values, "u": d["UPM_DIS"].astype(str).values, "z": z})
    zu = t.groupby(["h", "u"])["z"].sum().reset_index()
    var = 0.0
    for _, g in zu.groupby("h"):
        nh = len(g)
        if nh > 1:
            var += nh / (nh - 1) * ((g["z"] - g["z"].mean()) ** 2).sum()
    return 100 * p, 100 * np.sqrt(var), int(m.sum())


def indicadores(cb: pd.DataFrame, denom_oficial: bool = False) -> pd.DataFrame:
    filas = []
    vict = bool(cb["_vict_block"].iloc[0])
    for cve, nom in ALCALDIAS.items():
        dom = cb["cve_mun"] == cve
        fila = {"cve_mun": cve, "alcaldia": nom, "anio": int(cb["anio"].iloc[0]),
                "trimestre": int(cb["trimestre"].iloc[0]), "n": int(dom.sum())}
        for k, (var, si, validos) in PERSONA.items():
            x = num(cb[var])
            val = x.isin(validos + ((9,) if denom_oficial else ()))
            p, se, nv = prop_y_se(cb, (x == si).astype(float), val, cb["w_sel"], dom)
            fila[f"pct_{k}"], fila[f"se_{k}"] = p, se
        if vict:
            r = pd.concat([num(cb[v]) for v in ROBOS_HOGAR], axis=1)
            si_ = (r == 1).any(axis=1)
            val = si_ | (r == 2).all(axis=1) if not denom_oficial else pd.Series(True, index=cb.index)
            p, se, nv = prop_y_se(cb, si_.astype(float), val, cb["w_viv"], dom)
            fila["pct_hog_vict_robo"], fila["se_hog_vict_robo"], fila["n_hog_vict"] = p, se, nv
            x = num(cb["BP1_6_4"])
            val = x.isin((1, 2, 9) if denom_oficial else (1, 2))
            p, se, _ = prop_y_se(cb, (x == 1).astype(float), val, cb["w_viv"], dom)
            fila["pct_hog_vict_robo_calle"], fila["se_hog_vict_robo_calle"] = p, se
        else:
            for c in ["pct_hog_vict_robo", "se_hog_vict_robo", "pct_hog_vict_robo_calle",
                      "se_hog_vict_robo_calle", "n_hog_vict"]:
                fila[c] = np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def construir() -> pd.DataFrame:
    partes, excluidos = [], []
    for anio, trim, f in archivos():
        cb = cargar_trimestre(anio, trim, f)
        if cb is None:
            excluidos.append((anio, trim))
            continue
        partes.append(indicadores(cb))
    df = pd.concat(partes, ignore_index=True).sort_values(["anio", "trimestre", "cve_mun"])
    df["n_hog_vict"] = df["n_hog_vict"].astype("Int64")
    cols = ["cve_mun", "alcaldia", "anio", "trimestre", "n"]
    pct = [c for c in df.columns if c.startswith("pct_")]
    se = ["se_" + c[4:] for c in pct]
    df = df[cols + pct + se + ["n_hog_vict"]].reset_index(drop=True)
    print("Trimestres excluidos (CDMX por regiones):", excluidos)
    return df


def validar():
    """Reproduce cifras de los tabulados oficiales de dic-2024 (denominador con NS/NR, como INEGI)."""
    sys.path.insert(0, os.path.dirname(__file__))
    f = [x for x in archivos() if x[0] == 2024 and x[1] == 4][0]
    cb = cargar_trimestre(*f)
    of = indicadores(cb, denom_oficial=True)
    return of


if __name__ == "__main__":
    df = construir()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(df.shape, "->", OUT)
    print(df.groupby(["anio", "trimestre"]).size().to_string())
    if "--validar" in sys.argv:
        print(validar().to_string())
