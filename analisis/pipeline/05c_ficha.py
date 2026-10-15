"""05c · Ficha de la AGEB (contrato v0.9) → app/public/data/ficha.json

Contexto descriptivo para el detalle de cada AGEB: composición de su actividad económica, quién vive ahí, cómo cambió
su número de negocios y lo que dicen las encuestas de su alcaldía. Nada de esto entra al modelo ni cambia la lista: es
contexto para leer cada zona. Se exporta aparte de 05_export.py para no regenerar los demás archivos.

Fuentes:
- Composición, tamaño y densidad: DENUE 2023-11 (`E` fijo, el mismo corte que el modelo), del panel.
- Censo 2020 (RESAGEBURB, filas de total por AGEB): los mismos indicadores y la misma regla de faltantes que el
  bloque `vulnerabilidad` de validacion.json (v0.7), salvo el denominador de internet: VPH_INTER se cuenta sobre
  TVIVPARHAB (con VIVPAR_HAB, 234 AGEB tienen más viviendas con internet que viviendas).
- Cambio de negocios: DENUE 2019-11 → 2024-11 por `id` (08_rotacion.py, prueba de medición M2a aprobada).
- ENVIPE (04e): % de delitos que los hogares denuncian ante el MP por alcaldía de ocurrencia, ediciones 2021–2025.
- ENSU: percepción de inseguridad y hogares con víctima de robo por alcaldía, promedio de los trimestres de 2024.
  No se usa la denuncia relativa de 04d (se quitó de la app el 2026-10-14: mide población flotante).

Uso: .venv/bin/python analisis/pipeline/05c_ficha.py
"""
import json
import re
import zipfile
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from common import CANDIDATOS_EFECTIVO, CORTE_FIJO, EFECTIVO, INTERIM, PROCESSED, RAW, ROOT, SECTORES, lee_panel

SALIDA = ROOT / "app/public/data/ficha.json"
MODELOS = PROCESSED / "modelos"
ANIO_ENSU = 2024  # último año observado del análisis
GRUPOS = [*EFECTIVO, *SECTORES, "otros"]
GIROS_EF = list(CANDIDATOS_EFECTIVO)


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


# ============================================================ esquema (espejo del contrato v0.9)
Prob = Field(ge=0, le=1)
Grupo = Literal[tuple(GRUPOS)]
GiroEf = Literal[tuple(GIROS_EF)]


class Censo(BaseModel):
    escolaridad: Optional[float] = Field(default=None, ge=0)
    pct_sin_basica: Optional[float] = Field(default=None, ge=0, le=1)
    pct_sin_salud: Optional[float] = Field(default=None, ge=0, le=1)
    pct_sin_internet: Optional[float] = Field(default=None, ge=0, le=1)


class Dinamica(BaseModel):
    antes: int = Field(ge=0)
    despues: int = Field(ge=0)
    salidas: int = Field(ge=0)
    entradas: int = Field(ge=0)


class FichaAgeb(BaseModel):
    composicion: dict[Grupo, float]
    efectivo: dict[GiroEf, float]
    personal_ocupado: int = Field(ge=0)
    pct_grandes: float = Prob
    densidad: float = Field(ge=0)
    pct_densidad: float = Prob
    censo: Censo
    dinamica: Optional[Dinamica] = None


class Indicador(BaseModel):
    valor: float = Prob
    ee: float = Field(ge=0)
    rank: int = Field(ge=1, le=16)


class Envipe(BaseModel):
    denuncia: float = Prob
    ee: float = Field(ge=0)
    contraida: float = Prob
    contraida_ic: tuple[float, float]


class FichaAlcaldia(BaseModel):
    envipe: Envipe
    ensu: dict[Literal["inseg_ciudad", "inseg_calle", "inseg_mercado", "hog_vict_robo"], Indicador]


class Ciudad(BaseModel):
    composicion: dict[Grupo, float]
    efectivo: dict[GiroEf, float]
    pct_grandes: float = Prob
    densidad_mediana: float = Field(ge=0)
    censo: Censo
    dinamica: Dinamica
    cambio_mediano: float
    envipe_denuncia: float = Prob
    envipe_p_iguales: float = Prob
    anio_ensu: int


class Ficha(BaseModel):
    ciudad: Ciudad
    agebs: dict[str, FichaAgeb]
    alcaldias: dict[str, FichaAlcaldia]


# ============================================================ actividad económica (DENUE 2023-11, `E` fijo)
p = lee_panel()
p = p[p.anio == p.anio.max()].set_index("cvegeo")
p = p[p.E_fijo > 0]  # mismo universo que ageb_metrics.json
assert len(p) == 2419, len(p)
E = p.E_fijo
densidad = E / p.area_km2
comp = p[[f"ps_{g}_fijo" for g in GRUPOS]].set_axis(GRUPOS, axis=1)
efec = p[[f"pc_{g}_fijo" for g in GIROS_EF]].set_axis(GIROS_EF, axis=1)
assert np.allclose(comp.sum(axis=1), 1, atol=1e-6)

# ============================================================ Censo 2020 (mismos indicadores que v0.7)
CENSO_V = ["POBTOT", "P_15YMAS", "GRAPROES", "P15YM_SE", "P15PRI_IN", "P15PRI_CO", "P15SEC_IN", "PSINDER", "TVIVPARHAB", "VPH_INTER"]
zc = zipfile.ZipFile(RAW / "censo2020/RESAGEBURB_09_2020_csv.zip")
nc = [n for n in zc.namelist() if n.lower().endswith(".csv") and "resageburb" in n.lower()][0]
cz = pd.read_csv(zc.open(nc), dtype=str, usecols=["ENTIDAD", "MUN", "LOC", "AGEB", "MZA", *CENSO_V], encoding="utf-8-sig")
cz = cz[(cz.MZA == "000") & (cz.AGEB != "0000")]
cz.index = cz.ENTIDAD.str.zfill(2) + cz.MUN.str.zfill(3) + cz.LOC.str.zfill(4) + cz.AGEB
cz = cz[CENSO_V].apply(pd.to_numeric, errors="coerce").reindex(p.index)
assert cz.POBTOT.notna().all(), "AGEB sin censo"
cz["SIN_BASICA"] = cz[["P15YM_SE", "P15PRI_IN", "P15PRI_CO", "P15SEC_IN"]].sum(axis=1, min_count=4)


def cociente(x, num, den):
    ok = x[num].notna() & x[den].notna() & (x[den] > 0)
    return x.loc[ok, num].sum() / x.loc[ok, den].sum() if ok.any() else None


def censo(x):
    ok = x.GRAPROES.notna() & x.P_15YMAS.notna() & (x.P_15YMAS > 0)
    sin_int = cociente(x, "VPH_INTER", "TVIVPARHAB")
    return {"escolaridad": round(float(np.average(x.GRAPROES[ok], weights=x.P_15YMAS[ok])), 2) if ok.any() else None,
            "pct_sin_basica": r3(cociente(x, "SIN_BASICA", "P_15YMAS")), "pct_sin_salud": r3(cociente(x, "PSINDER", "POBTOT")),
            "pct_sin_internet": r3(1 - sin_int) if sin_int is not None else None}


# ============================================================ cambio de negocios (DENUE 2019-11 → 2024-11, 08_rotacion)
rot = pd.read_parquet(MODELOS / "rotacion_ageb.parquet").set_index("cvegeo")
dinamica = {cv: {"antes": int(r.N_2019), "despues": int(r.N_2024), "salidas": int(r.S), "entradas": int(r.En)}
            for cv, r in rot.iterrows() if cv in p.index}
con_n = rot[rot.N_2019 > 0]
cambio_mediano = float((con_n.N_2024 / con_n.N_2019 - 1).median())

# ============================================================ ENVIPE por alcaldía (04e)
mun = pd.read_parquet(INTERIM / "ensu_alcaldia.parquet")[["cve_mun", "alcaldia"]].drop_duplicates().set_index("alcaldia").cve_mun
cn = pd.read_csv(ROOT / "analisis/pipeline/outputs/modelos/cifra_negra_alcaldia.csv")
envipe = {}
for _, r in cn[cn.metrica.str.startswith("denuncia · ")].iterrows():
    nombre = r.metrica.removeprefix("denuncia · ")
    m = re.match(r"EE ([\d.]+);.*contraída ([\d.]+) \[([\d.]+), ([\d.]+)\]", r.nota)
    envipe[mun[nombre]] = {"denuncia": r3(r.valor), "ee": r3(float(m[1])), "contraida": r3(float(m[2])),
                           "contraida_ic": (r3(float(m[3])), r3(float(m[4])))}
assert len(envipe) == 16, sorted(envipe)
d_cdmx = float(cn.loc[cn.metrica == "denuncia CDMX · principal (ediciones 2021–2025)", "valor"].iloc[0])
p_iguales = float(re.search(r"p = ([\d.]+)", (ROOT / "analisis/pipeline/outputs/modelos/cifra_negra_alcaldia.md").read_text())[1])

# ============================================================ ENSU por alcaldía (promedio de los trimestres del año)
en = pd.read_parquet(INTERIM / "ensu_alcaldia.parquet")
en = en[en.anio == ANIO_ENSU]
ensu = {}
for clave, col in {"inseg_ciudad": "inseg_ciudad", "inseg_calle": "inseg_calle", "inseg_mercado": "inseg_mercado",
                   "hog_vict_robo": "hog_vict_robo"}.items():
    x = en.dropna(subset=[f"pct_{col}"]).groupby("cve_mun").agg(
        v=(f"pct_{col}", "mean"), ee=(f"se_{col}", lambda s: float(np.sqrt((s ** 2).sum()) / len(s))))
    assert len(x) == 16, (clave, len(x))
    x["rank"] = x.v.rank(ascending=False, method="first").astype(int)  # 1 = la más alta
    for cm, r in x.iterrows():
        ensu.setdefault(cm, {})[clave] = {"valor": r3(r.v / 100), "ee": r3(r.ee / 100), "rank": int(r["rank"])}

# ============================================================ armar y escribir
ficha = {
    "ciudad": {
        "composicion": {g: r3((comp[g] * E).sum() / E.sum()) for g in GRUPOS},
        "efectivo": {g: r3((efec[g] * E).sum() / E.sum()) for g in GIROS_EF},
        "pct_grandes": r3(p.n_grandes_fijo.sum() / E.sum()),
        "densidad_mediana": round(float(densidad.median()), 1),
        "censo": censo(cz),
        "dinamica": {k: int(sum(d[k] for d in dinamica.values())) for k in ["antes", "despues", "salidas", "entradas"]},
        "cambio_mediano": r3(cambio_mediano),
        "envipe_denuncia": r3(d_cdmx),
        "envipe_p_iguales": r3(p_iguales),
        "anio_ensu": ANIO_ENSU,
    },
    "agebs": {
        cv: {
            "composicion": {g: r3(comp.at[cv, g]) for g in GRUPOS},
            "efectivo": {g: r3(efec.at[cv, g]) for g in GIROS_EF if efec.at[cv, g] > 0},
            "personal_ocupado": int(round(p.at[cv, "po_est_fijo"])),
            "pct_grandes": r3(p.at[cv, "p_grandes_fijo"]),
            "densidad": round(float(densidad[cv]), 1),
            "pct_densidad": r3(densidad.rank(pct=True)[cv]),
            "censo": censo(cz.loc[[cv]]),
            **({"dinamica": dinamica[cv]} if cv in dinamica else {}),
        }
        for cv in p.index
    },
    "alcaldias": {cm: {"envipe": envipe[cm], "ensu": ensu[cm]} for cm in sorted(envipe)},
}
Ficha.model_validate(ficha)
SALIDA.write_text(json.dumps(ficha, ensure_ascii=False, separators=(",", ":")))

c = ficha["ciudad"]
print(f"ficha.json: {SALIDA.stat().st_size / 1024:.0f} KB · {len(ficha['agebs'])} AGEB ({len(dinamica)} con cambio de negocios)")
print(f"ciudad: densidad mediana {c['densidad_mediana']} neg/km² · escolaridad {c['censo']['escolaridad']} · "
      f"negocios {c['dinamica']['antes']:,} → {c['dinamica']['despues']:,} (mediana por AGEB {c['cambio_mediano']:+.1%}) · "
      f"ENVIPE {c['envipe_denuncia']:.1%} (p iguales {c['envipe_p_iguales']})")
print("composición ciudad:", {g: f"{v:.1%}" for g, v in c["composicion"].items()})
