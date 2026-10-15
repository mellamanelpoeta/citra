"""00b · Convierte los zip del DENUE (INEGI) a los parquet que lee el pipeline.

Entrada:  data/raw/_samples/denue/denue_09_<AAAA>_<MM>_csv.zip   (el corte 2026-05 es denue_09_csv_actual.zip)
Salida:   data/raw/_samples/denue/parquet/denue_09_<AAAA>_<MM>.parquet

Cada parquet conserva solo 11 columnas del CSV de INEGI, todas como texto y con sus nombres originales:
`id`, `clee` (solo existe desde 2021-05), `codigo_act`, `per_ocu`, `cve_mun`, `ageb`, `manzana`, `tipoUniEco`,
`latitud`, `longitud`, `fecha_alta`; más `version` = "<AAAA>_<MM>" (el corte). Las filas quedan en el orden del CSV.

Supuestos y detalles de lectura (verificados contra los parquet con los que se hizo el análisis):
- Codificación: INEGI publica unos cortes en UTF-8 (con o sin BOM) y otros en Latin-1. Se intenta UTF-8 y, si falla,
  Latin-1. Solo se usan campos de claves y coordenadas, que no cambian con la codificación.
- Todo se lee como texto (`dtype=str`): conserva ceros a la izquierda (`cve_mun`, `ageb`, `manzana`) y las coordenadas
  tal como vienen. Los campos vacíos quedan como nulos.
- Hay saltos de línea dentro de campos entre comillas (en nombres y domicilios); el lector de pandas los respeta, por eso
  el número de filas es menor que el de líneas del archivo.
- Con pandas 3.0.6 y pyarrow 25.0.1 (requirements.txt) la salida es idéntica byte a byte a los parquet originales.

Uso (desde la raíz del repo):
    .venv/bin/python analisis/pipeline/00b_denue_parquet.py                # cortes del pipeline
    .venv/bin/python analisis/pipeline/00b_denue_parquet.py --todos        # los 19 cortes (serie de factibilidad)
    .venv/bin/python analisis/pipeline/00b_denue_parquet.py --cortes 2023_11 --destino /tmp/pq --comparar
Es idempotente: no reescribe un parquet que ya existe, salvo con --forzar.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEN = ROOT / "data/raw/_samples/denue"

COLS = ["id", "clee", "codigo_act", "per_ocu", "cve_mun", "ageb", "manzana", "tipoUniEco", "latitud", "longitud",
        "fecha_alta"]
# Cortes que lee el pipeline: 2016-01 (04c), 2018-11 (06) y noviembre 2019–2024 (01, 05, 05b, 06, 08)
PIPELINE = ["2016_01", "2018_11", "2019_11", "2020_11", "2021_11", "2022_11", "2023_11", "2024_11"]
ACTUAL = "2026_05"   # el corte vigente se guarda como denue_09_csv_actual.zip


def zip_de(corte: str, origen: Path) -> Path:
    z = origen / f"denue_09_{corte}_csv.zip"
    if not z.exists() and corte == ACTUAL:
        z = origen / "denue_09_csv_actual.zip"
    return z


def lee_csv(zp: Path) -> tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(zp) as zf:
        nombre = [n for n in zf.namelist() if n.lower().endswith(".csv") and "conjunto" in n.lower()]
        assert len(nombre) == 1, (zp, nombre)
        b = zf.read(nombre[0])
    try:
        texto, enc = b.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        texto, enc = b.decode("latin-1"), "latin-1"
    del b
    cab = pd.read_csv(io.StringIO(texto), nrows=0).columns
    cols = [c for c in COLS if c in cab]
    faltan = set(COLS) - set(cab) - {"clee"}
    assert not faltan, f"{zp.name}: faltan columnas {faltan}"
    return pd.read_csv(io.StringIO(texto), dtype=str, usecols=cols)[cols], enc


def convierte(corte: str, origen: Path) -> pd.DataFrame:
    d, enc = lee_csv(zip_de(corte, origen))
    d["version"] = corte
    print(f"  {corte}: {len(d):,} filas, {d.shape[1]} columnas (con version), {enc}")
    return d


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compara(nuevo: Path, ref: Path) -> bool:
    """Compara contra el parquet de referencia: primero bytes; si difieren, contenido con assert_frame_equal."""
    if not ref.exists():
        print(f"    sin referencia en {ref}")
        return False
    if sha256(nuevo) == sha256(ref):
        print("    idéntico byte a byte a la referencia")
        return True
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(nuevo), pd.read_parquet(ref))
        print("    mismo contenido (assert_frame_equal) aunque los bytes difieren (versión de pyarrow o compresión)")
        return True
    except AssertionError as e:
        print(f"    DIFIERE de la referencia: {str(e)[:400]}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cortes", nargs="+", help=f"cortes AAAA_MM (por omisión los del pipeline: {' '.join(PIPELINE)})")
    ap.add_argument("--todos", action="store_true", help="todos los zip que haya en el origen")
    ap.add_argument("--origen", type=Path, default=DEN, help="carpeta de los zip")
    ap.add_argument("--destino", type=Path, default=DEN / "parquet", help="carpeta de salida")
    ap.add_argument("--forzar", action="store_true", help="reescribe aunque el parquet exista")
    ap.add_argument("--comparar", action="store_true",
                    help="compara cada salida con el parquet del mismo nombre en data/raw/_samples/denue/parquet/")
    a = ap.parse_args()

    if a.todos:
        cortes = sorted(p.name[len("denue_09_"):-len("_csv.zip")] for p in a.origen.glob("denue_09_20*_csv.zip"))
        if (a.origen / "denue_09_csv_actual.zip").exists():
            cortes.append(ACTUAL)
    else:
        cortes = a.cortes or PIPELINE
    a.destino.mkdir(parents=True, exist_ok=True)
    malos = 0
    for c in cortes:
        out = a.destino / f"denue_09_{c}.parquet"
        if out.exists() and not a.forzar:
            print(f"  {c}: ya existe {out}")
        else:
            if not zip_de(c, a.origen).exists():
                print(f"  {c}: falta {zip_de(c, a.origen)} (corre 00_descargar.py)")
                malos += 1
                continue
            convierte(c, a.origen).to_parquet(out, index=False)
        if a.comparar:
            malos += not compara(out, DEN / "parquet" / out.name)
    sys.exit(1 if malos else 0)


if __name__ == "__main__":
    main()
