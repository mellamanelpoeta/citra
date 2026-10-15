"""00a · Reproduce la consulta a la API de la FGJ con la que se armó la copia congelada de delitos.

Qué baja: las carpetas de investigación de la Fiscalía General de Justicia de la CDMX de los delitos de interés
(robo a negocio, a transeúnte, a repartidor, a cuentahabiente, extorsión y afines), con año del hecho 2016–2025, desde
el almacén de datos (CKAN `datastore_search_sql`) del portal https://datos.cdmx.gob.mx:

- Conjunto: "Carpetas de investigación FGJ" (`carpetas-de-investigacion-fgj-de-la-ciudad-de-mexico`).
- Recurso: `35e87ce4-8e40-4de4-8dac-0b59645c13eb`, "Carpetas de Investigación (acumulado 2016-2024)", que carga
  `carpetasFGJ_acumulado_2025_01.csv`: todas las carpetas iniciadas hasta el 31 de enero de 2025.
- Filtro: el mismo de `fgj_delitos_interes.json` (`delito` o `categoria_delito` contienen NEGOCIO, TRANSE, REPARTIDOR,
  CUENTAHABIENTE, EXTORSI o COBRO DE PISO), sin daño en propiedad ajena ni secuestro, y año del hecho 2016–2025.
  Con ese filtro, los conteos por delito y año coinciden exactamente con la copia congelada.
- Columnas: `_id`, `delito`, `categoria_delito`, `anio_hecho`, `fecha_hecho`, `hora_hecho`, `fecha_inicio`,
  `alcaldia_hecho`, `colonia_catalogo`, `alcaldia_catalogo`, `latitud`, `longitud` (todas texto salvo `_id`), con
  las filas ordenadas por `anio_hecho` y `_id`, como la copia congelada.

Por qué el análisis usa una copia congelada
    La FGJ reclasifica carpetas de forma continua (diaria, semanal y mensual, según su nota del 29-jul-2020) y el portal
    vuelve a cargar los recursos: el delito, la fecha o las coordenadas de una carpeta pueden cambiar entre versiones, y
    el portal se re-migró en agosto de 2026. Para que los resultados sean reproducibles, el análisis usa la copia
    `data/raw/_samples/fgj/fgj_delitos_objetivo_2016_2025.parquet` (332,027 filas, 12 columnas; sha256 en
    `analisis/MANIFIESTO_DATOS.csv`), descargada el 2026-10-03 (fecha de los archivos; las respuestas de la API que
    documentan la consulta, `fgj_delitos_interes.json` y `fgj_geo_por_anio_hecho.json`, son del mismo día). Su corte es
    `fecha_inicio` ≤ 2025-01-31: 2025 trae solo enero y el análisis lo excluye.

    Este script NO sobrescribe esa copia. Escribe un archivo nuevo con la fecha de descarga e imprime la comparación
    contra la congelada (filas por `anio_hecho` y por `delito`, `_id` comunes y valores que cambiaron). Una descarga
    nueva puede diferir ligeramente si la FGJ actualizó el recurso. Prueba del 2026-10-15: la descarga completa fue
    idéntica byte a byte a la copia congelada (mismo sha256), porque el recurso no ha cambiado desde su carga.

Uso (desde la raíz del repo):
    .venv/bin/python analisis/pipeline/00a_fgj_api.py                  # todo (≈ 11 páginas de 32,000 filas)
    .venv/bin/python analisis/pipeline/00a_fgj_api.py --anio 2024      # solo un año del hecho, para una prueba rápida
    .venv/bin/python analisis/pipeline/00a_fgj_api.py --salida /tmp/fgj.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FGJ = ROOT / "data/raw/_samples/fgj"
CONGELADO = FGJ / "fgj_delitos_objetivo_2016_2025.parquet"

API = "https://datos.cdmx.gob.mx/api/3/action/datastore_search_sql"
RECURSO = "35e87ce4-8e40-4de4-8dac-0b59645c13eb"
COLUMNAS = ["_id", "delito", "categoria_delito", "anio_hecho", "fecha_hecho", "hora_hecho", "fecha_inicio",
            "alcaldia_hecho", "colonia_catalogo", "alcaldia_catalogo", "latitud", "longitud"]
ANIOS = [f"{y}.0" for y in range(2016, 2026)]          # la API guarda el año como texto "2016.0"
FILTRO = ("(delito ~* '(NEGOCIO|TRANSE|REPARTIDOR|CUENTAHABIENTE|EXTORSI|COBRO DE PISO)' "
          "OR categoria_delito ~* '(NEGOCIO|TRANSE|REPARTIDOR|CUENTAHABIENTE)') "
          "AND delito !~* '^(DAÑO EN PROPIEDAD|SECUESTRO)'")
PAGINA = 32000                                          # tope de filas por respuesta del portal


def sql(anios: list[str], desde_id: int) -> str:
    cols = ", ".join(f'"{c}"' for c in COLUMNAS)
    lista = ", ".join(f"'{a}'" for a in anios)
    return (f'SELECT {cols} FROM "{RECURSO}" WHERE {FILTRO} AND anio_hecho IN ({lista}) AND _id > {desde_id} '
            f"ORDER BY _id LIMIT {PAGINA}")


def descarga(anios: list[str], limite: int | None) -> pd.DataFrame:
    import requests
    s = requests.Session()
    filas, ultimo = [], 0
    while True:
        for intento in range(4):
            try:
                r = s.get(API, params={"sql": sql(anios, ultimo)}, timeout=300)
                r.raise_for_status()
                res = r.json()
                assert res["success"], res
                break
            except Exception as e:  # noqa: BLE001
                if intento == 3:
                    raise
                print(f"    reintento tras {type(e).__name__}")
                time.sleep(5 * (intento + 1))
        rec = res["result"]["records"]
        filas += rec
        print(f"    página {len(filas) // PAGINA + (len(rec) < PAGINA)}: +{len(rec):,} (total {len(filas):,})")
        if len(rec) < PAGINA or (limite and len(filas) >= limite):
            break
        ultimo = rec[-1]["_id"]
    d = pd.DataFrame(filas, columns=COLUMNAS)
    if limite:
        d = d.head(limite)
    d["_id"] = d["_id"].astype("int64")
    for c in COLUMNAS[1:]:
        d[c] = d[c].astype("str").where(d[c].notna())
    # La copia congelada se bajó año por año: queda ordenada por año del hecho y, dentro, por _id
    return d.sort_values(["anio_hecho", "_id"], kind="stable").reset_index(drop=True)


def compara(nuevo: pd.DataFrame, anios: list[str]) -> None:
    if not CONGELADO.exists():
        print(f"No está la copia congelada ({CONGELADO}); no hay con qué comparar")
        return
    ref = pd.read_parquet(CONGELADO)
    ref = ref[ref.anio_hecho.isin(anios)]
    print(f"\n== Comparación con la copia congelada ({CONGELADO.name}, mismos años)")
    print(f"Filas: congelada {len(ref):,} · nueva {len(nuevo):,} · diferencia {len(nuevo) - len(ref):+,}")
    for col in ("anio_hecho", "delito"):
        t = pd.concat([ref[col].value_counts().rename("congelada"), nuevo[col].value_counts().rename("nueva")],
                      axis=1).fillna(0).astype(int)
        t["dif"] = t.nueva - t.congelada
        t = t.sort_index() if col == "anio_hecho" else t.sort_values("congelada", ascending=False)
        print(f"\nPor {col}:" + (" (solo filas con diferencia)" if col == "delito" and (t.dif != 0).any() else ""))
        print((t[t.dif != 0] if col == "delito" and (t.dif != 0).any() else t).to_string())
    a, b = set(ref._id), set(nuevo._id)
    print(f"\n_id: comunes {len(a & b):,} · solo en la congelada {len(a - b):,} · solo en la nueva {len(b - a):,}")
    m = ref.merge(nuevo, on="_id", suffixes=("_c", "_n"))
    cambios = {c: int((m[f"{c}_c"].fillna("∅") != m[f"{c}_n"].fillna("∅")).sum()) for c in COLUMNAS[1:]}
    print("Valores distintos en _id comunes:", {k: v for k, v in cambios.items() if v} or "ninguno")
    iguales = (len(a ^ b) == 0) and not any(cambios.values())
    print("→ La descarga reproduce exactamente la copia congelada" if iguales else
          "→ Hay diferencias: el análisis sigue usando la copia congelada")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--anio", type=int, nargs="+", help="años del hecho (por omisión 2016–2025)")
    ap.add_argument("--limite", type=int, help="máximo de filas (prueba rápida)")
    ap.add_argument("--salida", type=Path,
                    help="parquet de salida (por omisión data/raw/_samples/fgj/fgj_delitos_objetivo_descarga_<fecha>.parquet)")
    a = ap.parse_args()

    anios = [f"{y}.0" for y in a.anio] if a.anio else ANIOS
    hoy = dt.date.today().isoformat()
    sufijo = "" if not a.anio else "_" + "_".join(map(str, a.anio))
    salida = a.salida or FGJ / f"fgj_delitos_objetivo_descarga_{hoy}{sufijo}.parquet"
    if salida.resolve() == CONGELADO.resolve():
        sys.exit("La salida no puede ser la copia congelada")
    print(f"Consulta a {API} (recurso {RECURSO}), años {anios[0]}–{anios[-1]}")
    d = descarga(anios, a.limite)
    salida.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(salida, index=False)
    print(f"→ {salida} ({len(d):,} filas, {d.shape[1]} columnas)")
    if CONGELADO.exists() and not a.anio and not a.limite:
        import hashlib
        h = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (salida, CONGELADO)]
        print(f"sha256 nueva {h[0]}\nsha256 congelada {h[1]}" + ("  (idénticas)" if h[0] == h[1] else ""))
    if a.limite:
        print("Con --limite la comparación solo es indicativa")
    compara(d, anios)


if __name__ == "__main__":
    main()
