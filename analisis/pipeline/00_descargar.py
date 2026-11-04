"""00 · Descarga de los datos crudos desde las fuentes oficiales a data/raw/_samples/.

Reconstruye la carpeta `data/raw/_samples/` con la misma estructura que leen los scripts del análisis: baja cada
archivo de INEGI o del portal de datos abiertos de la CDMX y lo descomprime donde el código lo espera.

Qué NO hace:
- No descarga la base de la FGJ (carpetas de investigación). El análisis usa una copia congelada
  (`fgj/fgj_delitos_objetivo_2016_2025.parquet`) que se obtiene del paquete de datos publicado; para consultar la API
  de nuevo está `00a_fgj_api.py` (ver su docstring: la FGJ reclasifica carpetas y una descarga nueva puede diferir).
- No genera los parquet del DENUE (`denue/parquet/`): eso lo hace `00b_denue_parquet.py` a partir de los zip.

Uso (desde la raíz del repo):
    .venv/bin/python analisis/pipeline/00_descargar.py                 # núcleo: lo que necesita el pipeline
    .venv/bin/python analisis/pipeline/00_descargar.py --todo          # + complementos (scripts de factibilidad)
    .venv/bin/python analisis/pipeline/00_descargar.py --solo ensu,enve
    .venv/bin/python analisis/pipeline/00_descargar.py --verificar     # compara sha256 con analisis/MANIFIESTO_DATOS.csv
    .venv/bin/python analisis/pipeline/00_descargar.py --comprobar-urls  # HEAD a cada URL y tamaño contra la copia local
    .venv/bin/python analisis/pipeline/00_descargar.py --listar
    .venv/bin/python analisis/pipeline/00_descargar.py --destino /tmp/raw --solo colonias   # prueba en otra carpeta

Es idempotente: si los archivos que usa el código ya existen, no descarga nada (salvo con --forzar). Si el zip
ya está pero falta lo descomprimido, solo descomprime.

Notas de las fuentes:
- `archivo.datos.cdmx.gob.mx` tiene el certificado TLS vencido; esas descargas se hacen sin verificar el certificado
  (solo los CSV anuales de la FGJ, que son complemento).
- Algunas URL de la API del portal CDMX apuntan a una IP interna (10.80.148.134); aquí ya están sustituidas por
  https://datos.cdmx.gob.mx.
- INEGI responde 200 con una página HTML cuando un archivo no existe; el script lo detecta por el tipo de contenido.
- El DENUE "actual" (corte 2026-05, solo complemento) se publica en una URL que cambia con cada corte; si INEGI ya
  publicó otro, el sha256 no coincidirá y --verificar lo dirá.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/_samples"
MANIFIESTO = ROOT / "analisis/MANIFIESTO_DATOS.csv"

INEGI = "https://www.inegi.org.mx/contenidos"
CDMX = "https://datos.cdmx.gob.mx/dataset"
UA = {"User-Agent": "Mozilla/5.0 (reproduccion-analisis-titulacion)"}

LIC_INEGI = "Términos de Libre Uso de la Información del INEGI"
LIC_CDMX = "CC BY 4.0 (declarada en datos.cdmx.gob.mx)"

# Cortes del DENUE que usa el pipeline (01, 04c, 05, 05b, 06, 08) y los demás (serie de factibilidad, fig05a del EDA)
DENUE_PIPELINE = ["2016_01", "2018_11", "2019_11", "2020_11", "2021_11", "2022_11", "2023_11", "2024_11"]
DENUE_OTROS = ["2016_10", "2017_03", "2017_11", "2018_03", "2019_04", "2020_04", "2021_05", "2022_05", "2024_05",
               "2025_05", "2026_05"]

MESES = {"marzo": "03", "junio": "06", "septiembre": "09", "diciembre": "12"}


# ============================================================ extractores
def extraer_todo(sub: str) -> Callable[[Path, Path], None]:
    """Descomprime el zip completo en RAW/sub."""
    def f(zp: Path, raw: Path) -> None:
        with zipfile.ZipFile(zp) as z:
            z.extractall(raw / sub)
    return f


def extraer_patrones(sub: str, patrones: list[str], aplanar: bool = False) -> Callable[[Path, Path], None]:
    """Descomprime en RAW/sub solo los miembros que coinciden con algún patrón (fnmatch sobre la ruta del miembro).
    Con aplanar=True se descarta la carpeta interna y se conserva solo el nombre del archivo."""
    def f(zp: Path, raw: Path) -> None:
        with zipfile.ZipFile(zp) as z:
            for m in z.infolist():
                if m.is_dir() or not any(fnmatch.fnmatch(m.filename, p) for p in patrones):
                    continue
                if aplanar:
                    out = raw / sub / Path(m.filename).name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    z.extract(m, raw / sub)
    return f


def extraer_ensu(anio: int, fmt: str, meses: list[str] | None = None) -> Callable[[Path, Path], None]:
    """El zip anual de microdatos de la ENSU trae un zip por trimestre (`ensu_bd_<mes>_<año>_<fmt>.zip`, a veces con
    `.zip.zip` o dentro de una carpeta). Cada uno se descomprime, sin carpetas internas, en `encuestas/ensu<año>/<mes>_<fmt>/`,
    que es lo que busca `_factibilidad/ensu_alcaldia.py`. Con `meses` se limita a esos trimestres (en 2021 el zip CSV
    no trae marzo y se toma del DBF; los demás trimestres de 2021 se toman del CSV para no duplicarlos)."""
    def f(zp: Path, raw: Path) -> None:
        import io
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                m = re.search(r"ensu_bd_(\w+?)_(\d{4})_(csv|dbf)\.zip", Path(n).name)
                if not m or int(m.group(2)) != anio or m.group(3) != fmt:
                    continue
                if meses is not None and m.group(1) not in meses:
                    continue
                dest = raw / f"encuestas/ensu{anio}/{m.group(1)}_{fmt}"
                dest.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(z.read(n))) as zi:
                    for mi in zi.infolist():
                        if mi.is_dir() or not Path(mi.filename).name.upper().startswith("ENSU_"):
                            continue
                        with zi.open(mi) as src, open(dest / Path(mi.filename).name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
    return f


# ============================================================ catálogo de fuentes
@dataclass
class Fuente:
    clave: str
    grupo: str
    nivel: str                      # "nucleo" (pipeline) o "complementario" (scripts de factibilidad y validación)
    fuente: str
    licencia: str
    urls: list[str]                 # se prueban en orden
    archivo: str | None             # dónde se guarda la descarga (relativo a RAW); None = no se descarga aquí
    archivos: list[str]             # lo que lee el código (relativo a RAW); son las filas del manifiesto
    uso: str
    extraer: Callable[[Path, Path], None] | None = None
    tls: bool = True
    nota: str = ""
    extra: dict = field(default_factory=dict)


def _shp(base: str, exts=("shp", "shx", "dbf", "prj", "cpg")) -> list[str]:
    return [f"{base}.{e}" for e in exts]


def _ensu_archivos(anio: int, fmt: str, meses: list[str]) -> list[str]:
    ext = "dbf" if fmt == "dbf" else "csv"
    out = []
    for mes in meses:
        mmyy = f"{MESES[mes]}{str(anio)[2:]}"
        cb = f"ENSU_CB_sec1_2_3_{mmyy}" if (anio, mes) == (2020, "septiembre") else f"ENSU_CB_{mmyy}"
        out += [f"encuestas/ensu{anio}/{mes}_{fmt}/{cb}.{ext}", f"encuestas/ensu{anio}/{mes}_{fmt}/ENSU_VIV_{mmyy}.{ext}"]
    return out


def catalogo() -> list[Fuente]:
    F: list[Fuente] = []

    # ---------------------------------------------------------------- FGJ (no se descarga aquí)
    F.append(Fuente(
        "fgj_congelado", "fgj", "nucleo", "FGJ CDMX, carpetas de investigación (API CKAN, acumulado 2016 → ene-2025)",
        LIC_CDMX, ["https://datos.cdmx.gob.mx/dataset/carpetas-de-investigacion-fgj-de-la-ciudad-de-mexico"], None,
        ["fgj/fgj_delitos_objetivo_2016_2025.parquet"],
        "Variable objetivo (01, 04c, 06). Copia congelada descargada el 2026-10-03; ver 00a_fgj_api.py",
        nota="Se obtiene del paquete de datos; 00a_fgj_api.py reproduce la consulta"))
    F.append(Fuente(
        "fgj_consultas", "fgj", "complementario", "FGJ CDMX: respuestas de la API con las que se definió la consulta",
        LIC_CDMX, ["https://datos.cdmx.gob.mx/api/3/action/datastore_search_sql"], None,
        ["fgj/fgj_delitos_interes.json", "fgj/fgj_geo_por_anio_hecho.json"],
        "Documentan la consulta de la copia congelada (lista de delitos, conteos por año)"))
    for y in range(2019, 2025):
        F.append(Fuente(
            f"fgj_csv_{y}", "fgj_csv", "complementario", f"FGJ CDMX, carpetas de investigación, CSV anual {y}",
            LIC_CDMX, [f"https://archivo.datos.cdmx.gob.mx/FGJ/carpetas/carpetasFGJ_{y}.csv"],
            f"fgj/carpetasFGJ_{y}.csv", [f"fgj/carpetasFGJ_{y}.csv"],
            "Validación cruzada de la API (_factibilidad/01_fgj_load.py). El de 2024 llega solo a julio",
            tls=False))

    # ---------------------------------------------------------------- DENUE
    for c in DENUE_PIPELINE + DENUE_OTROS:
        y, m = c.split("_")
        if c == "2026_05":   # el corte vigente se publica en la URL "actual"; cuando haya otro, pasa a la carpeta del corte
            urls = [f"{INEGI}/masiva/denue/{c}/denue_09_{m}{y[2:]}_csv.zip", f"{INEGI}/masiva/denue/denue_09_csv.zip"]
            arch = "denue/denue_09_csv_actual.zip"
        else:
            urls = [f"{INEGI}/masiva/denue/{c}/denue_09_{m}{y[2:]}_csv.zip"]
            arch = f"denue/denue_09_{c}_csv.zip"
        nucleo = c in DENUE_PIPELINE
        F.append(Fuente(
            f"denue_{c}", "denue", "nucleo" if nucleo else "complementario", f"INEGI, DENUE, corte {y}-{m} (CDMX, CSV)",
            LIC_INEGI, urls, arch, [arch],
            ("Fuente de denue/parquet (00b)" + ("; 08_rotacion lee el nombre del establecimiento" if c == "2019_11" else ""))
            if nucleo else "Serie del DENUE de factibilidad (_factibilidad/03_denue.py → fig05a del EDA)"))
        F.append(Fuente(
            f"denue_parquet_{c}", "denue_parquet", "nucleo" if nucleo else "complementario",
            f"Derivado del DENUE {y}-{m} con 00b_denue_parquet.py", LIC_INEGI, [], None,
            [f"denue/parquet/denue_09_{c}.parquet"],
            {"2016_01": "E de 2016 (04c)", "2018_11": "DENUE 2018 (06)", "2019_11": "Exposición y rotación (01, 06, 08)",
             "2023_11": "E fijo, corte principal (01, 05, 05b)", "2024_11": "Exposición y rotación (01, 06, 08)"}
            .get(c, "Exposición por año (01)") if nucleo else "Serie de factibilidad (_factibilidad/03_denue.py)",
            nota="Se genera con 00b_denue_parquet.py"))

    # ---------------------------------------------------------------- Marco Geoestadístico 2020
    capas = ["09a", "09ar", "09ent", "09mun", "09m"]
    F.append(Fuente(
        "mg2020", "mg2020", "nucleo", "INEGI, Marco Geoestadístico 2020, Ciudad de México", LIC_INEGI,
        [f"{INEGI}/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469/"
         "09_ciudaddemexico.zip"],
        "mg2020/09_ciudaddemexico_mg2020.zip",
        [a for c in capas for a in _shp(f"mg2020/conjunto_de_datos/{c}")],
        "AGEB urbanas (09a) y rurales (09ar), entidad, alcaldías y manzanas (01, 04c, 05, 06, 08)",
        extraer_patrones("mg2020", [f"conjunto_de_datos/{c}.*" for c in capas])))

    # ---------------------------------------------------------------- Censo 2020
    F.append(Fuente(
        "censo2020_resageburb", "censo2020", "nucleo",
        "INEGI, Censo de Población y Vivienda 2020, resultados por AGEB y manzana urbana (CDMX)", LIC_INEGI,
        [f"{INEGI}/programas/ccpv/2020/microdatos/ageb_manzana/RESAGEBURB_09_2020_csv.zip"],
        "censo2020/RESAGEBURB_09_2020_csv.zip",
        ["censo2020/RESAGEBURB_09_2020_csv.zip", "censo2020/resageburb/RESAGEBURB_09CSV20.csv"],
        "Población y rasgos sociodemográficos por AGEB (01, 05, 05c, 06, T7)",
        extraer_todo("censo2020/resageburb")))

    # ---------------------------------------------------------------- Censos Económicos
    F.append(Fuente(
        "ce2019", "censos_economicos", "nucleo", "INEGI, Censos Económicos 2019, Ciudad de México (datos abiertos)",
        LIC_INEGI, [f"{INEGI}/programas/ce/2019/Datosabiertos/ce2019_cdmx_csv.zip"],
        "censos_economicos/ce2019_cdmx_csv.zip",
        ["censos_economicos/ce2019_cdmx_csv/conjunto_de_datos/ce2019_cdmx.csv"],
        "Prueba de medición M del DENUE (06)", extraer_todo("censos_economicos/ce2019_cdmx_csv")))
    F.append(Fuente(
        "ce2024", "censos_economicos", "nucleo", "INEGI, Censos Económicos 2024, Ciudad de México (datos abiertos)",
        LIC_INEGI, [f"{INEGI}/programas/ce/2024/datosabiertos/conjunto_de_datos_ce_cdmx_2024_csv.zip"],
        "censos_economicos/conjunto_de_datos_ce_cdmx_2024_csv.zip",
        ["censos_economicos/conjunto_de_datos_ce_cdmx_2024_csv/conjunto_de_datos/tr_ce_cdmx_2024.csv"],
        "Prueba de medición M del DENUE (06)", extraer_todo("censos_economicos/conjunto_de_datos_ce_cdmx_2024_csv")))

    # ---------------------------------------------------------------- Colonias y transporte (datos.cdmx)
    F.append(Fuente(
        "colonias", "colonias", "nucleo", "Gobierno de la CDMX, Catálogo de colonias (datos abiertos)", LIC_CDMX,
        [f"{CDMX}/02c6ce99-dbd8-47d8-aee1-ae885a12bb2f/resource/4f831409-b3c9-40c9-bc0e-66b0762b3cf4/download/"
         "4f831409-b3c9-40c9-bc0e-66b0762b3cf4.zip"],
        "colonias/catalogo_colonias_datos_abiertos.zip",
        _shp("colonias/catalogo/poligonos_colonias_cdmx", ("shp", "shx", "dbf", "prj")),
        "Colonia de cada AGEB y búsqueda en la app (01, 05, 05b)", extraer_todo("colonias/catalogo")))
    F.append(Fuente(
        "metro", "movilidad", "nucleo", "STC Metro, líneas y estaciones (datos abiertos CDMX)", LIC_CDMX,
        [f"{CDMX}/1b014317-ddb1-46c7-ac79-7330c652abe3/resource/288b10dd-4f21-4338-b1ed-239487820512/download/"
         "288b10dd-4f21-4338-b1ed-239487820512.zip"],
        "movilidad/metro_lineas_estaciones.zip",
        _shp("movilidad/metro/stcmetro_shp/STC_Metro_estaciones_utm14n"),
        "Distancia a la estación más cercana (01, 06)", extraer_todo("movilidad/metro")))
    F.append(Fuente(
        "metrobus", "movilidad", "nucleo", "Metrobús, líneas y estaciones (datos abiertos CDMX)", LIC_CDMX,
        [f"{CDMX}/63a08f93-e959-430f-8dc4-bf07182401e6/resource/66cd5773-d83f-4a80-8c15-8ab34851e046/download/"
         "66cd5773-d83f-4a80-8c15-8ab34851e046.zip"],
        "movilidad/metrobus_lineas_estaciones.zip",
        _shp("movilidad/metrobus/mb_shp/Metrobus_estaciones"),
        "Distancia a la estación más cercana (01, 06)", extraer_todo("movilidad/metrobus")))

    # ---------------------------------------------------------------- Encuestas INEGI
    for ed in range(2021, 2027):
        a = f"encuestas/conjunto_de_datos_envipe_{ed}_csv.zip"
        F.append(Fuente(
            f"envipe_{ed}", "envipe", "nucleo", f"INEGI, ENVIPE {ed} (microdatos, CSV)", LIC_INEGI,
            [f"{INEGI}/programas/envipe/{ed}/datosabiertos/conjunto_de_datos_envipe_{ed}_csv.zip"], a, [a],
            "Denuncia por alcaldía (04e, 05c)"))
    F.append(Fuente(
        "enve2024", "enve", "nucleo", "INEGI, ENVE 2024 (tabulados en datos abiertos)", LIC_INEGI,
        [f"{INEGI}/programas/enve/2024/datosabiertos/conjunto_de_datos_enve_2024_csv.zip"],
        "encuestas/conjunto_de_datos_enve_2024_csv.zip",
        ["encuestas/enve2024/conjunto_de_datos/enve2024_iii_denuncia_delito/t3_6.csv",
         "encuestas/enve2024/conjunto_de_datos/enve2024_iii_denuncia_delito/t3_8.csv"],
        "Cifra negra en negocios por tamaño (03_eda, 04e)", extraer_todo("encuestas/enve2024")))

    todos = list(MESES)
    ensu = [(2019, "dbf", todos), (2020, "dbf", ["marzo", "septiembre", "diciembre"]),
            (2021, "csv", ["junio", "septiembre", "diciembre"]), (2021, "dbf", ["marzo"]),
            (2022, "csv", todos), (2023, "csv", todos), (2024, "csv", todos), (2025, "csv", todos)]
    for anio, fmt, meses in ensu:
        arch = "encuestas/ensu_bd_2025_csv.zip" if anio == 2025 else f"encuestas/ensu{anio}/ensu_bd_{anio}_{fmt}.zip"
        F.append(Fuente(
            f"ensu_{anio}_{fmt}", "ensu", "nucleo", f"INEGI, ENSU {anio} (microdatos trimestrales, {fmt.upper()})",
            LIC_INEGI, [f"{INEGI}/programas/ensu/microdatos/ensu_bd_{anio}_{fmt}.zip"], arch,
            _ensu_archivos(anio, fmt, meses),
            "Percepción por alcaldía (_factibilidad/ensu_alcaldia.py → 04d, 05c)",
            extraer_ensu(anio, fmt, meses if meses != todos else None)))

    tab = {
        "ensu_tab_dic2024_est": ("2024", "ensu_tabulados_basicos_diciembre_2024_estimacion",
                                 ["1.Bas_percepcion_seguridad_dic_2024_est.xlsx", "3.Bas_atestiguacion_dic_2024_est.xlsx",
                                  "7.Bas_delitos_al_hogar_dic_2024_est.xlsx"]),
        "ensu_tab_dic2024_err": ("2024", "ensu_tabulados_basicos_diciembre_2024_error",
                                 ["1.Bas_percepcion_seguridad_dic_2024_err.xlsx"]),
        "ensu_tab_dic2019_est": ("2019", "ensu_tabulados_basicos_diciembre_2019_estimaciones",
                                 ["I_percepcion_dic_2019_est.xlsx",
                                  "X_delitos_contra_integrantes_del_hogar_dic_2019_est.xlsx"]),
    }
    for clave, (anio, nombre, xs) in tab.items():
        sub = f"encuestas/ensu_docs/tabulados/{nombre}"
        F.append(Fuente(
            clave, "ensu_tabulados", "complementario", f"INEGI, ENSU, tabulados básicos ({nombre.split('basicos_')[1]})",
            LIC_INEGI, [f"{INEGI}/programas/ensu/tabulados/{anio}/{nombre}_xlsx.zip"],
            f"encuestas/ensu_docs/tabulados/{nombre}_xlsx.zip", [f"{sub}/{x}" for x in xs],
            "Validación de la ENSU contra INEGI (_factibilidad/validar_ensu.py)",
            extraer_patrones(sub, ["*.xlsx"], aplanar=True)))
    return F


# ============================================================ utilidades
def sha256(p: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(bloque):
            h.update(b)
    return h.hexdigest()


def _sesion():
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = requests.Session()
    s.headers.update(UA)
    return s


def _es_html(r) -> bool:
    return "html" in r.headers.get("Content-Type", "").lower()


def descargar(s, f: Fuente, destino: Path) -> Path:
    out = destino / f.archivo
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    for url in f.urls:
        with s.get(url, stream=True, timeout=120, verify=f.tls, allow_redirects=True) as r:
            if r.status_code != 200 or _es_html(r):
                print(f"    {url} → {r.status_code} {r.headers.get('Content-Type', '')}; se prueba la siguiente")
                continue
            total = int(r.headers.get("Content-Length", 0))
            n = 0
            with open(tmp, "wb") as fh:
                for b in r.iter_content(1 << 20):
                    fh.write(b)
                    n += len(b)
            if total and n != total:
                raise IOError(f"descarga incompleta de {url}: {n:,} de {total:,} bytes")
            tmp.replace(out)
            print(f"    ↓ {url} ({n / 1e6:.1f} MB)")
            return out
    raise IOError(f"ninguna URL respondió para {f.clave}: {f.urls}")


def procesar(s, f: Fuente, destino: Path, forzar: bool) -> str:
    if f.archivo is None:
        return "no se descarga aquí: " + (f.nota or "ver docstring")
    faltan = [a for a in f.archivos if not (destino / a).exists()]
    zp = destino / f.archivo
    if not faltan and not forzar:
        return "ya existe"
    estado = "descomprimido (el zip ya estaba)"
    if forzar or not zp.exists():
        descargar(s, f, destino)
        estado = "descargado"
    if f.extraer is not None:
        f.extraer(zp, destino)
    faltan = [a for a in f.archivos if not (destino / a).exists()]
    if faltan:
        raise FileNotFoundError(f"{f.clave}: tras descargar faltan {faltan[:3]}…")
    return estado


def comprobar_urls(F: list[Fuente], destino: Path) -> int:
    s = _sesion()
    malos = 0
    print(f"{'clave':26s} {'estado':>6s} {'Content-Length':>15s} {'local':>15s}  coincide  url")
    for f in F:
        if f.archivo is None or not f.urls:
            continue
        local = destino / f.archivo
        loc = local.stat().st_size if local.exists() else None
        est, cl, url_ok = "—", None, f.urls[0]
        for url in f.urls:
            try:
                r = s.head(url, timeout=60, verify=f.tls, allow_redirects=True)
                est = str(r.status_code) + ("·html" if _es_html(r) else "")
                cl = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else None
                url_ok = url
                if r.status_code == 200 and not _es_html(r):
                    break
            except Exception as e:  # noqa: BLE001
                est = type(e).__name__
        ok = "sí" if (cl is not None and loc is not None and cl == loc) else ("—" if loc is None or cl is None else "NO")
        if not est.startswith("200") or "html" in est or ok == "NO":
            malos += 1
        print(f"{f.clave:26s} {est:>6s} {cl if cl is not None else '?':>15} {loc if loc is not None else '-':>15}  "
              f"{ok:8s}  {url_ok}")
    print(f"\n{malos} URL con problema (estado distinto de 200, página HTML o tamaño distinto del local)")
    return malos


def escribir_manifiesto(F: list[Fuente], raw: Path, salida: Path, respaldo: Path | None = None) -> None:
    """Una fila por archivo. `tipo`:
    - descarga: el archivo tal como lo publica la fuente (su sha256 se puede cotejar con una descarga nueva);
    - congelado: copia que no se puede volver a bajar igual (FGJ);
    - extraido: lo que el código lee después de descomprimir una descarga (00_descargar.py lo regenera sin red);
    - derivado: lo genera 00b_denue_parquet.py a partir de los zip del DENUE.
    El paquete de datos necesita solo `descarga` y `congelado`; `extraido` y `derivado` sirven para --verificar."""
    filas = []
    for f in F:
        url = f.urls[-1] if f.urls else ""   # la última URL es la que responde hoy (en el DENUE vigente, la "actual")
        rutas = ([f.archivo] if f.archivo else []) + [a for a in f.archivos if a != f.archivo]
        for a in rutas:
            p = raw / a
            if not p.exists() and respaldo is not None and (respaldo / a).exists():
                p = respaldo / a
                print(f"  {a}: se toma de {respaldo}")
            if not p.exists():
                print(f"  ⚠ falta {a}; no entra al manifiesto")
                continue
            if f.grupo == "denue_parquet":
                tipo = "derivado"
            elif f.archivo is None:
                tipo = "congelado"
            else:
                tipo = "descarga" if a == f.archivo else "extraido"
            filas.append({"ruta": a, "bytes": p.stat().st_size, "sha256": sha256(p), "tipo": tipo, "grupo": f.grupo,
                          "nivel": f.nivel, "fuente": f.fuente, "licencia": f.licencia,
                          "url": url if tipo in ("descarga", "congelado") else
                          f"(de {f.archivo})" if f.archivo else "(00b_denue_parquet.py)",
                          "uso": f.uso})
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    for nivel in ("nucleo", "complementario"):
        for tipo in ("descarga", "congelado", "extraido", "derivado"):
            x = [r["bytes"] for r in filas if r["nivel"] == nivel and r["tipo"] == tipo]
            if x:
                print(f"  {nivel:15s} {tipo:10s} {len(x):4d} archivos {sum(x) / 1e6:9,.1f} MB")
    print(f"→ {salida}")


def verificar(destino: Path, grupos: set[str] | None, niveles: set[str]) -> int:
    if not MANIFIESTO.exists():
        sys.exit(f"No existe {MANIFIESTO}")
    malos = 0
    with open(MANIFIESTO, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["nivel"] not in niveles or (grupos and r["grupo"] not in grupos):
                continue
            p = destino / r["ruta"]
            if not p.exists():
                est = "FALTA"
            elif p.stat().st_size != int(r["bytes"]) or sha256(p) != r["sha256"]:
                est = "DIFIERE"
            else:
                est = "ok"
            malos += est != "ok"
            print(f"  {est:8s} {r['ruta']}")
    print(f"\n{malos} archivo(s) faltan o difieren del manifiesto" if malos else "\nTodo coincide con el manifiesto")
    return malos


# ============================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", help="grupos separados por coma (ver --listar)")
    ap.add_argument("--todo", action="store_true", help="incluye los complementos (factibilidad y validación)")
    ap.add_argument("--forzar", action="store_true", help="descarga y descomprime aunque ya exista")
    ap.add_argument("--destino", type=Path, default=RAW, help=f"carpeta raíz de los datos crudos (por omisión {RAW})")
    ap.add_argument("--verificar", action="store_true", help="compara sha256 con analisis/MANIFIESTO_DATOS.csv")
    ap.add_argument("--comprobar-urls", action="store_true", help="HEAD a cada URL y compara su tamaño con la copia local")
    ap.add_argument("--manifiesto", action="store_true", help="regenera analisis/MANIFIESTO_DATOS.csv con los datos locales")
    ap.add_argument("--respaldo", type=Path, help="con --manifiesto: carpeta donde buscar lo que falte en data/raw")
    ap.add_argument("--listar", action="store_true", help="lista las fuentes y sale")
    a = ap.parse_args()

    F = catalogo()
    grupos = set(a.solo.split(",")) if a.solo else None
    if grupos:
        desconocidos = grupos - {f.grupo for f in F}
        if desconocidos:
            sys.exit(f"Grupos desconocidos: {sorted(desconocidos)}. Disponibles: {sorted({f.grupo for f in F})}")
    niveles = {"nucleo", "complementario"} if (a.todo or a.manifiesto) else {"nucleo"}
    sel = [f for f in F if f.nivel in niveles and (grupos is None or f.grupo in grupos)]
    destino = a.destino.resolve()

    if a.listar:
        for f in F:
            print(f"{f.grupo:18s} {f.nivel:15s} {f.clave:26s} {f.archivo or '(' + f.nota + ')'}")
        return
    if a.manifiesto:
        escribir_manifiesto(F if grupos is None else sel, RAW, MANIFIESTO, a.respaldo)
        return
    if a.verificar:
        sys.exit(1 if verificar(destino, grupos, niveles) else 0)
    if a.comprobar_urls:
        sys.exit(1 if comprobar_urls(sel, destino) else 0)

    print(f"Destino: {destino}")
    s = _sesion()
    errores = 0
    for f in sel:
        try:
            print(f"  {f.clave:26s} {procesar(s, f, destino, a.forzar)}")
        except Exception as e:  # noqa: BLE001
            errores += 1
            print(f"  {f.clave:26s} ERROR: {e}")
    print("\nSiguientes pasos: 00b_denue_parquet.py (parquet del DENUE) y copiar la FGJ congelada "
          "(fgj/fgj_delitos_objetivo_2016_2025.parquet) del paquete de datos; luego --verificar.")
    sys.exit(1 if errores else 0)


if __name__ == "__main__":
    main()
