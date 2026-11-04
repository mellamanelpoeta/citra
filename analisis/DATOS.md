# Datos del análisis

El código de `analisis/` lee todo de `data/raw/_samples/` (en la raíz del repo). Esa carpeta no está en git: se reconstruye con el paquete de datos publicado o con los scripts de descarga. Todas las fuentes son públicas.

## Fuentes

| Fuente | Qué se usa | Para qué | Licencia o términos |
|---|---|---|---|
| **FGJ CDMX**, [Carpetas de investigación](https://datos.cdmx.gob.mx/dataset/carpetas-de-investigacion-fgj-de-la-ciudad-de-mexico) (portal de datos abiertos de la CDMX) | Carpetas de robo a negocio, a transeúnte, a repartidor, a cuentahabiente y extorsión, año del hecho 2016–2025 (corte de inicio 2025-01-31), vía la API del portal | Variable objetivo | CC BY 4.0 (declarada en el portal) |
| **INEGI**, [DENUE](https://www.inegi.org.mx/app/descarga/?ti=6), CDMX | Cortes 2016-01, 2018-11 y noviembre de 2019 a 2024 | Actividad económica: número, giro y tamaño de los establecimientos | [Términos de Libre Uso de la Información del INEGI](https://www.inegi.org.mx/inegi/terminos.html) |
| **INEGI**, [Marco Geoestadístico 2020](https://www.inegi.org.mx/temas/mg/), CDMX | AGEB urbanas y rurales, alcaldías, entidad y manzanas | Unidad de análisis (AGEB) | Términos de Libre Uso del INEGI |
| **INEGI**, [Censo de Población y Vivienda 2020](https://www.inegi.org.mx/programas/ccpv/2020/), resultados por AGEB y manzana | Población y rasgos sociodemográficos por AGEB | Controles y ficha de cada zona | Términos de Libre Uso del INEGI |
| **INEGI**, [Censos Económicos 2019 y 2024](https://www.inegi.org.mx/programas/ce/2024/), CDMX | Unidades económicas por alcaldía, sector y estrato | Prueba de medición del DENUE | Términos de Libre Uso del INEGI |
| **INEGI**, [ENVIPE](https://www.inegi.org.mx/programas/envipe/) 2021–2026 | Microdatos de victimización y denuncia | Denuncia por alcaldía | Términos de Libre Uso del INEGI |
| **INEGI**, [ENVE 2024](https://www.inegi.org.mx/programas/enve/2024/) | Tabulados de cifra negra por tamaño de empresa (CDMX) | Contexto del sesgo de denuncia | Términos de Libre Uso del INEGI |
| **INEGI**, [ENSU](https://www.inegi.org.mx/programas/ensu/) 2019–2025 | Microdatos trimestrales | Percepción de inseguridad por alcaldía | Términos de Libre Uso del INEGI |
| **Gobierno de la CDMX**, [Catálogo de colonias](https://datos.cdmx.gob.mx/dataset/02c6ce99-dbd8-47d8-aee1-ae885a12bb2f) | Polígonos de colonias (2023) | Colonia de cada AGEB y búsqueda en la app | CC BY 4.0 (declarada en el portal) |
| **STC Metro** y **Metrobús**, [líneas y estaciones](https://datos.cdmx.gob.mx/dataset/1b014317-ddb1-46c7-ac79-7330c652abe3) ([Metrobús](https://datos.cdmx.gob.mx/dataset/63a08f93-e959-430f-8dc4-bf07182401e6)) | Estaciones con año de apertura | Distancia a la estación más cercana | CC BY 4.0 (declarada en el portal) |

Las URL exactas de descarga, el tamaño y el sha256 de cada archivo están en [`MANIFIESTO_DATOS.csv`](MANIFIESTO_DATOS.csv) (columnas `ruta`, `bytes`, `sha256`, `tipo`, `grupo`, `nivel`, `fuente`, `licencia`, `url`, `uso`).

## Por qué la FGJ va congelada

La Fiscalía reclasifica carpetas de forma continua (cambian el delito, la fecha o las coordenadas de carpetas ya publicadas) y el portal vuelve a cargar sus recursos. Una consulta a la API hecha otro día puede dar conteos un poco distintos. Para que los resultados se puedan reproducir, el análisis usa una copia fija de la consulta:

- `fgj/fgj_delitos_objetivo_2016_2025.parquet`: 332,027 carpetas, 12 columnas, descargada el 2026-10-03 del recurso "Carpetas de Investigación (acumulado 2016-2024)" (`35e87ce4-8e40-4de4-8dac-0b59645c13eb`), con carpetas iniciadas hasta el 31 de enero de 2025. 2025 trae solo enero y el análisis lo excluye.
- sha256: `f2bf89da4f098f3901b56ca449e993b4a46b43e565a3bd6f3e84655a88d2a744`.

`analisis/pipeline/00a_fgj_api.py` repite esa consulta y compara el resultado con la copia congelada (filas por año y por delito, carpetas comunes y valores que cambiaron). Escribe siempre un archivo nuevo con la fecha de descarga y nunca toca la copia congelada. El 2026-10-15 la descarga completa fue idéntica byte a byte a la copia congelada.

## Cómo obtener los datos

### Opción A: paquete congelado

1. Descarga el paquete: `<URL del paquete de datos (Zenodo) — pendiente>`.
2. Descomprímelo en la raíz del repo, de modo que quede `data/raw/_samples/…`.
3. Descomprime lo que el código lee y genera los parquet del DENUE (no usa la red porque los zip ya están):

```bash
.venv/bin/python analisis/pipeline/00_descargar.py
.venv/bin/python analisis/pipeline/00b_denue_parquet.py
.venv/bin/python analisis/pipeline/00_descargar.py --verificar
```

El paquete contiene los archivos de tipo `descarga` y `congelado` del nivel `nucleo` del manifiesto: 31 archivos, 676 MB. Son los zip tal como los publican INEGI y el portal de la CDMX (sus sha256 se pueden cotejar con una descarga nueva) más la copia congelada de la FGJ.

### Opción B: desde las fuentes oficiales

```bash
.venv/bin/python analisis/pipeline/00_descargar.py            # INEGI y datos.cdmx (≈ 670 MB)
.venv/bin/python analisis/pipeline/00b_denue_parquet.py       # zip del DENUE → parquet
# copia congelada de la FGJ: del paquete (opción A), o una consulta nueva con
.venv/bin/python analisis/pipeline/00a_fgj_api.py             # escribe fgj_delitos_objetivo_descarga_<fecha>.parquet
.venv/bin/python analisis/pipeline/00_descargar.py --verificar
```

Si usas una descarga nueva de la FGJ, cópiala como `fgj/fgj_delitos_objetivo_2016_2025.parquet` solo si `00a_fgj_api.py` reporta que reproduce la copia congelada; si hay diferencias, los resultados pueden variar un poco.

Opciones útiles de `00_descargar.py`: `--solo ensu,envipe` (por grupo), `--todo` (agrega los complementos), `--forzar`, `--destino /otra/carpeta`, `--comprobar-urls` (revisa que cada URL responda y que el tamaño coincida con la copia local) y `--listar`.

### Qué contiene el manifiesto

| `tipo` | Qué es | ¿Va en el paquete? |
|---|---|---|
| `descarga` | Archivo tal como lo publica la fuente | Sí (nivel `nucleo`) |
| `congelado` | Copia fija de la FGJ | Sí |
| `extraido` | Lo que el código lee después de descomprimir; `00_descargar.py` lo regenera | No; sirve para `--verificar` |
| `derivado` | Parquet del DENUE; `00b_denue_parquet.py` lo regenera idéntico byte a byte | No; sirve para `--verificar` |

El nivel `complementario` (otros cortes del DENUE, CSV anuales de la FGJ y tabulados de la ENSU) solo lo usan los scripts de factibilidad en `analisis/pipeline/_factibilidad/` y la figura de la serie del DENUE del EDA. Se baja con `00_descargar.py --todo` y `00b_denue_parquet.py --todos`.

## Notas de las fuentes

- `archivo.datos.cdmx.gob.mx` (CSV anuales de la FGJ, solo complemento) tiene el certificado TLS vencido; el script baja esos archivos sin verificar el certificado.
- Algunas URL de la API del portal CDMX apuntan a una IP interna (10.80.148.134); en los scripts ya están sustituidas por `https://datos.cdmx.gob.mx`.
- El DENUE se publica en UTF-8 en unos cortes y en Latin-1 en otros; `00b_denue_parquet.py` detecta la codificación.
- La ENSU trae un zip por trimestre dentro del zip anual. En 2021 el zip CSV no trae marzo y ese trimestre se toma del DBF.
