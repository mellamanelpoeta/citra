# Pipeline de datos y análisis

Python 3.12 (entorno `uv` en `/.venv`; versiones fijadas en `analisis/requirements.txt`). Todo corre desde cero a partir de `data/raw/_samples/`. Los comandos se corren desde la raíz del repositorio.

## 0. Datos crudos (`data/raw/_samples/`, fuera de git; ver `analisis/DATOS.md`)

```bash
.venv/bin/python analisis/pipeline/00_descargar.py           # INEGI y datos.cdmx → data/raw/_samples/ (idempotente; --todo agrega complementos)
.venv/bin/python analisis/pipeline/00b_denue_parquet.py      # zip del DENUE → denue/parquet/ (idéntico byte a byte; --todos para los 19 cortes)
# FGJ: copia congelada fgj/fgj_delitos_objetivo_2016_2025.parquet del paquete de datos.
.venv/bin/python analisis/pipeline/00a_fgj_api.py            # opcional: repite la consulta a la API y la compara con la copia congelada (escribe un archivo nuevo con fecha)
.venv/bin/python analisis/pipeline/00_descargar.py --verificar   # sha256 contra analisis/MANIFIESTO_DATOS.csv
```

## Orden completo para regenerar los datos de la app

`05_export.py` y `05c_ficha.py` leen salidas de varios scripts anteriores; en un clon limpio hay que correrlos en este orden (~60 min):

```bash
.venv/bin/python analisis/pipeline/_factibilidad/ensu_alcaldia.py   # → data/interim/ensu_alcaldia.parquet (lo usan 04d y 05c)
.venv/bin/python analisis/pipeline/01_datos.py
.venv/bin/python analisis/pipeline/02_features.py
.venv/bin/python analisis/pipeline/04_modelos.py
.venv/bin/python analisis/pipeline/04c_robustez_C.py
.venv/bin/python analisis/pipeline/04d_alcaldia_denuncia.py
.venv/bin/python analisis/pipeline/04e_cifra_negra_alcaldia.py      # necesita 04c y 04d
.venv/bin/python analisis/pipeline/04g_anticipacion.py
.venv/bin/python analisis/pipeline/06_dinamica_datos.py
.venv/bin/python analisis/pipeline/07_dinamica_modelos.py
.venv/bin/python analisis/pipeline/08_rotacion.py                   # necesita 06 y 07
.venv/bin/python analisis/lab/mecanismos/parte1c_nodo_estimado.py   # → app/src/data/colchon.json (lo lee 05_export)
.venv/bin/python analisis/pipeline/05_export.py
.venv/bin/python analisis/pipeline/05c_ficha.py
```

## Detalle por etapa

```bash
.venv/bin/python analisis/pipeline/01_datos.py      # ~10 s: homologación FGJ, unión espacial, atributos de AGEB
.venv/bin/python analisis/pipeline/02_features.py   # ~3 s: panel AGEB × año
cd analisis/pipeline && ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 03_eda.ipynb   # EDA + figuras
.venv/bin/python analisis/pipeline/04_modelos.py    # ~10 min: modelos, hipótesis A/B/C, robustez, supuestos → outputs/resultados.md
.venv/bin/python analisis/pipeline/04g_anticipacion.py # ~2 min: lista del año siguiente (más peso a lo reciente) y prueba un año atrás
.venv/bin/python analisis/pipeline/05_export.py     # ~1 min: 12 archivos del contrato (v0.5) → app/public/data/
.venv/bin/python analisis/pipeline/05c_ficha.py      # ficha.json (v0.9): contexto del detalle de cada AGEB
```

Análisis posteriores a T2b (2026-10-11; exploratorios, no cambian veredictos ni el ranking). Se corren desde la raíz del repositorio:

```bash
.venv/bin/python analisis/pipeline/04b_sesgo_denuncia.py    # ~2 min: β bajo los canales del sesgo de denuncia
.venv/bin/python analisis/pipeline/04c_robustez_C.py        # ~8 min: robustez de C, exposición, causalidad inversa, B aislado, validación 2024
.venv/bin/python analisis/pipeline/04d_alcaldia_denuncia.py # ~2 min: ranking con efectos de alcaldía y denuncia relativa con la ENSU
.venv/bin/python analisis/pipeline/04e_cifra_negra_alcaldia.py # ~15 min: denuncia por alcaldía con la ENVIPE (2021–2026) y su efecto en C, A y B
```

T6 · inseguridad y dinámica empresarial (hipótesis D y E, 2026-10-13; protocolo en `analisis/lab/dinamica/README.md`). No toca A–C, el contrato ni la app:

```bash
.venv/bin/python analisis/pipeline/06_dinamica_datos.py     # ~10 s (la 1.ª vez asigna el DENUE 2018-11 y el robo 2016–2018 a AGEB y los cachea)
.venv/bin/python analisis/pipeline/07_dinamica_modelos.py   # ~15 min: prueba M, D, D′, heterogeneidad, E, robustez → outputs/resultados_dinamica.md
```

`05_export.py` lee `outputs/modelos/robustez_C.csv` si existe, para los textos de `meta.json` y `giros.json`; conviene correr 04c antes que 05.

Dependencias (además de las del EDA): `lightgbm`, `libpysal`, `esda`, `h3`, `pydantic` (`uv pip install --python .venv/bin/python …`). En macOS, LightGBM necesita `libomp`; si no está instalado con Homebrew, `modelado.lightgbm()` usa la copia que trae scikit-learn y relanza el script (por eso 04 debe correrse como archivo, no con `python -c`).

| Archivo | Qué hace |
|---|---|
| `common.py` | Rutas, constantes (años, corte DENUE fijo, estratos, giros SCIAN candidatos, lista "efectivo", partición sectorial) y utilidades |
| `config/homologacion_fgj.csv` | **Fuente de verdad** de la homologación FGJ: cada valor de `delito` → grupo, objetivo, farderos y justificación. `01_datos.py` falla si aparece un valor no homologado |
| `00_descargar.py` | Descarga y descomprime las fuentes oficiales en `data/raw/_samples/`; `--verificar` y `--comprobar-urls` |
| `00a_fgj_api.py` | Repite la consulta a la API de la FGJ y la compara con la copia congelada |
| `00b_denue_parquet.py` | Zip del DENUE → parquet (11 columnas de texto + `version`) |
| `01_datos.py` | FGJ (API, año del hecho) → `data/interim/delitos_ageb.parquet`; DENUE (cortes de noviembre 2019–2024) → `denue_ageb.parquet`; atributos de AGEB, colonia dominante y distancia a estaciones |
| `02_features.py` | → `data/processed/panel_ageb_anio.parquet` |
| `03_eda.ipynb` | EDA del problema (T2a): hipótesis registradas, vistas previas de A/B/C y evidencia para S1–S4. La serie DENUE 2016–2026 (fig05a) se lee de `data/interim/_eda/q5_denue_serie.csv`, que genera `_factibilidad/03_denue.py` |
| `outputs/homologacion_fgj_final.csv` | Tabla de homologación con conteos por año del hecho (evidencia D2) |
| `outputs/eda/` | Figuras `figNN_*.png` y `decisiones.md` |
| `modelado.py` | Maquinaria de T2b, promovida de `analisis/lab/lab_comun.py` sin cambios de método: NB2 (M0/M1), LightGBM Poisson (M2), bloques espaciales, bootstrap, contracción Poisson-gamma, panel H3 |
| `04_modelos.py` | Modelos, veredicto de A/B/C con los umbrales ex ante, tabla de robustez, verificación de supuestos y figuras `outputs/modelos/fig1N_*.png`. Escribe `outputs/resultados.md` (conserva lo escrito a mano después del marcador de interpretación) y `data/processed/modelos/` |
| `04b_sesgo_denuncia.py` | Hipótesis A bajo los canales del sesgo de denuncia (tamaño de empresa con la ENVE, alcaldía, tipo de robo) y punto de quiebre → `outputs/modelos/sesgo_denuncia.*`, `fig15` |
| `04c_robustez_C.py` | C con solo robo con violencia (con referencia de ruido), Ê sin grupos de efectivo, exposición (personal ocupado, AGEB con pocos negocios), `E` de 2016 y causalidad inversa, B descompuesto (composición vs controles), validación 2020–2023 → 2024 → `outputs/modelos/robustez_C.*`, `fig16`. Cachea `data/interim/_eda/denue2016_robo1618_ageb.parquet` |
| `04d_alcaldia_denuncia.py` | Ê con efectos de alcaldía, denuncia relativa por alcaldía con la ENSU y Ê con *offset* de denuncia → `outputs/modelos/alcaldia_denuncia.*`, `fig17` |
| `04e_cifra_negra_alcaldia.py` | Denuncia ante el MP por alcaldía de ocurrencia con la ENVIPE (ediciones 2021–2026; EE por linealización de Taylor, contracción bayesiana empírica), Ê de M2 con *offset* log r (alcaldía × tamaño de empresa) y Monte Carlo con reajuste; efecto en la lista de C, β de A e IRR de B; magnitud con la cifra negra de la ENVE → `outputs/modelos/cifra_negra_alcaldia.*`, `fig18`, `data/interim/envipe_alcaldia.parquet` |
| `06_dinamica_datos.py` | T6: DENUE 2018-11/2019-11/2023-11/2024-11 por AGEB (N, personal ocupado, composición, estratos), robo 2016–2023 (V1, con/sin violencia, V3), distancia a estación 2016–2023, rasgos del Censo 2020 y % de superficie no residencial, Censos Económicos 2019/2024 armonizados e imputación monetaria calibrada → `data/processed/panel_dinamica.parquet`, `data/interim/ce_alc_sector_estrato.parquet`, `denue_alcaldia_sector.parquet`, `outputs/dinamica/imputacion_*.csv` |
| `07_dinamica_modelos.py` | T6: prueba de medición M (DENUE vs CE), `rr_pre`/`rr_post`, D (NB + bootstrap de bloques), D′ (devianza fuera de muestra), heterogeneidad con Holm, perfiles y clasificador de E, robustez → `outputs/dinamica/`, `fig20`–`fig23`, `data/processed/modelos/dinamica_ageb.parquet` y `outputs/resultados_dinamica.md` (conserva lo escrito tras el marcador) |
| `04g_anticipacion.py` | Lista del año siguiente (2025) con pesos de vida media 1.5 años sobre O y Ê de 04_modelos, esperado y anticipado del año siguiente, ranking con solo robo con violencia (aviso ⚠) y prueba un año atrás con M2 reentrenado sin el año evaluado → `data/processed/modelos/anticipado*.parquet`, `validacion_2024.parquet`, `outputs/modelos/anticipacion.csv` (diseño §13, contrato v0.5) |
| `05_export.py` | `app/public/data/*.json` validados con pydantic contra `docs/6.Contrato-Datos.md`. Necesita 04_modelos y 04g |
| `05c_ficha.py` | `app/public/data/ficha.json` (contrato v0.9): contexto del detalle de cada AGEB (composición por giro, densidad y tamaño del DENUE 2023-11; Censo 2020; negocios que entran y salen 2019-11 → 2024-11; ENVIPE y ENSU por alcaldía). Necesita 04e y 08_rotacion; no toca los demás archivos |
| `05b_ejemplo_colonia.py` | Ilustración de la página Contexto, fuera del contrato: establecimientos del DENUE 2023-11 en Barrio Santa Catarina (Coyoacán) con giro y estrato, proyectados a SVG → `app/src/data/ejemplo_colonia.json` |
| `_factibilidad/` | Scripts del EDA de factibilidad (2026-10-03); se conservan como referencia. Incluye `ensu_alcaldia.py` y `validar_ensu.py` (2026-10-11), que construyen y validan `data/interim/ensu_alcaldia.parquet` (`docs/_factibilidad/03-ensu.md`) |

## Intermedios (`data/interim/`)

| Archivo | Grano | Columnas principales |
|---|---|---|
| `delitos_ageb.parquet` | carpeta (delitos de interés, 2019–2024) | `_id`, `fecha_hecho`, `anio_hecho`, `delito_norm`, `grupo`, `objetivo`, `farderos`, `cvegeo`, `tipo_asig` (URBANA / RURAL / SIN_COORD), `latitud`, `longitud` |
| `denue_ageb.parquet` | establecimiento × corte (solo AGEB urbanas) | `corte` (año del corte de noviembre), `cvegeo`, `codigo_act` (SCIAN, 6 dígitos), `per_ocu`, `latitud`, `longitud` |
| `ageb_atributos.parquet` | AGEB urbana (2,431) | `cvegeo`, `cve_mun`, `alcaldia`, `area_km2`, centroide (`x`, `y` en LCC; `lon`, `lat`), `POBTOT`, `cve_col`, `colonia`, `pct_area_colonia` |
| `ageb_geom.parquet` | AGEB urbana | `cvegeo`, `geometry` (MG2020, LCC ITRF2008) |
| `ageb_colonia.parquet` | AGEB × colonia | colonias que cubren > 5% del área de la AGEB (`pct_ageb`), para `colonias.json` |
| `ageb_dist_estacion.parquet` | AGEB × año | `dist_estacion_m` a la estación de Metro/Metrobús abierta ese año |
| `envipe_alcaldia.parquet` | delito ocurrido en la CDMX con respuesta de denuncia (ENVIPE 2021–2026, 24,025 filas; lo arma `04e`) | `edicion`, `anio_delito`, `cve_mun` (alcaldía de ocurrencia; '999' no especificada), `bpcod`, `lugar`, `denuncia`, `carpeta`, `fac_del`, `estrato`, `upm`, `n_upm_est` (UPM de la muestra en el estrato) |
| `ensu_alcaldia.parquet` | alcaldía × trimestre (sep-2019 a dic-2025, 400 filas) | `cve_mun`, `alcaldia`, `anio`, `trimestre`, `n`, `pct_*` (percepción, atestiguar robos, victimización del hogar) y sus `se_*`. Esquema en `docs/_factibilidad/03-ensu.md` |

## Diccionario: `data/processed/panel_ageb_anio.parquet`

Una fila por AGEB urbana del MG2020 (2,431) × año del hecho (2019–2024) = 14,586 filas. Incluye las AGEB con E = 0; se filtran al modelar.

**Identificadores y etiquetas**

| Columna | Descripción |
|---|---|
| `cvegeo` | Clave AGEB INEGI (13 caracteres) |
| `anio` | Año del hecho |
| `cve_mun`, `alcaldia` | Alcaldía (MG2020) |
| `cve_col`, `colonia` | Colonia dominante por área (catálogo CDMX). Solo es etiqueta |

**Variables objetivo** (carpetas FGJ por año del hecho, solo coordenadas válidas en AGEB urbanas; definición por fila en `config/homologacion_fgj.csv`)

| Columna | Descripción | Comparable desde |
|---|---|---|
| `y_v1_negocio` | **V1**: robo a negocio agregado (con y sin violencia, todos los subtipos, sin oficina pública) | 2019 |
| `y_v2_sin_farderos` | **V2**: V1 sin farderos | 2020 |
| `y_v2b` | **V2b**: V2 sin "S/V dentro de negocios" (variante propuesta en el EDA) | 2020 |
| `y_v3_via_publica` | **V3**: robo a transeúnte en vía pública + repartidor + cuentahabiente | 2019 |
| `y_farderos` | Farderos (con y sin violencia, todos los subtipos) | 2020 |
| `y_negocio_sv_dentro` | "Robo S/V dentro de negocios, autoservicios, conveniencia" | 2020 |
| `y_negocio_cv` | Robo a negocio con violencia (todos los subtipos) | 2019 |
| `y_ampliado` | Robo a transeúnte dentro de negocio, restaurante, hotel, cine, terminal, parques y mercados | 2019 (con salto por reclasificación) |

**DENUE** (cada variable viene dos veces: `_nov` = corte de noviembre del mismo año; `_fijo` = corte 2023-11 para todos los años; experimento S2)

| Columna | Descripción |
|---|---|
| `E_*` | **Exposición**: número de establecimientos en la AGEB |
| `po_est_*` | Personal ocupado estimado (suma de los puntos medios del estrato; "251 y más" = 300). Descriptiva |
| `n_grandes_*`, `p_grandes_*` | Número y proporción de establecimientos con ≥ 51 personas (control del gradiente de denuncia) |
| `n_autoserv_grande_*` | Supermercados/minisúpers (SCIAN 4621xx) con ≥ 51 personas |
| `pc_<giro>_*` | Proporción de cada giro candidato a "intensivo en efectivo" (mutuamente excluyentes; prefijos en `common.CANDIDATOS_EFECTIVO`): `abarrotes`, `alim_otros`, `minisuper`, `licores`, `farmacias`, `gasolineras`, `restaurantes`, `bares`, `empeno_cambio`, `tortillerias` |
| `ps_<sector>_*` | Partición sectorial que suma 1 (composición S de M1): `efectivo_barrio` y `efectivo_formal` (lista final `common.EFECTIVO`) y el resto de `comercio_menor`, `comercio_mayor`, `manufactura`, `alojamiento_alim`, `serv_prof_fin`, `serv_salud_educ`, `serv_otros`, `otros` |

Las proporciones son NaN cuando E = 0.

**Controles**

| Columna | Descripción |
|---|---|
| `pobtot`, `log_pob` | Población del Censo 2020 y `log(POBTOT + 1)` |
| `no_residencial` | 1 si POBTOT = 0 (17 AGEB comerciales o de equipamiento) |
| `dist_estacion_km` | Distancia del centroide a la estación de Metro o Metrobús abierta ese año (la L12 cuenta como cerrada en 2021–2022) |
| `area_km2`, `x`, `y`, `lon`, `lat` | Área y centroide (LCC ITRF2008 en metros; WGS84), para los *clusters* espaciales de validación |
