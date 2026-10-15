# CITRA: riesgo delictivo y actividad económica en la Ciudad de México

Web app del proyecto de titulación (Ciencia de Datos Aplicada, ITAM), caso "Inseguridad urbana en la Ciudad de México".

**Pregunta:** ¿qué zonas de la CDMX tienen más robo a negocio del que se esperaría por su actividad económica, y cómo ha evolucionado esa relación?

Contar robos señala a los polos comerciales: donde hay más negocios, hay más robos. El análisis estima cuántos robos se esperarían en cada AGEB urbana según el número y el tipo de sus negocios (DENUE), y prioriza las zonas con más robo del esperado (**riesgo residual**). La app está pensada para dos usuarios:

- la **Secretaría de Desarrollo Económico (SEDECO)**, que prioriza zonas para sus programas de apoyo;
- los **pequeños negocios**, que consultan en lenguaje llano cómo está su colonia.

**App en producción:** _pendiente de deploy en Vercel_. Mientras tanto, se corre en local con Docker (ver abajo).

## Qué muestra la app

| Vista | Para qué sirve |
|---|---|
| **Contexto** | Página de entrada sin mapa: el problema, qué se mide (robo a negocio y actividad económica), dos dificultades del análisis (cifra negra y causalidad en dos sentidos), lo que muestran los datos (el robo crece más despacio que el comercio), el método y los hallazgos |
| **¿Qué es anormal?** | Recorrido en tres pasos que cambia el mapa al bajar por el panel: dónde está el comercio → dónde se denuncian más robos → dónde hay más robo del esperado. Se ve por alcaldía, colonia o AGEB |
| **¿Habría funcionado?** | La misma lista armada solo con 2020–2023 y comparada con lo que pasó en 2024: qué zonas acertaron y cuáles no, frente a la lista por conteo |
| **¿Qué priorizar?** | Lista 2025: zonas con más robo del que les corresponde por su actividad (más peso a los años recientes), con filtros y exportación a CSV |
| **Consulta mi zona** | Búsqueda de colonia y mensaje por tipo de negocio, sin cifras técnicas |

## Resultados en breve

Robo a negocio denunciado (FGJ CDMX), 2020–2024, en 2,431 AGEB urbanas.

| Hipótesis | Veredicto | Lectura |
|---|---|---|
| **A. ¿Cuánto?** El robo crece menos que proporcionalmente con los negocios (β < 1) | Confirmada: β = 0.88 [0.81, 0.95] | Comparar zonas por conteo o por robos por negocio no es justo. Depende de la escala: en hexágonos, β ≈ 1 |
| **B. ¿Cuáles?** La composición de giros explica el robo mejor que el número de negocios | Confirmada: +12.1% de devianza fuera de muestra | Los giros formales de efectivo concentran el robo con violencia; las tiendas de barrio denuncian menos |
| **C. ¿Dónde?** Priorizar por riesgo ajustado señala zonas distintas y persistentes | Confirmada: Jaccard = 0.37 con el top por conteo | La lista se arma para el año siguiente. Armada con 2020–2023 (sin que el modelo vea 2024), el 70% de sus zonas tuvo más robo del esperado en 2024 (lista por conteo: 62%; ciudad: 32%) |

## Cómo correr la app

Solo se necesita **Docker** (Docker Desktop en macOS/Windows). No hace falta instalar Node ni pnpm.

```bash
git clone <url-del-repositorio> && cd <carpeta>
docker compose up app                                  # desarrollo → http://localhost:5173
docker compose --profile preview up --build preview    # build de producción → http://localhost:4173
```

La primera vez tarda unos minutos en construir la imagen e instalar dependencias. Más comandos, decisiones técnicas y deploy: [`app/README.md`](app/README.md).

## Estructura

```
├── app/                  # web app (Vite + React + TypeScript + MapLibre), sin backend
│   ├── src/              # código de la app
│   ├── public/data/      # resultados del análisis en JSON/GeoJSON (los que lee la app)
│   └── scripts/          # validación de los datos contra el esquema
├── docker-compose.yml    # entorno de desarrollo y vista previa
└── .devcontainer/        # configuración opcional para VS Code
```

## De dónde salen los datos

Los archivos de `app/public/data/` son la salida de un pipeline de Python (limpieza, modelos de conteo y prueba de hipótesis) que no forma parte de este repositorio; el método se describe en el trabajo escrito. Su esquema está en [`app/src/types/data.ts`](app/src/types/data.ts) y se valida con `docker compose run --rm app pnpm valida-datos`.

**Fuentes:** carpetas de investigación de la FGJ CDMX (Datos Abiertos CDMX), DENUE (INEGI), Marco Geoestadístico 2020 (INEGI), Censo de Población y Vivienda 2020 (INEGI), catálogo de colonias de la CDMX y estaciones de Metro y Metrobús. Para la sensibilidad de la denuncia se usaron la ENVE y la ENSU (INEGI).

**Limitación principal:** se mide robo **denunciado**. La cifra negra es alta y se probó que no explica las conclusiones a través de los canales medibles (tamaño de empresa, alcaldía, tipo de robo y giro), pero no puede descartarse zona por zona.

## Uso de IA generativa

El proyecto se desarrolló con apoyo de Claude (Anthropic). Qué se delegó, qué decidió el alumno y cómo se verificó se declara en el apéndice del trabajo escrito.
