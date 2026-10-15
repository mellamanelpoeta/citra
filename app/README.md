# App web: riesgo delictivo y actividad económica en la CDMX

Web app estática (Vite + React + TypeScript + MapLibre GL) que muestra qué AGEB de la Ciudad de México tienen más robo a negocio del que se esperaría dada su actividad económica. Consume los JSON/GeoJSON estáticos de `public/data/`, que son la salida del pipeline de análisis (esquema en [`src/types/data.ts`](src/types/data.ts)). No tiene backend.

## Requisitos

Solo **Docker** (Docker Desktop en macOS/Windows). No se necesita Node, pnpm ni `node_modules` en tu máquina: todo corre en el contenedor.

## Uso

Todos los comandos se ejecutan desde la **raíz del repositorio**.

```bash
# Servidor de desarrollo con hot reload → http://localhost:5173
docker compose up app

# Build de producción (el mismo comando que ejecuta Vercel)
docker compose run --rm app pnpm build

# Revisar tipos
docker compose run --rm app pnpm typecheck

# Probar el build de producción → http://localhost:4173
docker compose --profile preview up --build preview

# Validar public/data/ contra el esquema con zod
docker compose run --rm app pnpm valida-datos

# Agregar una dependencia (queda fijada con versión exacta en pnpm-lock.yaml)
docker compose run --rm app pnpm add <paquete>
```

La primera vez, `docker compose up app` construye la imagen e instala las dependencias en un volumen de Docker. Si cambia `pnpm-lock.yaml`, el contenedor vuelve a instalar al arrancar.

### Editor (VS Code)

Como `node_modules` vive en el contenedor, VS Code en el host no encuentra los tipos. Abre el proyecto con **Dev Containers → Reopen in Container** y elige la configuración *Frontend (app/)* (`.devcontainer/frontend/`). Así el editor y la terminal trabajan dentro del contenedor y el servidor de desarrollo arranca solo.

## Seguridad de dependencias (pnpm)

- La versión de pnpm está fijada en `package.json → packageManager` y la activa corepack. Se eligió la línea 10.x (mantenida) en vez de la 12.x porque su configuración de seguridad está verificada.
- `pnpm-workspace.yaml`:
  - `minimumReleaseAge: 1440`: no se instalan versiones publicadas hace menos de un día.
  - `onlyBuiltDependencies: []`: ningún paquete puede ejecutar scripts de instalación. `esbuild` funciona sin el suyo.
- `.npmrc`: `save-exact` (sin rangos `^`) y `engine-strict`.
- Las instalaciones usan siempre `--frozen-lockfile`.
- La imagen de Node está fijada por digest y corre con un usuario sin privilegios.

## Datos

| Archivo en `public/data/` | Contenido |
|---|---|
| `meta.json` | Años, umbrales, veredictos de las hipótesis A/B/C, fuentes y cobertura del análisis |
| `ageb.geojson` | 2,431 AGEB urbanas (MG2020) simplificadas a 12 m conservando las fronteras compartidas |
| `ageb_metrics.json` | Observados, esperados, RR, P(exceso), categoría y ranks por AGEB y año |
| `prioridades.json` | Top 10% por riesgo ajustado, con persistencia y visibilidad por conteo |
| `colonias.json` | Índice de búsqueda de colonias y colonias similares |
| `giros.json` | Grupos de giros y su mensaje en lenguaje llano |
| `alcaldias.geojson`, `alcaldias_metrics.json` | Las 16 alcaldías, con las métricas de sus AGEB sumadas por año y la denuncia relativa |
| `colonias.geojson`, `colonias_metrics.json` | Polígonos del catálogo de colonias y sus métricas por año (AGEB repartidas por área) |
| `fuera_analisis.geojson` | Las 22 AGEB rurales, que el análisis excluye; el mapa las dibuja rayadas |

El modelo es por AGEB urbana. El mapa se puede ver por alcaldía, colonia o AGEB: los niveles mayores suman los robos observados y esperados de sus AGEB.

Los escribe el pipeline de análisis en Python (no incluido en este repositorio), que los valida con pydantic antes de escribirlos. La app los valida otra vez con zod contra el mismo esquema: en el navegador, solo en desarrollo (`src/data/schemas.ts`), y desde la terminal con `pnpm valida-datos`. En producción no se incluye la validación.

Aparte, `public/cdmx-mascara.geojson` es el contorno de la CDMX (Marco Geoestadístico 2020, simplificado a ~50 m) invertido: la app lo usa para atenuar lo que queda fuera de la ciudad. No forma parte de la salida del pipeline.

Toda la carga pasa por `src/data/index.ts`.

## Estructura

```
src/
├── types/data.ts       # Esquema de los archivos de public/data/
├── data/               # Única capa de carga (+ esquemas zod)
├── lib/                # Estado (zustand), derivados, formato, configuración
└── components/         # Mapa, controles, página de contexto y paneles (Anormal, Validación, Prioridades, Consulta, AGEB)
scripts/valida-datos.ts # Valida public/data/ con los esquemas zod
```

## Deploy en Vercel

Sitio estático; Vercel lo construye sin Docker.

1. Importa el repositorio en Vercel con **Root Directory = `app`**. Framework: Vite; salida: `dist`.
2. Vercel detecta pnpm por `pnpm-lock.yaml`. Para que use exactamente la versión de `packageManager`, define la variable de entorno `ENABLE_EXPERIMENTAL_COREPACK=1` (verificar en la documentación vigente de Vercel).
3. Node 24.x (lo toma de `engines.node`, igual que la imagen Docker).
4. Los archivos de `public/data/` deben estar versionados: Vercel no corre el pipeline.

## Notas

- **Basemap:** CARTO Dark Matter. Sus [términos](https://carto.com/legal/basemap-terms/) permiten el uso académico (no comercial) gratis, con atribución a OpenStreetMap y CARTO (visible en el mapa). Desde septiembre de 2026 CARTO pide una API key y puede poner marca de agua a los mosaicos que se piden sin ella; la app aún no usa key. Si el basemap no responde, la app usa un fondo liso y el mapa sigue funcionando sin internet externo.
- **MapLibre 6** ubica su worker con `import.meta.url`, lo que no sobrevive al empaquetado; `src/lib/maplibre.ts` lo resuelve con `?worker&url` de Vite.
- Pensada para laptop/proyector de 1366×768 o más. No es responsiva para móvil.
