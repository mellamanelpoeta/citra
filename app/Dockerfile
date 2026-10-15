# syntax=docker/dockerfile:1
# Imagen solo para desarrollo y verificación local. Vercel construye el sitio sin Docker.

# Node 24 LTS fijado por digest (misma versión mayor que engines.node / Vercel).
ARG NODE_IMAGE=node:24-alpine@sha256:ebfe2f90462722a7a4de65e91990e97fe0d401c70e0e762c5b53302f905ec1c1

FROM ${NODE_IMAGE} AS base
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    npm_config_store_dir=/pnpm/store \
    npm_config_package_import_method=copy
# corepack fija pnpm a la versión de "packageManager" (no se instala con npm -g).
RUN corepack enable pnpm \
 && mkdir -p /app/node_modules /pnpm/store \
 && chown -R node:node /app /pnpm
USER node
WORKDIR /app
COPY --chown=node:node package.json ./
RUN corepack install

# Desarrollo: el código llega por bind mount y node_modules vive en un volumen.
FROM base AS dev
EXPOSE 5173
CMD ["sh", "-c", "pnpm install --frozen-lockfile && pnpm dev"]

# Build: mismo comando que ejecuta Vercel.
FROM base AS build
COPY --chown=node:node package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
RUN pnpm install --frozen-lockfile
COPY --chown=node:node . .
RUN pnpm build

# Vista previa del artefacto de producción (dist/).
FROM build AS preview
EXPOSE 4173
CMD ["pnpm", "preview"]
