// Valida los archivos de public/data contra los esquemas zod de la app (los mismos que usa el navegador en
// desarrollo). Complementa la validación pydantic del pipeline que los exporta: si ambas pasan, pipeline y app
// coinciden en el contrato.
//
// Uso (dentro del contenedor): pnpm valida-datos

import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  agebGeoSchema,
  agebMetricsSchema,
  alcaldiasGeoSchema,
  coloniasGeoSchema,
  coloniasSchema,
  girosSchema,
  metaSchema,
  prioridadesSchema,
  ruralGeoSchema,
  validacionSchema,
  zonaMetricsSchema,
} from "../src/data/schemas";

const DIR = process.env.OUT_DIR ?? "public/data";
const archivos = {
  "meta.json": metaSchema,
  "ageb.geojson": agebGeoSchema,
  "ageb_metrics.json": agebMetricsSchema,
  "prioridades.json": prioridadesSchema,
  "colonias.json": coloniasSchema,
  "giros.json": girosSchema,
  "alcaldias.geojson": alcaldiasGeoSchema,
  "alcaldias_metrics.json": zonaMetricsSchema,
  "colonias.geojson": coloniasGeoSchema,
  "colonias_metrics.json": zonaMetricsSchema,
  "fuera_analisis.geojson": ruralGeoSchema,
  "validacion.json": validacionSchema,
};

let errores = 0;
for (const [nombre, esquema] of Object.entries(archivos)) {
  const r = esquema.safeParse(JSON.parse(readFileSync(join(DIR, nombre), "utf8")));
  if (r.success) {
    console.log(`ok     ${nombre}`);
  } else {
    errores++;
    console.log(`ERROR  ${nombre}`);
    for (const i of r.error.issues.slice(0, 5)) console.log(`       ${i.path.join(".")}: ${i.message}`);
  }
}
process.exit(errores ? 1 : 0);
