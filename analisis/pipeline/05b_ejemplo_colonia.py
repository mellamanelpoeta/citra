"""05b · Ejemplo de "actividad económica" para la página Contexto → app/src/data/ejemplo_colonia.json

Los establecimientos del DENUE 2023-11 dentro de una colonia de Coyoacán, ya proyectados a coordenadas de SVG, con su
giro (los mismos grupos que el modelo, agrupados en 5 para que la leyenda se lea) y su estrato de personal ocupado.
Es solo una ilustración: no entra al modelo ni al contrato de datos.

Uso: .venv/bin/python analisis/pipeline/05b_ejemplo_colonia.py
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from common import CANDIDATOS_EFECTIVO, EFECTIVO, GRANDES, PER_OCU, RAW, ROOT, SECTORES, clasifica

COLONIA, ALCALDIA = "Barrio Santa Catarina", "COYO"
ANCHO = 400  # unidades de SVG; el alto sale de la forma de la colonia
MARGEN = 10
SALIDA = ROOT / "app/src/data/ejemplo_colonia.json"

# Grupos de la leyenda (en orden de la tarjeta: comercio, manufactura, servicios y los dos grupos de efectivo).
GIROS = {
    "comercio": ("Otro comercio", ["comercio_menor", "comercio_mayor"]),
    "manufactura": ("Manufactura y otros", ["manufactura", "otros"]),
    "servicios": ("Servicios", ["serv_prof_fin", "serv_salud_educ", "serv_otros", "alojamiento_alim"]),
    "efectivo_barrio": ("Alto efectivo tradicional", ["efectivo_barrio"]),
    "efectivo_formal": ("Alto efectivo formal", ["efectivo_formal"]),
}
ESTRATOS = list(PER_OCU)  # de 0 a 5 personas a 251 y más

col = gpd.read_file(RAW / "colonias/catalogo/poligonos_colonias_cdmx.shp")
col = col[(col.colonia == COLONIA) & col.alc.str.upper().str.contains(ALCALDIA)].to_crs(32614)
assert len(col) == 1, len(col)
poli = col.geometry.iloc[0]

e = pd.read_parquet(RAW / "denue/parquet/denue_09_2023_11.parquet", columns=["codigo_act", "per_ocu", "latitud", "longitud"])
e = gpd.GeoDataFrame(e, geometry=gpd.points_from_xy(e.longitud.astype(float), e.latitud.astype(float)), crs=4326).to_crs(32614)
e = e[e.within(poli)].copy()

# Mismo giro que 02_features: primero los dos grupos de efectivo, luego el sector del resto.
cand = clasifica(e.codigo_act, CANDIDATOS_EFECTIVO, resto="")
sector = cand.map({c: g for g, cs in EFECTIVO.items() for c in cs}).fillna(clasifica(e.codigo_act, SECTORES))
e["giro"] = sector.map({s: k for k, (_, ss) in GIROS.items() for s in ss})
e["estrato"] = e.per_ocu.map(ESTRATOS.index)
assert e.giro.notna().all() and e.estrato.notna().all()

# Varios negocios comparten coordenada (plazas, edificios): se separan en espiral unos metros para que se vean todos.
x, y = e.geometry.x.to_numpy(), e.geometry.y.to_numpy()
orden = e.groupby([x.round(1), y.round(1)]).cumcount().to_numpy()
ang = orden * 2.39996  # ángulo áureo
rad = 6.0 * np.sqrt(orden)
x, y = x + rad * np.cos(ang), y + rad * np.sin(ang)

minx, miny, maxx, maxy = poli.bounds
k = (ANCHO - 2 * MARGEN) / (maxx - minx)
alto = round((maxy - miny) * k + 2 * MARGEN)
px = lambda v: round((v - minx) * k + MARGEN, 1)
py = lambda v: round((maxy - v) * k + MARGEN, 1)
ruta = "M" + "L".join(f"{px(a)},{py(b)}" for a, b in poli.simplify(2).exterior.coords) + "Z"

claves = list(GIROS)
puntos = [[px(a), py(b), claves.index(g), int(s)] for a, b, g, s in zip(x, y, e.giro, e.estrato)]
salida = {
    "_fuente": "analisis/pipeline/05b_ejemplo_colonia.py. No editar a mano.",
    "colonia": COLONIA,
    "alcaldia": "Coyoacán",
    "corte": "2023-11",
    "area_km2": round(poli.area / 1e6, 2),
    "ancho": ANCHO,
    "alto": alto,
    "poligono": ruta,
    "giros": [{"id": k, "nombre": GIROS[k][0], "n": int((e.giro == k).sum())} for k in claves],
    "estratos": [{"etiqueta": s.replace(" personas", ""), "n": int((e.per_ocu == s).sum())} for s in ESTRATOS],
    "n_grandes": int(e.per_ocu.isin(GRANDES).sum()),
    "puntos": puntos,  # [x, y, índice de giro, índice de estrato]
}
SALIDA.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")))
print(f"{COLONIA}: {len(puntos)} establecimientos, {salida['n_grandes']} con 51 o más, "
      f"{salida['area_km2']} km², SVG {ANCHO}×{alto} → {SALIDA}")
print({g["nombre"]: g["n"] for g in salida["giros"]})
