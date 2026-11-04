"""EDA de factibilidad DENUE: serie de conteos por corte, campos, coordenadas, estabilidad de ids (preguntas 5-6)."""
import io
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import EDA, FIGS, RAW, lee_mg, puntos

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
DEN = RAW / "denue"
PQ = DEN / "parquet"


def csv_en_zip(z):
    zf = zipfile.ZipFile(z)
    name = [n for n in zf.namelist() if n.lower().endswith(".csv") and "conjunto" in n.lower()][0]
    return zf, name


def header(z):
    zf, name = csv_en_zip(z)
    with zf.open(name) as fh:
        raw = fh.readline()
    for enc in ("utf-8-sig", "latin1"):
        try:
            return [c.strip().strip('"').lstrip("﻿").lstrip("ï»¿") for c in raw.decode(enc).strip().split(",")]
        except UnicodeDecodeError:
            continue


# ------------------------------------------------ campos por corte (encabezado de cada zip)
zips = sorted(DEN.glob("denue_09_*_csv.zip"))
campos = {z.name.replace("denue_09_", "").replace("_csv.zip", ""): header(z) for z in zips}
base = set(campos["2016_01"])
print("=== Q6 campos ===")
for k, v in campos.items():
    print(k, len(v), "extra vs 2016_01:", sorted(set(v) - base), "faltan:", sorted(base - set(v)))

# ------------------------------------------------ verificación parquet == zip (conteo de filas)
for corte in ["2016_01", "2019_11", "2024_11"]:
    zf, name = csv_en_zip(DEN / f"denue_09_{corte}_csv.zip")
    with zf.open(name) as fh:
        n = sum(1 for _ in fh) - 1
    npq = pd.read_parquet(PQ / f"denue_09_{corte}.parquet", columns=["id"]).shape[0]
    print(f"verificación {corte}: líneas zip={n} (puede incluir saltos de línea embebidos), parquet={npq}")

# ------------------------------------------------ serie y calidad por corte
ent = lee_mg("09ent")
poly = ent.geometry.iloc[0]
filas = []
ids = {}
for f in sorted(PQ.glob("denue_09_*.parquet")):
    corte = f.stem.replace("denue_09_", "")
    d = pd.read_parquet(f)
    lat = pd.to_numeric(d.latitud, errors="coerce")
    lon = pd.to_numeric(d.longitud, errors="coerce")
    nn = lat.notna() & lon.notna()
    sub = d[nn].assign(latitud=lat[nn], longitud=lon[nn])
    inpoly = puntos(sub, crs=ent.crs).within(poly).sum()
    ids[corte] = d
    filas.append(dict(
        corte=corte, n=len(d), ids_unicos=d.id.nunique(),
        clee_presente="clee" in d and d.clee.notna().mean() > 0,
        pct_coord=round(nn.mean() * 100, 2), pct_en_cdmx=round(inpoly / len(d) * 100, 2),
        pct_ageb=round(d.ageb.notna().mean() * 100, 2),
        pct_per_ocu=round(d.per_ocu.notna().mean() * 100, 2),
        fecha_alta_ultima=d.fecha_alta.dropna().astype(str).str[:7].max(),
        version=d.version.dropna().iloc[0] if "version" in d and d.version.notna().any() else None,
    ))
serie = pd.DataFrame(filas)
serie["delta_%"] = (serie.n.pct_change() * 100).round(2)
print("\n=== Q5/Q6 serie DENUE ===\n", serie.to_string(index=False))
serie.to_csv(EDA / "q5_denue_serie.csv", index=False)

# valores de estrato y fecha_alta en un corte antiguo y uno reciente
for corte in ["2016_01", "2024_11"]:
    d = ids[corte]
    print(f"\n{corte} per_ocu:", d.per_ocu.value_counts().to_dict())
    print(f"{corte} fecha_alta (top):", d.fecha_alta.value_counts().head(8).to_dict())
    print(f"{corte} codigo_act len:", d.codigo_act.astype(str).str.len().value_counts().to_dict(),
          "ejemplo id/clee:", d[["id"] + (["clee"] if "clee" in d else [])].head(2).values.tolist())

# altas en 2019-11 -> 2020-11 por fecha_alta
d20 = ids["2020_11"]
print("\n2020_11 fecha_alta top:", d20.fecha_alta.value_counts().head(6).to_dict())

# ------------------------------------------------ estabilidad de ids
pares = [("2019_11", "2020_11"), ("2023_11", "2024_11"), ("2018_11", "2019_11"), ("2020_11", "2021_11"),
         ("2021_11", "2022_11"), ("2022_11", "2023_11"), ("2024_11", "2025_05")]
est = []
for a, b in pares:
    A, B = ids[a], ids[b]
    ia, ib = set(A.id), set(B.id)
    r = dict(par=f"{a}->{b}", n_a=len(ia), n_b=len(ib), ids_comunes=len(ia & ib),
             pct_a_que_sigue=round(len(ia & ib) / len(ia) * 100, 1), pct_b_que_existia=round(len(ia & ib) / len(ib) * 100, 1))
    if "clee" in A and "clee" in B and A.clee.notna().any() and B.clee.notna().any():
        ca, cb = set(A.clee.dropna()), set(B.clee.dropna())
        r["clee_pct_a_que_sigue"] = round(len(ca & cb) / len(ca) * 100, 1)
    # mismo id: ¿mismo SCIAN y misma ubicación?
    m = A[["id", "codigo_act", "latitud"]].merge(B[["id", "codigo_act", "latitud"]], on="id")
    r["mismo_scian_%"] = round((m.codigo_act_x == m.codigo_act_y).mean() * 100, 1)
    r["misma_lat_%"] = round((m.latitud_x.astype(str).str[:7] == m.latitud_y.astype(str).str[:7]).mean() * 100, 1)
    est.append(r)
est = pd.DataFrame(est)
print("\n=== Q6 estabilidad de id ===\n", est.to_string(index=False))
est.to_csv(EDA / "q6_denue_estabilidad_ids.csv", index=False)

# ------------------------------------------------ figura
fig, ax = plt.subplots(figsize=(8, 3.5))
x = pd.to_datetime(serie.corte.str.replace("_", "-") + "-01")
ax.plot(x, serie.n / 1000, marker="o")
for xi, yi, c in zip(x, serie.n / 1000, serie.corte):
    ax.annotate(c.replace("_", "-"), (xi, yi), fontsize=6, rotation=60, xytext=(0, 4), textcoords="offset points")
ax.set_ylabel("Establecimientos (miles)")
ax.set_title("DENUE CDMX: establecimientos por corte")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(FIGS / "denue_serie.png", dpi=130)
