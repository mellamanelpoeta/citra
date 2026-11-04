"""08 · Hipótesis D2 de T6: ¿las zonas con más inseguridad previa rotan más? (entradas y salidas de negocios)

Protocolo y umbrales registrados antes de correr: analisis/lab/dinamica/README.md, sección "D2". Los umbrales no se mueven
después de ver resultados.

Salidas S_i: ids del DENUE 2019-11 en la AGEB i que no están en el DENUE 2024-11. Entradas En_i: ids del DENUE 2024-11
en la AGEB i que no estaban en 2019-11. Mismo predictor (rr_pre), controles, universo y bootstrap que D.

Entradas: DENUE 2019-11, 2023-11 y 2024-11 (parquet + nom_estab del CSV 2019-11), data/processed/panel_dinamica.parquet,
data/processed/modelos/dinamica_rr.parquet (07_dinamica_modelos.py).
Salidas: analisis/pipeline/outputs/dinamica/rotacion_*.csv, fig24_D2_forest.png; data/processed/modelos/rotacion_ageb.parquet.

Uso: .venv/bin/python analisis/pipeline/08_rotacion.py   (~20 min; cachea la asignación espacial de los ids)
"""
import zipfile

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (CANDIDATOS_EFECTIVO, EFECTIVO, INTERIM, PROCESSED, RAW, ROOT, SECTORES, asigna_ageb,  # noqa: E402
                    clasifica, lee_mg)
from modelado import MODELOS, REF_SECTOR, SEED, ajusta_nb, bloques  # noqa: E402

OUT = ROOT / "analisis/pipeline/outputs/dinamica"
B = 1000
UMBRAL = {"M2a_razon": 1.5, "D2a_ic_inf": 1.0, "D2b_ic_sup": 1.0}
# EDN 2023 (INEGI, nota técnica): muertes y nacimientos por cada 10 mil establecimientos al mes, mayo 2019 → mayo 2023,
# nacional, mipymes de manufactura, comercio y servicios privados no financieros
EDN = {"muertes": 71, "nacimientos": 61}
MESES = 60  # Censos Económicos 2019 (levantamiento feb–may 2019) → 2024 (feb–jul 2024)
FUERA_EDN = ("11", "21", "22", "52", "93")
GRUPOS = [*EFECTIVO, *SECTORES, "otros"]
COMP = [g for g in GRUPOS if g != REF_SECTOR]
TAMANO = {"0 a 5 personas": "t0_10", "6 a 10 personas": "t0_10", "11 a 30 personas": "t11_50", "31 a 50 personas": "t11_50",
          "51 a 100 personas": "t51_mas", "101 a 250 personas": "t51_mas", "251 y más personas": "t51_mas"}
TAMANOS = {"t0_10": "0 a 10 personas", "t11_50": "11 a 50", "t51_mas": "51 o más"}
TIPOS = {"cv": "robo a negocio con violencia", "sv": "robo a negocio sin violencia", "v3": "robo en vía pública (V3)"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
FILAS = []


def reg(seccion, metrica, valor, nota=""):
    FILAS.append({"seccion": seccion, "metrica": metrica, "valor": valor, "nota": nota})
    v = f"{valor:.4f}" if isinstance(valor, float) else valor
    print(f"  [{seccion}] {metrica}: {v}  {nota}")


# ============================================================ ids del DENUE con su AGEB
print("== ids DENUE")
urb = lee_mg("09a")[["CVEGEO", "geometry"]]
rur = lee_mg("09ar")[["CVEGEO", "geometry"]]


def ids(corte):
    cache = INTERIM / f"_eda/denue{corte}_11_ids_ageb.parquet"
    if not cache.exists():
        e = pd.read_parquet(RAW / f"denue/parquet/denue_09_{corte}_11.parquet",
                            columns=["id", "codigo_act", "per_ocu", "tipoUniEco", "latitud", "longitud"])
        e[["latitud", "longitud"]] = e[["latitud", "longitud"]].apply(pd.to_numeric, errors="coerce")
        e = e.dropna(subset=["latitud", "longitud"])
        e["cvegeo"] = asigna_ageb(e, urb, rur)
        e["codigo_act"] = e.codigo_act.astype(str).str.zfill(6)
        e.drop(columns=["latitud", "longitud"]).to_parquet(cache, index=False)
    e = pd.read_parquet(cache)
    e["id"] = e.id.astype(str)
    return e


den = {c: ids(c) for c in (2019, 2023, 2024)}
for c, e in den.items():
    e["cand"] = clasifica(e.codigo_act, CANDIDATOS_EFECTIVO, resto="")
    e["grupo"] = e.cand.map({k: g for g, cs in EFECTIVO.items() for k in cs}).fillna(clasifica(e.codigo_act, SECTORES))
    e["tamano"] = e.per_ocu.map(TAMANO)
    e["seis_mas"] = e.per_ocu != "0 a 5 personas"
    assert e.tamano.notna().all() and e.id.is_unique
d19, d23, d24 = den[2019], den[2023], den[2024]
d19["sale"] = ~d19.id.isin(set(d24.id))
d19["sale23"] = ~d19.id.isin(set(d23.id))
d24["entra"] = ~d24.id.isin(set(d19.id))
d23["entra23"] = ~d23.id.isin(set(d19.id))
# Movimientos: mismo id en otra AGEB (no son salidas ni entradas)
mov = d19[["id", "cvegeo"]].merge(d24[["id", "cvegeo"]], on="id", suffixes=("_19", "_24"))
reg("flujos", "ids 2019-11", len(d19))
reg("flujos", "tasa de salida 2019 → 2024 (ciudad)", float(d19.sale.mean()))
reg("flujos", "tasa de entrada 2019 → 2024 (/ N_2019, ciudad)", float(d24.entra.sum() / len(d19)))
reg("flujos", "tasa de salida 2019 → 2023 (entre censos)", float(d19.sale23.mean()))
reg("flujos", "tasa de entrada 2019 → 2023 (entre censos)", float(d23.entra23.sum() / len(d19)))
reg("flujos", "ids que siguen y cambian de AGEB", float((mov.cvegeo_19 != mov.cvegeo_24).mean()), "proporción de los que siguen")
for t, nom in TAMANOS.items():
    reg("flujos", f"tasa de salida 2019 → 2023 · {nom}", float(d19[d19.tamano == t].sale23.mean()))
    reg("flujos", f"tasa de salida 2019 → 2024 · {nom}", float(d19[d19.tamano == t].sale.mean()))

# Nombre (solo en el CSV): "SIN NOMBRE" como aproximación de difícil de censar
cache = INTERIM / "_eda/denue2019_11_sin_nombre.parquet"
if not cache.exists():
    z = zipfile.ZipFile(RAW / "denue/denue_09_2019_11_csv.zip")
    f = next(n for n in z.namelist() if n.endswith("denue_inegi_09_.csv"))
    nm = pd.read_csv(z.open(f), usecols=["id", "nom_estab"], dtype=str, encoding="latin-1")
    nm.assign(sin_nombre=nm.nom_estab.fillna("").str.upper().str.contains("SIN NOMBRE"))[["id", "sin_nombre"]].to_parquet(cache, index=False)
nm = pd.read_parquet(cache)
d19["sin_nombre"] = d19.id.map(nm.set_index("id").sin_nombre).fillna(False).astype(bool)
reg("flujos", "proporción 'SIN NOMBRE' 2019", float(d19.sin_nombre.mean()))

# ============================================================ M2a. Calibración con el EDN 2023
print("== M2a")


def comparable(e):
    return ~e.codigo_act.str[:2].isin(FUERA_EDN) & (e.per_ocu != "251 y más personas")


c19, c24 = d19[comparable(d19)], d24[comparable(d24)]
tasa_s = c19.sale.mean() / MESES * 1e4
tasa_e = c24.entra.sum() / len(c19) / MESES * 1e4
razon_s, razon_e = tasa_s / EDN["muertes"], tasa_e / EDN["nacimientos"]
M2a_OK = razon_s <= UMBRAL["M2a_razon"]
reg("M2a", "salidas DENUE por 10 mil al mes (universo comparable)", float(tasa_s), f"EDN: {EDN['muertes']}")
reg("M2a", "razón salidas DENUE / muertes EDN", float(razon_s), f"umbral ≤ {UMBRAL['M2a_razon']} → {'pasa' if M2a_OK else 'NO pasa'}")
reg("M2a", "entradas DENUE por 10 mil al mes (universo comparable)", float(tasa_e), f"EDN: {EDN['nacimientos']}")
reg("M2a", "razón entradas DENUE / nacimientos EDN (descriptiva)", float(razon_e))
reg("M2a", "salidas acumuladas a 49 meses con la tasa DENUE", float(tasa_s * 49 / 1e4),
    f"EDN a 49 meses: {EDN['muertes'] * 49 / 1e4:.3f}")

# ============================================================ Conteos por AGEB
print("== Conteos por AGEB")


def cuenta(e, mascara, por=None):
    x = e[mascara]
    if por is None:
        return x.groupby("cvegeo").size()
    return pd.crosstab(x.cvegeo, x[por])


pan = pd.read_parquet(PROCESSED / "panel_dinamica.parquet")
rr = pd.read_parquet(MODELOS / "dinamica_rr.parquet").set_index("cvegeo")
U = pan.set_index("cvegeo").loc[rr.index].reset_index()  # mismo universo y orden que D
blq = bloques()
idx = U.cvegeo
n19_ids = d19.groupby("cvegeo").size().reindex(idx).fillna(0)
assert (n19_ids.values == U.N_2019.values).all(), "los ids de 2019 no reproducen N_2019 del panel"
assert (d24.groupby("cvegeo").size().reindex(idx).fillna(0).values == U.N_2024.values).all()
F = pd.DataFrame(index=idx)
F["S"] = cuenta(d19, d19.sale)
F["En"] = cuenta(d24, d24.entra)
F["S23"] = cuenta(d19, d19.sale23)
F["En23"] = cuenta(d23, d23.entra23)
F["S6"] = cuenta(d19, d19.sale & d19.seis_mas)
F["En6"] = cuenta(d24, d24.entra & d24.seis_mas)
for g, c in cuenta(d19, d19.sale, "grupo").items():
    F[f"S_{g}"] = c
for g, c in cuenta(d24, d24.entra, "grupo").items():
    F[f"En_{g}"] = c
for t, c in cuenta(d19, d19.sale, "tamano").items():
    F[f"S_{t}"] = c
for t, c in cuenta(d24, d24.entra, "tamano").items():
    F[f"En_{t}"] = c
for t, c in cuenta(d19, d19.sale23, "tamano").items():
    F[f"S23_{t}"] = c
F = F.reindex(idx).fillna(0)
g19 = d19.groupby("cvegeo")
F["p_sin_nombre"] = g19.sin_nombre.mean()
F["p_semifijo"] = g19.tipoUniEco.apply(lambda s: (s == "Semifijo").mean())
F["p_micro"] = g19.per_ocu.apply(lambda s: (s == "0 a 5 personas").mean())
F["mov_sale"] = mov[mov.cvegeo_19 != mov.cvegeo_24].groupby("cvegeo_19").size()
F["mov_entra"] = mov[mov.cvegeo_19 != mov.cvegeo_24].groupby("cvegeo_24").size()
F[["mov_sale", "mov_entra"]] = F[["mov_sale", "mov_entra"]].fillna(0)
F = F.reset_index(drop=True)
# Identidad contable: N_2024 = N_2019 − S + En − movimientos que salen + movimientos que entran
assert (U.N_2019 - F.S + F.En - F.mov_sale + F.mov_entra == U.N_2024).all(), "la identidad de flujos no cuadra"
reg("datos", "AGEB en el universo", len(U))
reg("datos", "mediana de la tasa de salida por AGEB", float((F.S / U.N_2019).median()))
reg("datos", "mediana de la tasa de entrada por AGEB", float((F.En / U.N_2019).median()))
reg("datos", "Spearman tasa de salida vs tasa de entrada por AGEB",
    float(pd.Series(F.S / U.N_2019).corr(pd.Series(F.En / U.N_2019), method="spearman")))

# ============================================================ Modelos (idénticos a D en 07_dinamica_modelos.py)
ALC = pd.get_dummies(U.cve_mun, prefix="alc", drop_first=True).astype(float)
CTRL = ["logN19"] + [f"s_{g}" for g in COMP] + ["p_grandes", "log_pob", "no_residencial", "log_dist"] + list(ALC.columns)
COB = ["p_sin_nombre", "p_semifijo", "p_micro"]


def marco(y, logN19, rr_key="pre_v1", filtro=None):
    d = pd.DataFrame({"cvegeo": U.cvegeo, "y": np.asarray(y, dtype=float), "z": np.log(rr[rr_key].values),
                      "logN19": logN19, "p_grandes": U.p_grandes_2019, "log_pob": U.log_pob,
                      "no_residencial": U.no_residencial, "log_dist": np.log(U.dist_km_2019 + 0.05)})
    for g in COMP:
        d[f"s_{g}"] = U[f"s_{g}_2019"].fillna(0)
    d[COB] = F[COB].values
    d = pd.concat([d, ALC], axis=1)
    if filtro is not None:
        d = d[np.asarray(filtro)]
    d["z"] = (d.z - d.z.mean()) / d.z.std()
    return d.reset_index(drop=True)


def _ajusta_gamma(X, y):
    try:
        return ajusta_nb(X, y).params[1]
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def estima(d, nombre, flujo, cobertura=False):
    """IRR por +1 DE de log rr_pre con IC 95% (bootstrap de bloques espaciales) y p-valor bootstrap bilateral."""
    cols = ["z"] + CTRL + (COB if cobertura else [])
    X, y = d[cols].values.astype(float), d.y.values
    punto = float(ajusta_nb(X, y).params[1])
    grupos = {b_: g.index.values for b_, g in d.groupby(d.cvegeo.map(blq.bloque))}
    ids_b = np.array(list(grupos))
    rng = np.random.default_rng(SEED)
    muestras = [np.concatenate([grupos[b_] for b_ in rng.choice(ids_b, size=len(ids_b), replace=True)]) for _ in range(B)]
    bt = np.array(Parallel(n_jobs=-1)(delayed(_ajusta_gamma)(X[i], y[i]) for i in muestras))
    bt = bt[np.isfinite(bt)]
    assert len(bt) >= 0.95 * B, f"{nombre}: solo {len(bt)} réplicas válidas"
    lo, hi = np.exp(np.percentile(bt, [2.5, 97.5]))
    p = float(min(1.0, 2 * min((bt >= 0).mean(), (bt <= 0).mean())))
    if flujo == "salidas":
        ok = lo > UMBRAL["D2a_ic_inf"]
    elif flujo == "entradas":
        ok = hi < UMBRAL["D2b_ic_sup"]
    else:  # neto (reproducción de D)
        ok = hi < 1
    r = {"modelo": nombre, "flujo": flujo, "cobertura": cobertura, "n": len(d), "irr": np.exp(punto), "irr_lo": lo,
         "irr_hi": hi, "p_boot": p, "replicas": len(bt), "cumple": bool(ok)}
    print(f"  {nombre} [{flujo}{' + cobertura' if cobertura else ''}]: IRR = {r['irr']:.3f} [{lo:.3f}, {hi:.3f}], "
          f"p = {p:.3f}, n = {len(d)} → {'cumple' if ok else 'no cumple'}")
    return r


logN19 = np.log(U.N_2019)

print("== Reproducción de D (control de que el predictor y los controles son los mismos)")
Drep = estima(marco(U.N_2024, logN19), "D reproducida", "neto")
reg("control", "IRR de D reproducida", float(Drep["irr"]), f"[{Drep['irr_lo']:.3f}, {Drep['irr_hi']:.3f}]; 07 reporta 1.011 [1.002, 1.019]")

print("== D2a y D2b")
res = []
for flujo, col in [("salidas", "S"), ("entradas", "En")]:
    for cob in (False, True):
        res.append(estima(marco(F[col], logN19), f"Principal · {flujo}", flujo, cobertura=cob))
res = pd.DataFrame(res)
VER = {}
for flujo, h in [("salidas", "D2a"), ("entradas", "D2b")]:
    r = res[res.flujo == flujo]
    p_ok, c_ok = bool(r[~r.cobertura].cumple.iloc[0]), bool(r[r.cobertura].cumple.iloc[0])
    if not M2a_OK:
        VER[h] = "no medible (M2a)"
    elif p_ok and c_ok:
        VER[h] = "confirmada"
    elif p_ok:
        VER[h] = "no distinguible de un error de cobertura"
    else:
        VER[h] = "rechazada"
    for _, x in r.iterrows():
        reg(h, f"IRR {'con controles de cobertura' if x.cobertura else 'principal'}", float(x.irr),
            f"IC 95% [{x.irr_lo:.3f}, {x.irr_hi:.3f}], p = {x.p_boot:.3f}")
    reg(h, "veredicto", VER[h])
res.to_csv(OUT / "rotacion_D2.csv", index=False)

# ============================================================ Robustez fijada
print("== Robustez")
rob = []
rob.append(estima(marco(F.S23, logN19), "Bajas entre censos 2019 → 2023", "salidas"))
rob.append(estima(marco(F.En23, logN19), "Altas entre censos 2019 → 2023", "entradas"))
for t, nom in TAMANOS.items():
    n19 = U[f"n_{t}_2019"]
    rob.append(estima(marco(F[f"S23_{t}"], np.log(n19.clip(lower=1)), filtro=n19 > 0), f"Bajas entre censos · {nom}", "salidas"))
n6 = U.N6_2019
rob.append(estima(marco(F.S6, np.log(n6.clip(lower=1)), filtro=n6 > 0), "Solo 6+ personas", "salidas"))
rob.append(estima(marco(F.En6, np.log(n6.clip(lower=1)), filtro=n6 > 0), "Solo 6+ personas", "entradas"))
for flujo, col in [("salidas", "S"), ("entradas", "En")]:
    rob.append(estima(marco(F[col], logN19, filtro=U.N_2019 >= 5), "Excluye AGEB con N_2019 < 5", flujo))
    rob.append(estima(marco(F[col], logN19, rr_key="pre_v1_M0"), "rr_pre con M0", flujo))
    for t, nom in TIPOS.items():
        rob.append(estima(marco(F[col], logN19, rr_key=f"pre_{t}"), f"rr_pre de {nom}", flujo))
rob = pd.DataFrame(rob)
rob.to_csv(OUT / "rotacion_robustez.csv", index=False)
for flujo, h in [("salidas", "D2a"), ("entradas", "D2b")]:
    r = rob[rob.flujo == flujo]
    signo = np.sign(np.log(res[(res.flujo == flujo) & ~res.cobertura].irr.iloc[0]))
    reg(h, "robustez: misma dirección que el principal", float((np.sign(np.log(r.irr)) == signo).mean()), f"{len(r)} variantes")
    reg(h, "robustez: cumplen la regla", float(r.cumple.mean()))

# ============================================================ Heterogeneidad (Holm sobre toda la familia)
print("== Heterogeneidad")
het = []
for flujo, pre in [("salidas", "S"), ("entradas", "En")]:
    for g in GRUPOS:
        n19 = U[f"n_{g}_2019"]
        het.append({"familia": "giro", "subgrupo": g, **estima(marco(F[f"{pre}_{g}"], np.log(n19.clip(lower=1)), filtro=n19 > 0),
                                                                f"giro · {g}", flujo)})
    for t, nom in TAMANOS.items():
        n19 = U[f"n_{t}_2019"]
        het.append({"familia": "tamaño", "subgrupo": nom, **estima(marco(F[f"{pre}_{t}"], np.log(n19.clip(lower=1)), filtro=n19 > 0),
                                                                   f"tamaño · {nom}", flujo)})
het = pd.DataFrame(het)
het["p_holm"] = multipletests(het.p_boot, method="holm")[1]
het["significativo"] = het.p_holm < 0.05
het.to_csv(OUT / "rotacion_heterogeneidad.csv", index=False)
print(het[["flujo", "familia", "subgrupo", "n", "irr", "irr_lo", "irr_hi", "p_holm"]].round(3).to_string())
reg("heterogeneidad", "subgrupos significativos tras Holm", int(het.significativo.sum()), f"de {len(het)}")
H3 = "no se corre (regla: solo si D2a o D2b se confirma)" if "confirmada" not in VER.values() else "PENDIENTE: la regla pide H3"
reg("heterogeneidad", "malla H3", 0, H3)

# ============================================================ Figura
FIL = {}
for flujo in ["salidas", "entradas"]:
    f = [(f"Principal{' + cobertura' if x.cobertura else ''}", x.irr, x.irr_lo, x.irr_hi, "#7f3c8d")
         for x in res[res.flujo == flujo].itertuples()]
    f += [(f"{'Giro' if x.familia == 'giro' else 'Tamaño'} · {x.subgrupo}", x.irr, x.irr_lo, x.irr_hi,
           "#e74c3c" if x.significativo else "#3498db") for x in het[het.flujo == flujo].itertuples()]
    f += [(x.modelo, x.irr, x.irr_lo, x.irr_hi, "#999999") for x in rob[rob.flujo == flujo].itertuples()]
    FIL[flujo] = f
fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.2))
for ax, flujo, h in zip(axes, ["salidas", "entradas"], ["D2a", "D2b"]):
    filas = FIL[flujo][::-1]
    for i, (nom, x, lo, hi, c) in enumerate(filas):
        ax.errorbar(x, i, xerr=[[x - lo], [hi - x]], fmt="o", color=c, ms=4, capsize=2)
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels([f[0] for f in filas], fontsize=7)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_title(f"{h} · {flujo}: {VER[h]}", fontsize=9, loc="left")
    ax.set_xlabel("Razón de tasas por +1 DE de inseguridad previa (IC 95%)")
fig.suptitle("D2: ¿las zonas con más robo previo rotan más? (DENUE 2019-11 → 2024-11)", fontsize=9, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig24_D2_forest.png", bbox_inches="tight")
plt.close(fig)

# ============================================================ Salidas
out = pd.concat([U[["cvegeo", "alcaldia", "colonia", "N_2019", "N_2024"]], F[["S", "En", "S23", "En23", "mov_sale", "mov_entra"] + COB]], axis=1)
out["log_rr_pre"] = np.log(rr.pre_v1.values)
out.to_parquet(MODELOS / "rotacion_ageb.parquet", index=False)
pd.DataFrame(FILAS).to_csv(OUT / "rotacion_metricas.csv", index=False)
print("listo:", VER)
