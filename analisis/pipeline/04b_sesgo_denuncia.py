"""04b · ¿Hacia dónde empuja el sesgo de denuncia la estimación de β (hipótesis A)? Análisis exploratorio
posterior a T2b: no modifica veredictos, umbrales ni salidas de 04_modelos.py.

Marco: lo observado es O = r · C (r = tasa de denuncia, C = robo real). Si r ∝ E^γ, entonces β_obs = β_real + γ.
La propensión a denunciar no se observa por AGEB, pero se pueden probar sus canales plausibles:
1. Tamaño de empresa (ENVE 2024, CDMX): r_i se aproxima con la mezcla de tamaños del DENUE y entra como offset.
2. Diferencias entre alcaldías (fiscalías, acceso al MP; lo más fino que daría una encuesta como la ENSU):
   efectos fijos de alcaldía.
3. Tipo de robo: el robo con violencia se denuncia más (ENVE: cifra negra de robo/asalto 82% frente a 94% de
   robo hormiga), así que su β está menos contaminado por la cifra negra.
4. Punto de quiebre: qué γ haría falta para que el IC 95% de β alcanzara 1.

Uso: .venv/bin/python analisis/pipeline/04b_sesgo_denuncia.py   (~2 min)
Salidas: analisis/pipeline/outputs/modelos/sesgo_denuncia.csv, sesgo_denuncia_quintiles.csv, fig15_sesgo_denuncia.png
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import CORTE_FIJO, INTERIM  # noqa: E402
from modelado import OUT_MOD, SEED, bloques, cols_m0, datos  # noqa: E402

B = 1000
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ENVE 2024 (año 2023), tabulado t3_8, Ciudad de México: cifra negra (%) por tamaño de la unidad económica.
CIFRA_NEGRA_CDMX = {"micro": 88.78, "pequena": 87.64, "mediana": 83.09, "grande": 72.76}
# Supuesto: los estratos de personal ocupado del DENUE se asignan a los tamaños de la ENVE con el corte de
# industria (≤10, 11–50, 51–250, >250). La ENVE usa cortes distintos por sector; esto es una aproximación.
TAMANO = {"0 a 5 personas": "micro", "6 a 10 personas": "micro", "11 a 30 personas": "pequena",
          "31 a 50 personas": "pequena", "51 a 100 personas": "mediana", "101 a 250 personas": "mediana",
          "251 y más personas": "grande"}


# ============================================================ NB2 con offset y bootstrap de bloques
def ajusta(X, y, off=None):
    warnings.filterwarnings("ignore")  # los procesos de joblib no heredan el filtro de modelado.py
    Xc = sm.add_constant(X, has_constant="add")
    start = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=off).fit().params
    m = sm.NegativeBinomial(y, Xc, loglike_method="nb2", offset=off)
    try:
        r = m.fit(start_params=np.append(start, 0.5), method="newton", maxiter=200, disp=0)
        if r.mle_retvals.get("converged", True) and np.isfinite(r.params).all():
            return r
    except np.linalg.LinAlgError:
        pass
    return m.fit(start_params=np.append(start, 0.5), method="bfgs", maxiter=2000, disp=0)


def beta_boot(p, cols, off=None):
    """β de logE (índice 1, tras la constante) y su IC 95% percentil por bootstrap de bloques espaciales
    (mismo esquema que modelado.bootstrap_beta: se remuestrean bloques completos con todas sus unidades y años)."""
    X, y = p[cols].values.astype(float), p.y.values
    off = None if off is None else np.asarray(off, float)
    grupos = {b_: g.index.values for b_, g in p.groupby(p.cvegeo.map(bloques().bloque))}
    ids = np.array(list(grupos))
    rng = np.random.default_rng(SEED)
    muestras = [np.concatenate([grupos[b_] for b_ in rng.choice(ids, size=len(ids), replace=True)]) for _ in range(B)]

    def uno(i):
        return ajusta(X[i], y[i], None if off is None else off[i]).params[1]

    out = np.array(Parallel(n_jobs=-1)(delayed(uno)(i) for i in muestras))
    out = out[np.isfinite(out)]
    return ajusta(X, y, off).params[1], out


# ============================================================ canales
def tasa_denuncia_tamano():
    """r_i = Σ_k s_ik · (1 − cifra negra_k): tasa de denuncia esperada de la AGEB según su mezcla de tamaños.
    Supuesto: cada establecimiento tiene la misma probabilidad de ser víctima sin importar su tamaño."""
    e = pd.read_parquet(INTERIM / "denue_ageb.parquet")
    e = e[e.corte == CORTE_FIJO]
    s = pd.crosstab(e.cvegeo, e.per_ocu.map(TAMANO), normalize="index")
    return s.mul(pd.Series({k: 1 - v / 100 for k, v in CIFRA_NEGRA_CDMX.items()})).sum(axis=1)


def tamano_por_quintil(p, r, gamma_quiebre):
    """Mezcla de tamaños por quintil de E: ¿las grandes son una proporción mayor donde hay más negocios?
    La tasa de denuncia depende de la proporción, no del número (más negocios trae más grandes en número).
    denuncia_necesaria: la que haría falta para rechazar A, r ∝ E^γ con γ del punto de quiebre, anclada en el quintil
    central. piso_micro: la denuncia de una zona solo con micronegocios (la mínima que permite la mezcla de tamaños)."""
    e = pd.read_parquet(INTERIM / "denue_ageb.parquet")
    e = e[e.corte == CORTE_FIJO]
    s = pd.crosstab(e.cvegeo, e.per_ocu.map(TAMANO))
    u = p.drop_duplicates("cvegeo").set_index("cvegeo")[["E"]].join(s).join(r.rename("r"))
    u["mas_de_50"] = u.mediana + u.grande
    u["q"] = pd.qcut(u.E.rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    g = u.groupby("q", observed=True)
    t = pd.DataFrame({"agebs": g.size(), "negocios_mediana": g.E.median(),
                      "pct_mas_de_50": g.mas_de_50.sum() / g.E.sum() * 100,
                      "pct_micro": g.micro.sum() / g.E.sum() * 100,
                      "pct_de_las_mas_de_50": g.mas_de_50.sum() / u.mas_de_50.sum() * 100,
                      "pct_de_los_negocios": g.E.sum() / u.E.sum() * 100,
                      "denuncia_esperada": g.r.mean() * 100})
    ancla = t.loc["Q3"]
    t["denuncia_necesaria"] = ancla.denuncia_esperada * (t.negocios_mediana / ancla.negocios_mediana) ** gamma_quiebre
    t["piso_micro"] = 100 - CIFRA_NEGRA_CDMX["micro"]
    t.to_csv(OUT_MOD / "sesgo_denuncia_quintiles.csv", float_format="%.2f")
    print(t.round(2).to_string())
    return t


def panel_por_tipo(p):
    """Divide V2b en robo con violencia (negocio_cv, negocio_cv_otros) y sin violencia (negocio_sv)."""
    d = pd.read_parquet(INTERIM / "delitos_ageb.parquet")
    d = d[(d.tipo_asig == "URBANA") & d.anio_hecho.isin(p.anio.unique())]
    tipos = {"cv": d.grupo.isin(["negocio_cv", "negocio_cv_otros"]), "sv": d.grupo == "negocio_sv"}
    out = {}
    for k, m in tipos.items():
        n = d[m].groupby(["cvegeo", "anio_hecho"]).size()
        q = p.copy()
        q["y"] = pd.MultiIndex.from_arrays([q.cvegeo, q.anio]).map(n).fillna(0).values.astype(float)
        out[k] = q
    assert (out["cv"].y + out["sv"].y == p.y).all(), "cv + sv no reproduce V2b"
    return out


def main():
    p = datos("V2b")
    c0 = cols_m0(p)
    filas = []

    def registra(nombre, canal, b, boot, nota=""):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        filas.append({"escenario": nombre, "canal": canal, "beta": b, "ic_inf": lo, "ic_sup": hi,
                      "A": "confirmada" if hi < 1 else "rechazada", "nota": nota})
        print(f"{nombre:<48} β = {b:.3f} [{lo:.3f}, {hi:.3f}]  {nota}")

    # 0. referencia (debe reproducir 04_modelos: 0.877 [0.808, 0.952])
    b, boot = beta_boot(p, c0)
    registra("M0 · V2b (principal)", "referencia", b, boot)
    hi0 = np.percentile(boot, 97.5)

    # 1. tamaño de empresa (ENVE)
    r = tasa_denuncia_tamano()
    tamano_por_quintil(p, r, hi0 - 1)
    p["log_r"] = np.log(p.cvegeo.map(r).values)
    u = p.drop_duplicates("cvegeo")
    gamma = np.polyfit(u.logE, u.log_r, 1)[0]
    b, boot = beta_boot(p, c0, off=p.log_r)
    registra("Corrigiendo la denuncia por tamaño (ENVE)", "tamaño", b, boot,
             f"γ_tamaño = {gamma:+.4f}: la denuncia {'baja' if gamma < 0 else 'sube'} "
             f"{abs(1 - 2 ** gamma):.1%} por cada duplicación de E")

    # 2. alcaldía
    alc = pd.get_dummies(p.cvegeo.str[2:5], prefix="alc", drop_first=True).astype(float)
    q = pd.concat([p, alc], axis=1)
    b, boot = beta_boot(q, c0 + list(alc.columns))
    registra("Efectos fijos de alcaldía", "alcaldía", b, boot)

    # 3. tipo de robo
    for k, nombre in [("cv", "Solo robo con violencia (más denunciado)"), ("sv", "Solo robo sin violencia")]:
        q = panel_por_tipo(p)[k]
        b, boot = beta_boot(q, c0)
        registra(nombre, "tipo de robo", b, boot, f"{int(q.y.sum()):,} carpetas")

    # 4. punto de quiebre: β_real = β_obs − γ; A se rechazaría si ic_sup − γ ≥ 1
    g_ic, g_pt = hi0 - 1, filas[0]["beta"] - 1
    quiebre = (f"Para que A se rechace, la denuncia tendría que caer ≥ {1 - 2 ** g_ic:.1%} por cada duplicación de E "
               f"(γ ≤ {g_ic:+.3f}); para borrar la sublinealidad puntual, {1 - 2 ** g_pt:.1%} (γ = {g_pt:+.3f}).")
    print("\n" + quiebre)

    t = pd.DataFrame(filas)
    t.to_csv(OUT_MOD / "sesgo_denuncia.csv", index=False, float_format="%.4f")
    (OUT_MOD / "sesgo_denuncia.txt").write_text(quiebre + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    yy = np.arange(len(t))[::-1]
    ax.errorbar(t.beta, yy, xerr=[t.beta - t.ic_inf, t.ic_sup - t.beta], fmt="o", color="#1f4e79", capsize=3)
    ax.axvline(1, color="#b22222", ls="--", lw=1)
    ax.text(0.995, yy.max() + 0.35, "proporcional (β = 1)", color="#b22222", fontsize=7, ha="right")
    ax.set_yticks(yy, t.escenario)
    ax.set_ylim(-0.6, yy.max() + 0.7)
    ax.set_xlabel("β (IC 95%, bootstrap de bloques espaciales)")
    fig.suptitle("Hipótesis A bajo los canales del sesgo de denuncia", x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_MOD / "fig15_sesgo_denuncia.png")


if __name__ == "__main__":
    main()
