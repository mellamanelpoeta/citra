"""Aplica mecánicamente las reglas ex ante de analisis/lab/README.md a los resultados y escribe resultados/decisiones_lab.md."""
import pandas as pd

from lab_comun import RES

SIMPLICIDAD = ["V1", "V2", "V2b"]


def regla_s1(s1):
    d = s1[(s1.lista == "b") & s1.variable.isin(SIMPLICIDAD) & (s1.max_pct_ceros_anual < 70)].set_index("variable")
    pct = d.pct_top10_autoserv_grande
    mejor = pct.idxmin()
    i = SIMPLICIDAD.index(mejor)
    while i > 0 and SIMPLICIDAD[i - 1] in pct and pct[SIMPLICIDAD[i - 1]] - pct[SIMPLICIDAD[i]] < 3:
        i -= 1
    return {"eleccion": SIMPLICIDAD[i], "minimo": mejor, "pct": pct.to_dict()}


def regla_lista(r):
    w = r.pivot(index="fold", columns="lista", values="dev_M1")
    media = w.mean()
    folds_b_mejor_c = int((w.b < w.c).sum())
    mejora_a = (media.b - media.a) / media.b
    a_c, a_a = folds_b_mejor_c < 3, mejora_a >= 0.01
    eleccion = "c" if a_c else ("a" if a_a else "b")
    return {"eleccion": eleccion, "media": media.to_dict(), "folds_b_mejor_c": folds_b_mejor_c,
            "mejora_a_sobre_b_pct": mejora_a * 100, "ambas_condiciones": a_c and a_a}


def regla_s2(s2):
    d = s2.set_index("e").dev_media_folds
    mejora = (d.fijo - d.nov) / d.fijo
    return {"eleccion": "nov" if mejora >= 0.02 else "fijo", "mejora_nov_pct": mejora * 100}


def regla_s4(s4):
    out = {}
    for _, f in s4.iterrows():
        excluir = (not f.beta_con_dentro_ic) or f.spearman_rankings < 0.90
        out[f.variable] = "excluir 2019" if excluir else "incluir 2019"
    return out


def tabla(df, fmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: fmt.format(x))
    return df.to_markdown(index=False)


if __name__ == "__main__":
    s1 = pd.read_csv(RES / "s1.csv")
    li = pd.read_csv(RES / "lista_giros.csv")
    s2 = pd.read_csv(RES / "s2.csv")
    s3 = pd.read_csv(RES / "s3.csv")
    s4 = pd.read_csv(RES / "s4.csv")
    d1, dl, d2, d4 = regla_s1(s1), regla_lista(li), regla_s2(s2), regla_s4(s4)

    rob = s1[s1.lista == "b"][["variable", "rob_beta_M0", "rob_beta_M0_ee", "rob_deltaD_pct", "rob_folds_mejora",
                               "rob_jaccard_conteo_vs_residual", "rob_persistencia_rho"]]
    s1t = s1.pivot(index="variable", columns="lista", values="pct_top10_autoserv_grande").reset_index()
    s1t.columns = ["variable", "% top 10% con autoserv. grande (lista a)", "% top 10% con autoserv. grande (lista b)"]
    s1t["máx. % ceros anual"] = s1[s1.lista == "b"].set_index("variable").max_pct_ceros_anual.reindex(s1t.variable).values
    lt = li.pivot(index="fold", columns="lista", values="dev_M1").reset_index()
    lt["dev_M0"] = li.drop_duplicates("fold").set_index("fold").dev_M0.reindex(lt.fold).values

    md = f"""# Decisiones del lab (generado por `analisis/lab/decidir.py`)

Reglas: `analisis/lab/README.md` (registradas antes de correr los experimentos). Valores por defecto: V2b, E fijo 2023-11, lista (b), 2020–2024.
Base: {s1.pct_base_autoserv_grande.iloc[0]:.1f}% de las AGEB de la muestra tiene un autoservicio grande.

## Resumen

| Decisión | Elección | Regla aplicada |
|---|---|---|
| S1 variable principal | **{d1['eleccion']}** | mínimo % del top 10% residual con autoservicio grande: {d1['minimo']}; desempate de < 3 pp hacia la más simple |
| Lista de giros | **({dl['eleccion']})** | (b) mejora a (c) en {dl['folds_b_mejor_c']}/5 folds (cambia a (c) si < 3); (a) mejora a (b) en {dl['mejora_a_sobre_b_pct']:.2f}% (cambia a (a) si ≥ 1%) |
| S2 corte DENUE | **{'(b) fijo 2023-11' if d2['eleccion'] == 'fijo' else '(a) noviembre de cada año'}** | E anual mejora la devianza en {d2['mejora_nov_pct']:.2f}% (cambia si ≥ 2%) |
| S3 validación temporal | **(a) E 2023-11** | Por validez: el corte 2024-11 no existe al momento de predecir 2024 |
| S4 2019 en V1 | **{d4.get('V1')}** | β con 2019 dentro del IC 95% sin 2019 y Spearman de rankings ≥ 0.90 |
| S4 2019 en V3 | **{d4.get('V3')}** | ídem |

## S1. Variable objetivo (métrica de decisión)

{tabla(s1t, "{:.1f}")}

### Robustez por variante (informativa; no entra en ninguna regla)

β de M0 (NB2, toda la muestra 2020–2024); ΔD = mejora relativa de la devianza fuera de muestra M0 → M1 (media de folds); Jaccard entre el top 10% por conteo y por P(θ > 1); persistencia = ρ medio del riesgo contraído entre años consecutivos.

{tabla(rob)}

## Lista de giros (variable {d1['eleccion']}): devianza fuera de muestra de M1 por fold

{tabla(lt, "{:.4f}")}

Media: (a) {dl['media']['a']:.4f} · (b) {dl['media']['b']:.4f} · (c) {dl['media']['c']:.4f}.{' **Ojo: se cumplen las dos condiciones de cambio; se aplicó (c) primero.**' if dl['ambas_condiciones'] else ''}

## S2. Corte del DENUE (validación espacial, 2020–2024)

{tabla(s2, "{:.4f}")}

## S3. Validación temporal (entrenar 2020–2023 → 2024)

{tabla(s3, "{:.4f}")}

## S4. 2019

{tabla(s4)}
"""
    (RES / "decisiones_lab.md").write_text(md)
    print(md)
