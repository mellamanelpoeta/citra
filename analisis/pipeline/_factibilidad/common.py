"""Utilidades compartidas del EDA de factibilidad (homologación FGJ, rutas, geometrías)."""
from pathlib import Path
import unicodedata

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/_samples"
EDA = ROOT / "data/interim/_eda"
FIGS = ROOT / "analisis/pipeline/outputs/factibilidad/figs"
MG = RAW / "mg2020/conjunto_de_datos"
EDA.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

PATRON = "NEGOCIO|TRANSE|REPARTIDOR|CUENTAHABIENTE|EXTORSI|COBRO DE PISO"


def norm(s):
    """Mayúsculas, sin acentos, espacios colapsados."""
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


def grupo_delito(d):
    """Homologación tentativa delito (normalizado) -> (grupo, objetivo, nota)."""
    if not isinstance(d, str):
        return ("otro", "excluir", "")
    if "EXTORSI" in d or "SECUESTRO" in d:
        return ("extorsion", "excluir", "fuera de alcance (cifra negra)")
    if d.startswith("DANO EN PROPIEDAD"):
        return ("dano_negocio", "excluir", "no es robo")
    if "OFICINA PUBLICA" in d:
        return ("negocio_cv_otros", "excluir", "oficina pública: no es negocio (está en la categoría oficial de robo a negocio c/v)")
    if "TAXI" in d:
        return ("transeunte_taxi", "excluir", "robo en/ a taxi: no es vía pública")
    if "NOMINA" in d or "SUCURSAL BANCARIA" in d or "TRASLADO DE VALORES" in d:
        if "TRANSEUNTE" in d:
            return ("transeunte_via_publica", "via_publica", "nómina en vía pública; marginal")
        return ("negocio_cv_otros", "negocio", "nómina/banco/valores: marginal (<60/año)")
    if d.startswith("ROBO S/V DENTRO DE NEGOCIOS"):
        return ("negocio_sv", "negocio", "aparece en 2020; revisar si la víctima es el negocio o un cliente")
    if d.startswith("ROBO A NEGOCIO"):
        cv = "CON VIOLENCIA" in d
        far = "FARDEROS" in d
        g = ("negocio_cv" if cv else "negocio_sv") + ("_farderos" if far else "")
        nota = "subtipo creado en 2020 (antes dentro de robo a negocio genérico)" if far else ""
        return (g, "negocio", nota)
    if d.startswith("ROBO A REPARTIDOR"):
        return ("repartidor", "via_publica", "")
    if "SALIENDO DEL BANCO" in d or "SALIENDO DEL CAJERO" in d:
        return ("cuentahabiente", "via_publica", "categoría oficial: robo a cuentahabiente")
    if d.startswith("ROBO A TRANSEUNTE EN VIA PUBLICA") or d.startswith("ROBO A TRANSEUNTE DE CELULAR"):
        nota = "celular: absorbido en vía pública desde 2019-2020" if "CELULAR" in d else ""
        return ("transeunte_via_publica", "via_publica", nota)
    if d.startswith("ROBO A TRANSEUNTE EN"):
        return ("transeunte_en_lugar", "ampliado", "transeúnte en negocio/restaurante/hotel/parques y mercados: candidato a variante ampliada")
    if d.startswith("ROBO A TRANSEUNTE Y VEHICULO"):
        return ("transeunte_via_publica", "via_publica", "marginal")
    return ("otro", "excluir", "")


def agrega_grupos(df, col="delito"):
    dn = df[col].map(norm)
    m = pd.DataFrame(dn.drop_duplicates())
    m[["grupo", "objetivo", "nota"]] = m[col].apply(lambda d: pd.Series(grupo_delito(d)))
    m = m.rename(columns={col: "delito_norm"})
    out = df.copy()
    out["delito_norm"] = dn
    return out.merge(m, on="delito_norm", how="left")


def lee_mg(capa):
    return gpd.read_file(MG / f"{capa}.shp")


def puntos(df, lat="latitud", lon="longitud", crs=None):
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
    return g.to_crs(crs) if crs is not None else g
