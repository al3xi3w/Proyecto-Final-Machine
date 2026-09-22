"""Carga de datos y construcción de la variable objetivo (Lending Club).

El archivo crudo de Lending Club trae ~150 columnas y, según la fuente, a
veces incluye filas de encabezado/pie no tabulares. Aquí centralizamos la
carga robusta, el filtrado de préstamos con resultado observado y la creación
del target binario.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# Columnas crudas necesarias para modelado (allowlist + target + derivación).
# Cargar solo estas hace la lectura del archivo completo (~2.2M filas) segura en
# memoria sin renunciar a usar todas las filas.
RAW_USECOLS = [
    # target y partición
    "loan_status", "issue_d",
    # fuentes de derivadas
    "emp_length", "earliest_cr_line",
    # numéricas de originación
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "pub_rec_bankruptcies",
    # categóricas de originación
    "term", "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "addr_state", "application_type", "initial_list_status",
]


def _resolve_path(path):
    path = Path(path) if path is not None else config.DATA_DIR / config.RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en {path}.\n"
            "Coloca el CSV de Lending Club en data/ o pasa la ruta explícita. "
            "Ver data/README.md."
        )
    return path


def load_raw(path: str | Path | None = None, nrows: int | None = None,
             usecols: list[str] | None = None, frac: float | None = None) -> pd.DataFrame:
    """Carga el CSV crudo de Lending Club de forma robusta.

    - Soporta .csv y .parquet.
    - `usecols` limita las columnas (clave para no agotar memoria con el archivo
      completo). low_memory=False evita inferencias de tipo inconsistentes.
    - `frac` (0-1): muestra aleatoria reproducible leyendo por chunks, para no
      cargar el archivo entero en memoria. Usa la semilla global.
    """
    path = _resolve_path(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=usecols)
        if frac is not None:
            df = df.sample(frac=frac, random_state=config.SEED)
        if nrows is not None:
            df = df.head(nrows)
    elif frac is not None:
        # Muestreo aleatorio por chunks (memoria-segura y reproducible)
        parts = []
        for chunk in pd.read_csv(path, low_memory=False, usecols=usecols,
                                 chunksize=200_000):
            parts.append(chunk.sample(frac=frac, random_state=config.SEED))
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False, nrows=nrows, usecols=usecols)

    # Elimina filas basura sin loan_amnt (encabezados/pies no tabulares)
    if "loan_amnt" in df.columns:
        df = df[df["loan_amnt"].notna()].copy()
    return df


def stream_missingness(path: str | Path | None = None,
                       chunksize: int = 200_000) -> "pd.Series":
    """% de faltantes por columna sobre TODAS las columnas, leyendo en chunks.

    Permite reportar faltantes del archivo completo (151 columnas) sin cargarlo
    entero en memoria. Devuelve una Serie (0-100) ordenada desc.
    """
    path = _resolve_path(path)
    na_counts = None
    total = 0
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
        chunk = chunk[chunk["loan_amnt"].notna()] if "loan_amnt" in chunk else chunk
        c = chunk.isna().sum()
        na_counts = c if na_counts is None else na_counts.add(c, fill_value=0)
        total += len(chunk)
    return (na_counts / total * 100).sort_values(ascending=False)


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Crea `target_default` (1 = mal desempeño) y descarta resultados no observados.

    Solo se conservan préstamos ya resueltos (Fully Paid / Charged Off /
    Default / etc.). Los préstamos 'Current' o similares se excluyen porque su
    resultado todavía no es observable y entrenarlos sería un error de etiqueta.
    """
    df = df.copy()
    status = df[config.TARGET_COL].astype(str).str.strip()

    is_bad = status.isin(config.BAD_STATUSES)
    is_good = status.isin(config.GOOD_STATUSES)

    keep = is_bad | is_good
    df = df.loc[keep].copy()
    df[config.TARGET_BINARY] = is_bad.loc[keep].astype(int).values
    return df


def load_dataset(path: str | Path | None = None, nrows: int | None = None,
                 usecols: list[str] | None = "default",
                 frac: float | None = None) -> pd.DataFrame:
    """Pipeline de carga: crudo -> limpieza de columnas -> target.

    Por defecto carga solo `RAW_USECOLS` (seguro en memoria). `frac` toma una
    muestra aleatoria reproducible. Pasa `usecols=None` para todas las columnas.
    """
    from .features import clean_columns  # import local para evitar ciclos

    if usecols == "default":
        usecols = RAW_USECOLS
    df = load_raw(path, nrows=nrows, usecols=usecols, frac=frac)
    df = clean_columns(df)
    df = make_target(df)
    return df
