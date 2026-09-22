"""Selección de variables SIN leakage y preprocesamiento.

Regla de oro del proyecto: solo se usan variables conocidas en el momento de
la decisión crediticia (originación del préstamo). Todo lo que se genera
DESPUÉS de otorgar el crédito (pagos, recuperaciones, FICO actualizado,
acuerdos de liquidación, hardship, etc.) es LEAKAGE y queda prohibido.

Usamos un enfoque de *lista blanca* (allowlist): es más seguro que una lista
negra, porque si Lending Club agrega una columna nueva en el futuro, por
defecto NO entra al modelo hasta revisarla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Variables disponibles en el momento de la decisión (originación)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "loan_amnt",          # monto solicitado
    "int_rate",           # tasa asignada por LC al originar (disponible)
    "installment",        # cuota (derivada de monto, plazo y tasa)
    "annual_inc",         # ingreso anual declarado
    "dti",                # deuda/ingreso al momento de solicitar
    "delinq_2yrs",        # morosidades en los últimos 2 años
    "fico_range_low",     # FICO al originar (NO last_fico_*, que es posterior)
    "fico_range_high",
    "inq_last_6mths",     # consultas de crédito últimos 6 meses
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",           # cuentas de crédito abiertas
    "pub_rec",            # registros públicos negativos
    "revol_bal",          # saldo revolvente
    "revol_util",         # utilización revolvente (%)
    "total_acc",          # total de cuentas de crédito
    "pub_rec_bankruptcies",
    "emp_length_years",   # derivada (ver clean_columns)
    "credit_history_yrs", # derivada de earliest_cr_line
]

CATEGORICAL_FEATURES = [
    "term",               # 36 / 60 meses
    "grade",              # grado de riesgo LC (asignado al originar)
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
    "initial_list_status",
]

# ---------------------------------------------------------------------------
# Columnas PROHIBIDAS por leakage (documentadas para el informe).
# Se conocen solo DESPUÉS de la decisión / durante la vida del préstamo.
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = [
    # Fondeo posterior a la aprobación
    "funded_amnt", "funded_amnt_inv",
    # Pagos y saldos vivos
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    # Información de crédito actualizada durante la vida del préstamo
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    # Acuerdos de liquidación / hardship (posteriores al default)
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "hardship_start_date", "hardship_end_date", "hardship_amount",
    "payment_plan_start_date", "pymnt_plan",
    # El propio estado / fecha de emisión (target y variable de partición)
    "loan_status", "issue_d",
]


def _pct_to_float(s: pd.Series) -> pd.Series:
    """Convierte '13.56%' -> 13.56 (float).

    Robusto a object y al nuevo dtype `str` de pandas>=3.0: si la columna no es
    numérica, se limpia como texto antes de convertir.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas crudas y crea las variables derivadas seguras.

    No hace imputación ni escalado (eso va en el pipeline, ajustado solo con
    train para evitar leakage estadístico).
    """
    df = df.copy()

    # Porcentajes almacenados como texto
    for col in ("int_rate", "revol_util"):
        if col in df.columns:
            df[col] = _pct_to_float(df[col])

    # emp_length: '10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3
    if "emp_length" in df.columns:
        emp = df["emp_length"].astype(str)
        emp = emp.str.replace("years", "", regex=False).str.replace("year", "", regex=False)
        emp = emp.str.replace("+", "", regex=False).str.replace("<", "", regex=False)
        df["emp_length_years"] = pd.to_numeric(emp.str.strip(), errors="coerce")

    # term: ' 36 months' -> '36'
    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.replace("months", "", regex=False).str.strip()

    # Antigüedad crediticia = issue_d - earliest_cr_line (en años).
    # Usa issue_d SOLO para derivar antigüedad al originar (no como feature).
    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        iss = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
        df["credit_history_yrs"] = (iss - ecl).dt.days / 365.25

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las columnas de la lista blanca presentes en el df."""
    cols = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if c in df.columns]
    return df[cols].copy()


def present_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Listas de features numéricas y categóricas realmente presentes."""
    num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num, cat


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """ColumnTransformer: imputación + escalado (num) y imputación + OHE (cat).

    Se ajusta dentro del pipeline SOLO con datos de entrenamiento.
    """
    num, cat = present_features(df)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, num),
        ("cat", categorical_pipe, cat),
    ], remainder="drop")
