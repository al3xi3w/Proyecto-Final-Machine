"""Entrenamiento del baseline con partición temporal (out-of-time).

Decisión de diseño clave: en scoring crediticio la partición aleatoria induce
leakage temporal (el modelo "ve el futuro"). Partimos por fecha de emisión
(issue_d): entrenamos con los préstamos más antiguos y validamos con los más
recientes, replicando cómo se usaría el modelo en producción.

Uso:
    python -m src.train --data data/accepted_2007_to_2018Q4.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .data import load_dataset
from .evaluate import compute_metrics
from .features import build_preprocessor, select_features


def temporal_split(df: pd.DataFrame, valid_frac: float = 0.2):
    """Divide por issue_d: los más recientes van a validación (out-of-time)."""
    iss = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.assign(_issue=iss).sort_values("_issue")
    cutoff = df["_issue"].quantile(1 - valid_frac)
    train = df[df["_issue"] <= cutoff]
    valid = df[df["_issue"] > cutoff]
    return train.drop(columns="_issue"), valid.drop(columns="_issue")


def build_baseline_pipeline(df: pd.DataFrame) -> Pipeline:
    """Regresión logística baseline SIN balanceo ni tuning (proposal §11).

    El manejo del desbalance (class_weight / resampling) se deja para la fase de
    modelado (03_modelos.ipynb); aquí el baseline debe ser lo más simple posible.
    """
    pre = build_preprocessor(df)
    clf = LogisticRegression(max_iter=1000, random_state=config.SEED)
    return Pipeline(steps=[("prep", pre), ("clf", clf)])


def main(data_path: str | None = None, valid_frac: float = 0.2,
         nrows: int | None = None, frac: float | None = None):
    np.random.seed(config.SEED)

    df = load_dataset(data_path, nrows=nrows, frac=frac)
    train_df, valid_df = temporal_split(df, valid_frac=valid_frac)

    X_train, y_train = select_features(train_df), train_df[config.TARGET_BINARY]
    X_valid, y_valid = select_features(valid_df), valid_df[config.TARGET_BINARY]

    results = {}

    # --- Baseline 0: clase mayoritaria (referencia trivial) ----------------
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    results["dummy_prior"] = compute_metrics(
        y_valid, dummy.predict_proba(X_valid)[:, 1]
    )

    # --- Baseline 1: regresión logística -----------------------------------
    pipe = build_baseline_pipeline(train_df)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_valid)[:, 1]
    results["logistic_regression"] = compute_metrics(y_valid, proba)

    summary = {
        "seed": config.SEED,
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "valid_frac": valid_frac,
        "default_rate_train": float(y_train.mean()),
        "default_rate_valid": float(y_valid.mean()),
        "metrics": results,
    }

    out = config.OUTPUTS_DIR / "metrics.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Guardado {out}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Ruta al CSV/parquet de Lending Club")
    ap.add_argument("--valid-frac", type=float, default=0.2)
    ap.add_argument("--nrows", type=int, default=None, help="Limitar filas (debug)")
    ap.add_argument("--frac", type=float, default=None,
                    help="Muestra aleatoria reproducible (0-1), p. ej. 0.4")
    args = ap.parse_args()
    main(args.data, valid_frac=args.valid_frac, nrows=args.nrows, frac=args.frac)
