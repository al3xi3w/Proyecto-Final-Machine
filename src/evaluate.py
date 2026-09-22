"""Métricas de evaluación para clasificación desbalanceada (riesgo crediticio).

NO se reporta accuracy como métrica principal: con ~20% de defaults, un modelo
que predice "siempre paga" tendría ~80% de accuracy y sería inútil. Se prioriza
ROC-AUC, PR-AUC (average precision) y el estadístico KS, estándar en scoring
crediticio.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: máxima separación entre las CDF de buenos y malos.

    Muy usado en credit scoring. KS = max(TPR - FPR) sobre todos los umbrales.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_true = y_true[order]
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tpr = np.cumsum(y_true) / pos
    fpr = np.cumsum(1 - y_true) / neg
    return float(np.max(np.abs(tpr - fpr)))


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Devuelve un dict de métricas a partir de probabilidades predichas."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "ks": ks_statistic(y_true, y_proba),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }
