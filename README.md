# Predicción de incumplimiento de préstamos — Lending Club

Proyecto final del curso de Machine Learning. **Entrega previa.**

Clasificación binaria del riesgo de **incumplimiento (default)** de préstamos
personales de Lending Club, usando **únicamente** variables disponibles en el
momento de la decisión crediticia (control estricto de *leakage*).

- **Problema:** clasificación supervisada, `target_default` ∈ {0,1}.
- **Métrica principal:** ROC-AUC. **Secundarias:** PR-AUC y KS.
- **Validación:** partición temporal (out-of-time) por `issue_d`.
- **Baseline:** clase mayoritaria + regresión logística con `class_weight`.

Ver `proposal.md` para la propuesta completa (13 puntos).

## Estructura

```
proyecto-final/
├── README.md
├── proposal.md
├── requirements.txt
├── data/                     # datos (no versionados) + instrucciones
│   └── README.md
├── notebooks/
│   └── 01_exploracion_inicial.ipynb
├── src/
│   ├── config.py             # rutas, semilla, definición del target
│   ├── data.py               # carga y construcción del target
│   ├── features.py           # lista blanca sin leakage + preprocesamiento
│   ├── train.py              # partición temporal + baselines
│   └── evaluate.py           # métricas (ROC-AUC, PR-AUC, KS, ...)
├── reports/figures/          # figuras generadas por el notebook
└── outputs/
    └── metrics.json          # métricas del baseline (generado)
```

## Reproducir la exploración

Requiere Python 3.11.

```bash
# 1. Entorno
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt

# 2. Datos: coloca el CSV de Lending Club en data/ (ver data/README.md)
#    data/accepted_2007_to_2018Q4.csv

# 3a. Baseline por línea de comandos (genera outputs/metrics.json)
python -m src.train --data data/accepted_2007_to_2018Q4.csv

#     (opcional) muestra rápida para probar:
python -m src.train --data data/accepted_2007_to_2018Q4.csv --nrows 300000

# 3b. Exploración completa (figuras + baseline)
jupyter notebook notebooks/01_exploracion_inicial.ipynb
```

El notebook regenera todas las figuras en `reports/figures/` y las métricas en
`outputs/metrics.json`.

## Reproducibilidad

- Semilla global fija (`SEED = 42` en `src/config.py`).
- Todo el preprocesamiento (imputación, escalado, one-hot) se ajusta **solo con
  el conjunto de entrenamiento** dentro de un `Pipeline` de scikit-learn.
- El modelo final nunca se evalúa sobre datos usados para tomar decisiones de
  modelado (partición temporal train/validación).
- Dependencias con versiones fijadas en `requirements.txt`.

## Control de leakage

El riesgo central de este dataset. `src/features.py` define una **lista blanca**
de variables de originación y una **lista negra** documentada (`LEAKAGE_COLUMNS`)
con las columnas generadas después de otorgar el crédito (pagos, recuperaciones,
FICO actualizado, acuerdos de liquidación/hardship). Solo entran al modelo las
variables conocidas en la decisión crediticia.
