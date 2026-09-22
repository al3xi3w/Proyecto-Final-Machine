# Propuesta de proyecto — Entrega previa

> **Nota:** Las cifras (filas, tasa de default, métricas del baseline) provienen
> de una corrida real sobre una muestra aleatoria del 40% del dataset, ejecutada
> en `notebooks/01_exploracion_inicial.ipynb` y guardada en `outputs/metrics.json`.
> Falta completar los **nombres de los integrantes** (punto 2).

## 1. Título del proyecto
**Predicción de incumplimiento de préstamos personales en Lending Club usando
únicamente información disponible al momento de la decisión crediticia.**

## 2. Integrantes
- ⟨Nombre 1⟩
- ⟨Nombre 2⟩
- ⟨Nombre 3⟩

*(Reemplazar por los integrantes reales del equipo.)*

## 3. Dataset elegido
**Lending Club Loan Data** — préstamos personales originados entre 2007 y 2018.

- **Fuente:** Lending Club, distribuida vía Kaggle
  (`wordsforthewise/lending-club`, archivo `accepted_2007_to_2018Q4.csv`).
- **Tamaño:** 2 260 701 préstamos y 151 columnas en el archivo completo. Para la
  exploración inicial se trabajó con una **muestra aleatoria reproducible del
  40%**; tras filtrar a préstamos con resultado observado quedan **547 578
  préstamos** (442 948 train / 104 630 validación out-of-time).
- **Licencia / acceso:** datos públicos publicados por Lending Club; el mirror
  de Kaggle requiere una cuenta gratuita. No hay restricciones de uso académico.
- **Complejidad (por qué califica):** cumple varias de las condiciones exigidas
  — >50 000 filas, >50 variables, variables temporales (fechas de emisión y de
  historial crediticio), fuerte **desbalance de clases**, abundantes **valores
  faltantes**, columnas de texto libre de alta cardinalidad (`emp_title`,
  `title`) y un **riesgo de leakage severo y realista** que es el corazón del
  ejercicio.

## 4. Pregunta predictiva
Dado un solicitante y las condiciones del préstamo **en el momento en que
Lending Club debe decidir si aprueba y a qué tasa**, ¿el préstamo terminará en
**incumplimiento / mal desempeño** (Charged Off, Default, mora tardía) en lugar
de ser pagado por completo?

Es un problema de **clasificación binaria supervisada**.

## 5. Variable objetivo
`target_default` ∈ {0, 1}, derivada de `loan_status`:

- **1 (evento positivo, "malo"):** `Charged Off`, `Default`,
  `Late (31-120 days)` y sus variantes "Does not meet the credit policy…".
- **0 ("bueno"):** `Fully Paid` y su variante de política.
- **Excluidos:** `Current`, `In Grace Period`, `Late (16-30 days)`, `Issued` —
  su resultado **aún no es observable**, por lo que etiquetarlos sería incorrecto.

## 6. Unidad de predicción
**Un préstamo individual** en el instante de su solicitud/originación. Cada fila
= un préstamo. La predicción es la probabilidad de incumplimiento de ese
préstamo a lo largo de su vida.

## 7. Variables disponibles antes de la predicción
Solo se usan variables conocidas en la **originación** (lista blanca en
`src/features.py`). Ejemplos:

- **Solicitud/préstamo:** `loan_amnt`, `term`, `int_rate`, `installment`,
  `grade`, `sub_grade`, `purpose`, `application_type`, `initial_list_status`.
- **Solicitante:** `annual_inc`, `emp_length`, `home_ownership`,
  `verification_status`, `addr_state`, `dti`.
- **Historial de buró al originar:** `fico_range_low/high`, `delinq_2yrs`,
  `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`,
  `total_acc`, `pub_rec_bankruptcies`, `earliest_cr_line`
  (→ antigüedad crediticia).

`int_rate`, `grade` y `sub_grade` **sí** son válidas: Lending Club las asigna en
la originación, antes de fondear, por lo que están disponibles en la decisión.

## 8. Riesgos de leakage
Es el riesgo dominante de este dataset. Muchas columnas se generan **después**
de otorgar el crédito y filtrarían el resultado:

- **Pagos y saldos vivos:** `total_pymnt`, `total_rec_prncp`, `total_rec_int`,
  `out_prncp`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`.
- **Recuperaciones tras el default:** `recoveries`, `collection_recovery_fee`.
- **Buró actualizado durante la vida del préstamo:** `last_fico_range_high/low`,
  `last_credit_pull_d`.
- **Acuerdos de liquidación / hardship:** `debt_settlement_flag`,
  `settlement_*`, `hardship_*` (solo existen si el préstamo ya se deterioró).
- **Fondeo:** `funded_amnt`, `funded_amnt_inv` (posteriores a la aprobación; se
  usa `loan_amnt` en su lugar).

**Control:** lista blanca explícita de features de originación + lista negra
documentada (`LEAKAGE_COLUMNS`). Toda imputación/escalado se ajusta **solo con
train** dentro de un `Pipeline`.

## 9. Métrica principal y métrica secundaria
- **Principal: ROC-AUC** — mide la capacidad de ordenar solicitantes por riesgo,
  robusta al desbalance y al umbral. Estándar en scoring crediticio.
- **Secundarias: PR-AUC (average precision)** y **estadístico KS**, ambas
  sensibles al desempeño sobre la clase minoritaria (los malos), más
  informativas que accuracy. Se reportan además precision/recall a un umbral
  operativo. **No** se usará accuracy como métrica principal (penalizado).

## 10. Plan de validación
**Partición temporal (out-of-time)** por `issue_d`: entrenamiento con los
préstamos más antiguos, validación con el ~20% más reciente. Evita el leakage
temporal de un split aleatorio y refleja el uso real (predecir préstamos
futuros con un modelo entrenado en el pasado). La exploración confirma esta
necesidad: la tasa de default sube de ~20% (train, préstamos antiguos) a 26.5%
(validación, préstamos recientes), señal de **deriva temporal**. Para la entrega
final se añadirá
validación cruzada temporal (`TimeSeriesSplit`) para la búsqueda de
hiperparámetros, dejando el bloque más reciente como test intocado.

## 11. Modelo baseline
Dos referencias honestas:
- **Baseline trivial:** `DummyClassifier(strategy="prior")` (predice la tasa
  base) — piso de comparación.
- **Baseline real: Regresión logística sin balanceo ni tuning** sobre las
  features de originación, dentro de un pipeline con imputación + escalado +
  one-hot. Resultado en validación out-of-time: **ROC-AUC ≈ 0.696**,
  **PR-AUC ≈ 0.43**, **KS ≈ 0.29** (vs. ROC-AUC 0.5 de la referencia trivial).
  Al umbral 0.5 el **recall es bajo (~6%)**: sin balanceo el modelo casi no
  marca defaults, lo que motiva ajustar el umbral y probar balanceo/resampling
  en la fase de modelado.

## 12. Riesgos técnicos
- **Desbalance de clases** (~20% de default en train, 26.5% en la validación
  out-of-time más reciente): se usan métricas apropiadas (no accuracy). El
  baseline es sin balanceo; el manejo del desbalance (`class_weight` /
  resampling / ajuste de umbral) se aborda en la fase de modelado.
- **Alto volumen** (2.26M filas): se trabaja sobre una muestra aleatoria
  representativa (40%) leída por chunks y solo con las columnas necesarias; el
  muestreo es reproducible (semilla) y está documentado.
- **Faltantes no aleatorios:** columnas como `mths_since_last_delinq` faltan
  cuando *no hubo* morosidad → la ausencia es informativa; se evalúa un
  indicador de faltante.
- **Alta cardinalidad** (`emp_title`, `addr_state`): se excluye o agrupa.
- **Deriva temporal:** las políticas de crédito de LC cambiaron entre 2007–2018;
  la validación out-of-time lo hace explícito.

## 13. Plan de trabajo (semanas restantes)
| Semana | Actividad |
|---|---|
| 2 | Limpieza completa, tratamiento formal de faltantes/outliers, feature engineering (`02_limpieza_features.ipynb`). |
| 3 | Modelos comparables: regresión logística regularizada, árboles/Random Forest y gradient boosting (`03_modelos.ipynb`). |
| 4 | Búsqueda de hiperparámetros con validación temporal; selección de modelo. |
| 5 | Análisis de errores por segmentos, interpretabilidad y sesgos (`04_analisis_errores.ipynb`). |
| 6 | Informe final, figuras, presentación y verificación de reproducibilidad. |
