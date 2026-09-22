# Informe final — Predicción de riesgo crediticio (default) en Lending Club

> **Estado:** esqueleto de la entrega final. Las secciones marcadas con `TODO`
> se completan a lo largo de las semanas. Las cifras ya presentes provienen de
> la entrega previa (baseline) y se actualizarán con los modelos finales.

## 1. Resumen ejecutivo
`TODO` — 1 párrafo: problema, datos, mejor modelo, métrica lograda vs. baseline,
principal limitación. Se redacta al final.

## 2. Descripción del problema
Predecir, **al momento de la aprobación** de un préstamo peer-to-peer de Lending
Club, si el solicitante terminará en **default / mal desempeño**, usando solo
información disponible antes del desembolso. Motivación: mejores decisiones de
aprobación y tasas más justas según el riesgo real. (Ver `proposal.md` §4.)

## 3. Descripción del dataset
- Fuente: Lending Club vía Kaggle (`wordsforthewise/lending-club`).
- Volumen: **2 260 701** préstamos, archivo consolidado 2007–2018.
- Variables de trabajo: subconjunto de originación (ver `proposal.md` §7).
- Licencia: datos públicos, descarga libre. `TODO`: decisiones de muestreo.

## 4. Limpieza y preparación de datos
`TODO` (notebook `02_limpieza_features.ipynb`):
- Definición final del target a partir de `loan_status` (**ver nota §5 sobre
  préstamos `Current`**).
- Parsing de `int_rate`/`revol_util` (texto `%`→float), `emp_length`, `term`.
- `earliest_cr_line` → antigüedad crediticia.
- Tratamiento justificado de faltantes (patrón MNAR) y outliers.
- Codificación de categóricas (one-hot / target encoding según cardinalidad).

## 5. Análisis exploratorio
Resumen de la entrega previa (`01_exploracion_inicial.ipynb`):
- Tasa de default global ≈ 20%; **desbalance** de clases.
- Relación monótona riesgo/`grade` (A ≈ 6.6% → G ≈ 50.9% de default).
- **Deriva temporal**: la tasa de default sube en los años recientes.
- Faltantes concentrados en variables de historial (algunos MNAR).
`TODO`: consolidar figuras finales en `reports/figures/`.

## 6. Estrategia de partición y validación
- Split **temporal**: train 2007–2015; validación/test 2016–2018.
- Validación cruzada temporal (ventanas móviles) para tuning.
- Test intocado: no se usa para decisiones de modelado. Semilla = 42.

## 7. Baselines
- Trivial: clase mayoritaria (AUC-ROC = 0.5).
- Logístico sin tuning ni balanceo (referencia real).
`TODO`: tabla con AUC-ROC / AUC-PR de ambos baselines.

## 8. Modelos entrenados
`TODO` (notebook `03_modelos.ipynb`): ≥3 familias —
1. Regresión logística regularizada (L1/L2).
2. Random Forest.
3. Gradient Boosting (HistGradientBoosting / XGBoost).
Con manejo de desbalance (class weights o resampling).

## 9. Búsqueda de hiperparámetros
`TODO`: espacio de búsqueda por modelo, método (grid/random/Bayes) y validación
temporal usada; tabla de mejores configuraciones.

## 10. Evaluación final
`TODO`: métricas en el **test** (2018) del modelo seleccionado, comparación
justa vs. baseline; curvas ROC y PR.

## 11. Interpretación
`TODO`: variables más influyentes (coeficientes / importancias / SHAP) y su
lectura de negocio.

## 12. Análisis de errores
`TODO` (notebook `04_analisis_errores.ipynb`): errores por segmento (`grade`,
tramo de ingreso, `purpose`, año), casos problemáticos (falsos negativos
costosos).

## 13. Riesgos éticos, sesgos y limitaciones
- **Sesgo de selección**: solo solicitantes aprobados por LC (no rechazados).
- **Leakage**: columnas post-desembolso excluidas (ver `proposal.md` §8).
- **Deriva temporal** de políticas de crédito.
- `TODO`: equidad por segmentos sensibles.

## 14. Trabajo futuro
`TODO`: calibración de probabilidades, análisis costo-beneficio del umbral,
modelos de supervivencia (tiempo hasta default), datos de rechazados.

## 15. Instrucciones de reproducción
Ver `README.md`. `TODO`: fijar comando único que regenera modelos y métricas.
