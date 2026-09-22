"""Configuración central del proyecto: rutas, semilla y constantes.

Mantener aquí todo lo que se comparte entre módulos y notebooks evita
inconsistencias (p. ej. usar semillas distintas en train y evaluate).
"""
from pathlib import Path

# --- Reproducibilidad -------------------------------------------------------
SEED = 42

# --- Rutas ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUTS_DIR = ROOT / "outputs"

for _d in (DATA_DIR, FIGURES_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Nombre por defecto del archivo de datos crudos.
# Cámbialo si tu archivo se llama distinto (ver data/README.md).
RAW_FILENAME = "accepted_2007_to_2018Q4.csv"

# --- Variable objetivo ------------------------------------------------------
TARGET_COL = "loan_status"
TARGET_BINARY = "target_default"  # 1 = mal desempeño, 0 = pagado

# Estados que representan "mal desempeño" (evento positivo = 1).
BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
    "Late (31-120 days)",
}
# Estados que representan "buen desempeño" (evento negativo = 0).
GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
# Todo lo demás (Current, In Grace Period, Late (16-30 days), Issued...) es un
# resultado aún NO observado -> se excluye del entrenamiento supervisado.
