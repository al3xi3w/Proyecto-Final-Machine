# Datos

Los datos **no se versionan** en el repositorio por su tamaño (~1.8 GB) y por la
política de Kaggle. Este directorio solo contiene instrucciones.

## Cómo obtener el dataset

**Fuente:** Lending Club Loan Data, mirror de Kaggle
`wordsforthewise/lending-club`
(<https://www.kaggle.com/datasets/wordsforthewise/lending-club>).

1. Crea una cuenta gratuita en Kaggle y acepta las reglas del dataset.
2. Descarga `accepted_2007_to_2018Q4.csv` (préstamos aprobados).
3. Coloca el archivo en esta carpeta:

   ```
   data/accepted_2007_to_2018Q4.csv
   ```

   Si usas otro nombre o una muestra, actualiza `RAW_FILENAME` en
   `src/config.py` o pasa la ruta con `--data`.

## Muestra de trabajo

El archivo completo tiene ~2.26M filas. Para la exploración inicial puedes usar
una muestra ajustando `SAMPLE_ROWS` en el notebook o `--nrows` en el script:

```bash
python -m src.train --data data/accepted_2007_to_2018Q4.csv --nrows 300000
```

## Descarga por línea de comandos (opcional)

Con la API de Kaggle configurada (`~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d wordsforthewise/lending-club -f accepted_2007_to_2018Q4.csv.gz -p data/
gunzip data/accepted_2007_to_2018Q4.csv.gz
```
