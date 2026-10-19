# scripts/seleccion_modelo/

Cálculo del índice Niño 3.4 para observaciones y los 7 modelos CMIP6, y comparación de métricas de fidelidad ENSO (estándar Planton et al. 2021) para verificar que el ensemble reproduce razonablemente el ciclo ENSO antes de continuar con el resto del pipeline.

**Orden de ejecución:** `seleccion_modelo_nino34.py` → `plot_nino34_comparison.py`.

| Script | Descripción |
|---|---|
| `seleccion_modelo_nino34.py` | Calcula el índice Niño 3.4 (anomalía mensual bruta y suavizada con filtro triangular de 5 meses, sin tendencia lineal) y el perfil zonal de SST ecuatorial (5°S-5°N). El origen de datos se selecciona a mano mediante la variable `datos` al inicio del script (`'hadisst'` o `'cmip6'`); **hay que ejecutarlo dos veces**, una por cada valor, para generar los índices de observaciones y de los 7 modelos. |
| `plot_nino34_comparison.py` | Compara cada modelo frente a HadISST mediante las cuatro métricas de Planton et al. (2021): amplitud, estacionalidad, patrón espacial (RMSE del beta de regresión, evaluado en los diciembres) y sesgo de SST ecuatorial. Guarda siempre una figura de comparación temporal por modelo y una figura conjunta del perfil zonal ecuatorial de todo el ensamble, y escribe `resultados/resumen_metricas_planton.csv` con las métricas y los valores σ, R y β_max de cada modelo y de HadISST. |

