# scripts/compute_indices/

Cálculo de índices climáticos, EOFs observacionales, reducción de nodos y extensión de las matrices causales al ensemble CMIP6.

**Orden de ejecución:** los scripts están agrupados en dos bloques secuenciales. Dentro de cada bloque, el orden de la tabla es el orden de ejecución.

## Bloque 1 - Índices y nodos observacionales

| Script | Descripción |
|---|---|
| `compute_wind_variables.py` | Descomposición de Helmholtz (función corriente ψ y potencial de velocidad χ) del campo de viento global mediante armónicos esféricos (`windspharm`), para ERA5 y los 7 modelos CMIP6, en los niveles 200 y 850 hPa. Resta explícitamente la media zonal de la función corriente. **Requiere WSL/Linux.** |
| `sensibilidad_orden_helmholtz.py` | Test de sensibilidad al orden entre la interpolación a 1° y la descomposición de Helmholtz en CHI200 y PSI200 (orden A, el del pipeline, frente a orden B, descomposición en rejilla nativa e interpolación posterior), en ERA5 y en el histórico de los modelos. **Requiere WSL/Linux.** |
| `compute_solvers_obs.py` | Calcula y guarda los solvers EOF observacionales (uno por variable candidata) sobre los dominios espaciales definidos para cada nodo, junto con la máscara de tierra usada en las variables terrestres (PR_land, TAS_land). |
| `validacion_PC1sst_N34.py` | Valida que el PC1 de SST observacional (y de cada modelo) se corresponde con el índice Niño 3.4, mediante correlación y alineación de signo. Genera las figuras de comparación PC1 vs. Niño 3.4 y los patrones de regresión EOF1. |
| `compute_nodos_obs.py` | Calcula series temporales y patrones espaciales de todas las variables candidatas observacionales: PC1 y PC2 como proyección sobre el solver y patrón de regresión sobre el campo global de anomalías. Genera las figuras comparativas de patrones y series. |
| `north_test_obs.py` | Aplica el criterio de North ajustado (grados de libertad efectivos por autocorrelación lag-1, Bretherton et al. 1999) para decidir si un PC está bien separado de sus vecinos en cada variable. Elimina manualmente SST_PC2 por no cumplir el criterio. |
| `compute_matriz_obs.py` | Selecciona, para cada variable candidata, el PC físicamente correcto (PC1 o PC2, decisión basada en inspección visual de los patrones) y construye la matriz observacional candidata `W_obs` (780 meses). |
| `reduccion_nodos.py` | Reduce el conjunto de nodos candidatos mediante correlación de Pearson (descarte directo \|r\|>0.85, zona de inspección 0.70-0.85) y Factor de Inflación de Varianza (VIF) iterativo (umbral 10), con criterios de desempate físicos y jerárquicos. Genera los 8 nodos finales (`nodos_finales.npy`) y el log de reducción (`log_reduccion.txt`). |

## Bloque 2 - Extensión a CMIP6

| Script | Descripción |
|---|---|
| `compute_matrices_cmip6.py` | Construye las matrices `W_hist`, `W_ssp245` y `W_ssp585` para cada uno de los 7 modelos, proyectando cada campo sobre la base EOF observacional (estrategia B) para los 8 nodos finales. |
| `diagnostico_climatologia.py` | Chequea la convención de anomalías: compara el sesgo climatológico entre el periodo histórico (base fija 1981-2010) y cada ventana de escenario (base propia de cada ventana) frente a la desviación estándar interna de cada nodo, para 3 modelos de familias distintas. |
| `diagnostico_nodos_cmip6.py` | Calcula las EOFs propias de cada modelo para evaluar (1) congruencia con el patrón observacional (coseno con signo) y (2) separación según el criterio de North, tanto frente a observaciones como entre el histórico y cada escenario del mismo modelo (congruencia intra-modelo). Incluye un control de sensibilidad frente a la tendencia lineal. |
