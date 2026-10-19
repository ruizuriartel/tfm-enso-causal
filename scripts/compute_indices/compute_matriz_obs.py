"""
compute_matriz_obs.py
Elige para cada variable la PC físicamente correcta (PC1 o PC2, por inspección visual de los patrones) y construye la matriz observacional candidata de 780 meses, con la figura resumen de los patrones de regresión. Pregunta por pantalla antes de mostrar las figuras.
Entradas: data/indices/nodos_obs/series_pc1_pc2.pkl y patrones_pc1_pc2.pkl (de compute_nodos_obs.py)
Salidas: data/indices/nodos_obs/W_obs_matriz_780_candidata.nc; figures/nodos_obs/regression_patterns.png
Memoria: sección 5.3.
"""

import xarray as xr
import numpy as np
import warnings
import os
from pathlib import Path
import pickle

import compute_nodos_obs as cno

warnings.filterwarnings("ignore")

# Definimos directorios de entrada y salida
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

in_dir = root_data / "data" / "indices" / "nodos_obs"
fig_dir = root / "figures" / "nodos_obs"; fig_dir.mkdir(parents=True, exist_ok=True)

# Subimos series y patrones precalculados
with open(in_dir / "series_pc1_pc2.pkl", "rb") as f:
    series_todas = pickle.load(f)
with open(in_dir / "patrones_pc1_pc2.pkl", "rb") as f:
    patrones_todos = pickle.load(f)

# Selección de componentes principales físicas basada en el análisis visual
        # (PC1 o PC2 según el comportamiento del campo -> analizamos patrones)
nodos_pc1 = ['OLR', 'PSL', 'CHI200', 'CHI850', 'PSI200', 'PSI850', 'Z200_PNA']
nodos_pc2 = ['SST', 'Z200_PSA', 'PR_land', 'TAS_land']
keys = ['N34'] + [f"{n}_PC1" for n in nodos_pc1] + [f"{n}_PC2" for n in nodos_pc2]

series_finales = {k.replace('_PC1', ''): series_todas[k] for k in keys if k in series_todas}
patrones_finales = {k.replace('_PC1', '').replace('_PC2', ''): patrones_todos[k] for k in keys if k in patrones_todos}

# Figura resumen con los patrones de regresión EOF
cno.plot_pattern_grid(patrones_finales, fig_dir / "regression_patterns.png", 4)

# Construimos y guardamos la matriz candidata
nombres = list(series_finales.keys())
t = series_finales[nombres[0]].time.values
for n in nombres[1:]:
    t = np.intersect1d(t, series_finales[n].time.values)

ds_matriz_780 = xr.Dataset(
    {n: xr.DataArray(series_finales[n].sel(time=t).values, coords=[t], dims=['time']) for n in nombres})
ds_matriz_780.to_netcdf(in_dir / "W_obs_matriz_780_candidata.nc")
