"""
preprocess_hadisst.py
Preprocesa la SST de HadISST: recorta 1950-2014 y 60°S-60°N, normaliza el calendario a proleptic_gregorian (día 15 de cada mes) y guarda solo sst en float32. Ya viene en la rejilla de 1°, no se interpola.
Entradas: data/raw/hadisst/sea_surface_temperature/HadISST_sst.nc
Salidas: data/processed/hadisst/sea_surface_temperature/HadISST_sst.nc
Memoria: sección 4.4.
"""

import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

# Cargamos datos
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

data_dir = root_data / "data" / "raw" / "hadisst" / "sea_surface_temperature"
filename = data_dir / "HadISST_sst.nc"
ds = xr.open_dataset(filename, engine="netcdf4")

# Analizamos datos
aux = input('HadISST data set initial info [y/n]  ')
if aux == 'y':
    print(ds)
    print("Fecha de fin:", ds.time.values[-1])
    print(ds.sizes)
    print(ds.sst.attrs)
    print(ds.time.encoding)
    print(ds.data_vars)

# Periodo temporal común
ds = ds.sel(time=slice('1950-01-01', '2014-12-31'))

# Rejilla común
ds = ds.sortby("latitude")
ds = ds.sel(latitude=slice(-60, 60))

# Normalización de calendario
ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False)
new_timestamps = pd.date_range(start="1950-01-01", periods=ds.sizes["time"], freq="MS") + pd.Timedelta(days=14)
ds = ds.assign_coords(time=new_timestamps.normalize())

# Comprobación visual de valores físicamente razonables (t0)
aux = input('Sea surface temperature plots for t0 [y/n]  ')
if aux == 'y':
    sst0 = ds.sst.isel(time=0)
    sst0.plot(cmap='magma')
    plt.show()
    sst0_clean = sst0.where(sst0 > -50)
    sst0_clean.plot(cmap='magma')
    plt.show()
    mask = sst0.isnull()
    mask.plot(cmap='magma')
    plt.show()

# Limpieza
ds = ds[['sst']].astype(np.float32)
extra_coords = [c for c in ds.coords if c not in ['time', 'latitude', 'longitude']]
ds = ds.drop_vars(extra_coords)
ds.attrs = {
    "title": "HadISST Processed Dataset - Sea Surface Temperature",
    "source": "Met Office Hadley Centre Sea Ice and Sea Surface Temperature data set",
    "variable_id": "sst",
    "grid": "Native regular 1.0° x 1.0° grid",
    "lat_definition": "cell centers at -59.5 to 59.5 (1° spacing)",
    "lon_definition": "cell centers at -179.5 to 179.5 (1° spacing)",
    "calendar": "proleptic_gregorian",
    "time_resolution": "monthly (middle of month ~ day 15)",
    "time_coverage": "1950-01 to 2014-12",
    "units": "degC" }

# Detalles finales
aux = input('HadISST data set final info [y/n]  ')
if aux == 'y':
    print(ds)
    print(ds.sizes)

# Salida
out_dir = root_data / "data" / "processed" / "hadisst" / "sea_surface_temperature"
out_dir.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(out_dir / "HadISST_sst.nc")






