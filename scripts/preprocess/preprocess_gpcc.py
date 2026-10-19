"""
preprocess_gpcc.py
Preprocesa la precipitación de GPCC: pasa de mm/mes a kg m-2 s-1, recorta 1950-2014 y 60°S-60°N y normaliza el calendario. Ya viene en la rejilla de 1°, no se interpola.
Entradas: data/raw/gpcc/precipitation/gpcc_monthly_1946_2015.nc
Salidas: data/processed/gpcc/precipitation/GPCC_pr.nc
Memoria: sección 4.4.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings("ignore")

root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

variables = ["precipitation"]
nv_cmip = ["pr"]
units = ["kg m-2 s-1"]
v_u = dict(zip(variables, zip(units, nv_cmip)))

for v, e in v_u.items():
    u = e[0]
    nv = e[1]
    print(f"\n----{v}----\n")

    # Cargamos datos
    ds_dir = root_data / "data" / "raw" / "gpcc" / v
    print(ds_dir)
    filename = next(ds_dir.glob("*.nc"))
    ds = xr.open_dataset(filename, engine="netcdf4")

    # Analizamos datos
    aux = input('Dataset initial info [y/n]  ')
    nv_gpcc = list(ds.data_vars)[0]
    ds = ds[[nv_gpcc]]
    if aux == 'y':
        print(ds)
        print(ds[nv_gpcc].attrs)
        print(ds.time.encoding)

    with tqdm(total=6, desc=f"Procesando", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}') as pbar:

        # Renombramos variable y coordenada de tiempo a nombre estándar del proyecto
        pbar.set_postfix_str("Preparando: Nombres de variables y coordenadas")
        ds = ds.rename({nv_gpcc: nv})
        pbar.update(1)

        # Homogeneizamos variables y unidades (mm/month --> kg m-2 s-1)
        pbar.set_postfix_str("Preparando: Correcciones Físicas")
            # Convertir unidades precipitacion
        if v == "precipitation":
            with xr.set_options(keep_attrs=True):
                sec_per_month = ds.time.dt.days_in_month * 24 * 3600
                ds[nv] = ds[nv] / sec_per_month
            ds[nv].attrs['units'] = 'kg m-2 s-1'
        pbar.set_postfix_str("Preparando: Unidades")
        current_units = ds[nv].attrs.get('units')
        if current_units != u:
            raise ValueError(f"La variable {v} no tiene las unidades correctas ({current_units}).")
        pbar.update(1)

        # Periodo temporal común y normalización de calendario
        pbar.set_postfix_str("Preparando: Calendario")
        ds = ds.sel(time=slice('1950-01-01', '2014-12-31'))
        new_timestamps = pd.date_range(start="1950-01-01", periods=ds.sizes["time"], freq="MS") + pd.Timedelta(days=14)
        ds = ds.assign_coords(time=new_timestamps.normalize())
        pbar.update(1)

        # Rejilla común
        # GPCC ya está en la rejilla de 1°: solo renombramos coordenadas y recortamos 60°S-60°N, sin interpolar
        pbar.set_postfix_str("Preparando: Rejilla común")
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
        ds = ds.sortby("latitude")
        ds = ds.sel(latitude=slice(-60, 60), longitude=slice(-179.5, 179.5))
        pbar.update(1)

        # Limpieza para quitar peso a la RAM
        pbar.set_postfix_str("Preparando: Limpieza")
        ds = ds[[nv]].astype(np.float32)
        extra_coords = [c for c in ds.coords if c not in ['time', 'latitude', 'longitude']]
        ds = ds.drop_vars(extra_coords)
        ds.attrs = {
            "title": "GPCC Precipitation processed dataset",
            "source": "GPCC Monthly Analysis v2025",
            "variable_id": nv,
            "grid": "Regular 1.0° x 1.0° grid centered on 0.5°",
            "lat_definition": "cell centers at -59.5 to 59.5 (1° spacing)",
            "lon_definition": "cell centers at -179.5 to 179.5 (1° spacing)",
            "calendar": "proleptic_gregorian",
            "time_resolution": "monthly (middle of month ~ day 15)",
            "time_coverage": "1950-01 to 2014-12",
            "units": u,
            "units_note": "precipitation converted to kg m-2 s-1"}
        pbar.update(1)

        # Ejecución
        pbar.set_postfix_str("Ejecutando todo el plan...")
        ds = ds.load()
        pbar.update(1)
        pbar.set_postfix_str("¡Finalizado!")

    # Comprobación visual de valores físicamente razonables (t0)
    aux = input('Plots for t0 [y/n]  ')
    if aux == 'y':
        var0 = ds[nv].isel(time=0)
        var0.plot(cmap='viridis')
        plt.title(f"GPCC - {nv}")
        plt.show()

    # Salida
    aux = input(f'Dataset final info [y/n]  ')
    if aux == 'y':
        print(ds)

    out_dir = root_data / "data" / "processed" / "gpcc" / v
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_dir / f"GPCC_{nv}.nc")
