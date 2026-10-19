"""
preprocess_cru.py
Preprocesa precipitación (mm/mes a kg m-2 s-1) y temperatura de CRU TS4.09: normaliza calendario y unidades e interpola a la rejilla de 1° (bilineal, xesmf). La precipitación se recorta a 60°S-60°N.
Entradas: data/raw/cru/<variable>/cru_ts4.09.1901.2024.<pre|tmp>.dat.nc
Salidas: data/processed/cru/<variable>/CRU_<pr|tas>.nc
Memoria: sección 4.4.
"""

import xarray as xr
import pandas as pd
import numpy as np
import xesmf as xe
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings("ignore")

root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

variables = ["precipitation", "near_surface_air_temperature"]
nv_cmip = ["pr", "tas"]
units = ["kg m-2 s-1", "degC"]
v_u = dict(zip(variables, zip(units, nv_cmip)))

target_grid = xr.Dataset({
    "lat": (["lat"], np.arange(-89.5, 90.5, 1.0)),
    "lon": (["lon"], np.arange(-179.5, 180.5, 1.0))})

for v, e in v_u.items():
    u = e[0]
    nv = e[1]
    print(f"\n----{v}----\n")

    # Cargamos datos
    ds_dir = root_data / "data" / "raw" / "cru" / v
    print(ds_dir)
    filename = next(ds_dir.glob("*.nc"))
    ds = xr.open_dataset(filename, engine="netcdf4")

    # Analizamos datos
    aux = input('Dataset initial info [y/n]  ')
    nv_cru = list(ds.data_vars)[0]
    ds = ds[[nv_cru]]
    if aux == 'y':
        print(ds)
        print(ds[nv_cru].attrs)
        print(ds.time.encoding)

    with tqdm(total=10, desc=f"Procesando", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}') as pbar:

        # Renombramos variable y coordenada de tiempo a nombre estándar del proyecto
        pbar.set_postfix_str("Preparando: Nombres de variables y coordenadas")
        ds = ds.rename({nv_cru: nv})
        pbar.update(1)

        # Homogeneizamos variables y unidades (la precipitación pasa de mm/mes a kg m-2 s-1)
        pbar.set_postfix_str("Preparando: Correcciones Físicas")
            # Convertir unidades precipitación
        if v == "precipitation":
            with xr.set_options(keep_attrs=True):
                sec_per_month = ds.time.dt.days_in_month * 24 * 3600
                ds[nv] = ds[nv] / sec_per_month
            ds[nv].attrs['units'] = 'kg m-2 s-1'
        elif v == "near_surface_air_temperature":
                # La temperatura ya viene en grados Celsius: solo normalizamos el nombre de la unidad
            ds[nv].attrs['units'] = 'degC'
            # Comprobamos que las unidades finales son las esperadas
        pbar.set_postfix_str("Preparando: Unidades")
        current_units = ds[nv].attrs.get('units')
        if current_units != u:
            raise ValueError(f"La variable {v} no tiene las unidades correctas ({current_units}).")
        pbar.update(1)

        # Periodo temporal común y normalización de calendario
        pbar.set_postfix_str("Preparando: Calendario")
        ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False, align_on="year")
        ds = ds.sel(time=slice('1950-01-01', '2014-12-31'))
        new_timestamps = pd.date_range(start="1950-01-01", periods=ds.sizes["time"], freq="MS") + pd.Timedelta(days=14)
        ds = ds.assign_coords(time=new_timestamps.normalize())
        pbar.update(1)

        # Rejilla común (también pasamos de 0-360º a -180-180º)
        pbar.set_postfix_str("Preparando: Pesos Regrid ")
        weights_file = root_data / "data" / "processed" / "cru" / v / f"weights.nc"
        weights_file.parent.mkdir(parents=True, exist_ok=True)
        ds_in = ds
        pbar.update(1)
        pbar.set_postfix_str("Preparando: Pesos Regrid ")
        regridder = xe.Regridder(ds_in, target_grid, "bilinear", periodic=True, ignore_degenerate=True,
                                 reuse_weights=weights_file.exists(),
                                 filename=str(weights_file))
        pbar.update(1)
        pbar.set_postfix_str("Preparando: Pesos Regrid ")
        ds = regridder(ds_in)
        pbar.update(1)
        pbar.set_postfix_str("Preparando: Pesos Regrid ")
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})
        pbar.update(1)
        pbar.set_postfix_str("Preparando: Pesos Regrid ")
        if v != "near_surface_air_temperature":
            ds = ds.sel(latitude=slice(-60, 60), longitude=slice(-179.5, 179.5))
        pbar.update(1)

        # Limpieza para quitar peso a la RAM
        pbar.set_postfix_str("Preparando: Limpieza")
        ds = ds[[nv]].astype(np.float32)
        extra_coords = [c for c in ds.coords if c not in ['time', 'latitude', 'longitude']]
        ds = ds.drop_vars(extra_coords)
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
        plt.title(f"CRU - {nv}")
        plt.show()

    # Reescribimos atributos
    ds.attrs = {
        "title": f"CRU TS Processed Dataset - {v.replace('_', ' ').title()}",
        "source": "Climatic Research Unit (CRU) Time Series",
        "variable_id": nv,
        "grid": "Regridded to regular 1.0° x 1.0° grid (bilinear interpolation)",
        "lat_definition": "cell centers at -89.5 to 89.5 (1° spacing)" if (v == "near_surface_air_temperature") else "cell centers at -59.5 to 59.5 (1° spacing)",
        "lon_definition": "cell centers at -179.5 to 179.5 (1° spacing)",
        "calendar": "proleptic_gregorian",
        "time_resolution": "monthly (middle of month ~ day 15)",
        "time_coverage": "1950-01 to 2014-12",
        "units": u }
    if v == "precipitation":
        ds.attrs["units_note"] = "Precipitation converted from mm/month to kg m-2 s-1"

    # Salida
    aux = input(f'Dataset final info [y/n]  ')
    if aux == 'y':
        print(ds)

    out_dir = root_data / "data" / "processed" / "cru" / v
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_dir / f"CRU_{nv}.nc")
