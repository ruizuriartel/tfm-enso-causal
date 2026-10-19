"""
preprocess_era5.py
Preprocesa ERA5: nombres estándar, geopotencial a altura geopotencial (·/g), signo de la radiación neta invertido para obtener OLR, calendario proleptic_gregorian e interpolación bilineal (xesmf) a 1°. OLR y PSL se recortan a 60°S-60°N.
Entradas: data/raw/era5/<variable>/*.nc
Salidas: data/processed/era5/<variable>/ERA5_<ua|va|zg|rlut|psl>.nc
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
from scipy.constants import g
import os

warnings.filterwarnings("ignore")

root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

variables_cmip = ["eastward_wind_200hPa", "eastward_wind_850hPa", "northward_wind_200hPa",
                  "northward_wind_850hPa", "geopotential_height_200hPa",
                  "toa_outgoing_longwave_radiation", "sea_level_pressure"]
variables_era = ["u_component_of_wind_200hPa", "u_component_of_wind_850hPa", "v_component_of_wind_200hPa",
                 "v_component_of_wind_850hPa", "geopotential_200hPa",
                  "mean_top_net_long_wave_radiation_flux" , "mean_sea_level_pressure"]
nv_cmip = ["ua", "ua", "va", "va", "zg", "rlut", "psl"]
units = ["m s-1", "m s-1", "m s-1", "m s-1", "m", "W m-2", "Pa"]
v_u = dict(zip(variables_cmip, zip(variables_era, units, nv_cmip)))

target_grid = xr.Dataset({
    "lat": (["lat"], np.arange(-89.5, 90.5, 1.0)),
    "lon": (["lon"], np.arange(-179.5, 180.5, 1.0))})

for v, e in v_u.items():
    v_era = e[0]
    u = e[1]
    nv = e[2]
    print(f"\n----{v}----\n")

    # Cargamos datos
    ds_dir = root_data / "data" / "raw" / "era5" / v_era
    print(ds_dir)
    filename = next(ds_dir.glob("*.nc"))
    ds = xr.open_dataset(filename, engine="netcdf4")

    # Analizamos datos
    aux = input('Dataset initial info [y/n]  ')
    nv_era = list(ds.data_vars)[0]
    if aux == 'y':
        print(ds)
        print(ds[nv_era].attrs)
        print(ds.valid_time.encoding)

    with tqdm(total=10, desc=f"Procesando", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}') as pbar:

        # Renombramos variable y coordenada de tiempo a nombre estándar del proyecto
        pbar.set_postfix_str("Preparando: Nombres de variables y coordenadas")
        ds = ds.rename({nv_era: nv, 'valid_time': 'time'})
        pbar.update(1)

        # Homogeneizamos variables y unidades
        pbar.set_postfix_str("Preparando: Correcciones Físicas")
            # Convertir Geopotencial a Altura Geopotencial
        if "geopotential" in v:
            with xr.set_options(keep_attrs=True):
                ds[nv] = ds[nv] / g
            ds[nv].attrs['units'] = 'm'
            # Convertir Mean Top Net Thermal Radiation a OLR
        elif v == "toa_outgoing_longwave_radiation":
            with xr.set_options(keep_attrs=True):
                ds[nv] = -ds[nv]
            ds[nv].attrs['units'] = 'W m-2'
            # Convertir Temperatura de Kelvin a Celsius
        elif v == "near_surface_air_temperature":
            with xr.set_options(keep_attrs=True):
                ds[nv] = ds[nv] - 273.15
            ds[nv].attrs['units'] = 'degC'
            # Renombrar unidad de viento
        elif "wind" in v:
            ds[nv].attrs['units'] = 'm s-1'
            # Analisis unidades finales
        pbar.set_postfix_str("Preparando: Unidades")
        current_units = ds[nv].attrs.get('units')
        if current_units != u:
            raise ValueError(f"La variable {v} no tiene las unidades correctas ({current_units}).")
        pbar.update(1)


        # Eliminamos la dimension de presion
        pbar.set_postfix_str("Preparando: Dimensiones")
        dim = 'pressure_level'
        if dim in ds.dims:
            ds = ds.squeeze(dim, drop=True)
        elif dim in ds.coords:              # Si la presión queda como coordenada escalar, también se elimina en la limpieza
            ds = ds.drop_vars(dim)
        pbar.update(1)

        # Periodo temporal común y normalización de calendario
        pbar.set_postfix_str("Preparando: Calendario")
        ds = ds.sel(time=slice('1950-01-01', '2014-12-31'))
        new_timestamps = pd.date_range(start="1950-01-01", periods=ds.sizes["time"], freq="MS") + pd.Timedelta(days=14)
        ds = ds.assign_coords(time=new_timestamps.normalize())
        pbar.update(1)

        # Rejilla común (también pasamos de 0-360º a -180-180º)
        pbar.set_postfix_str("Preparando: Pesos Regrid ")
        ds = ds.sortby("latitude")
        weights_file = root_data / "data" / "processed" / "era5" / v / f"weights.nc"
        weights_file.parent.mkdir(parents=True, exist_ok=True)
        ds_in = ds.rename({"latitude": "lat", "longitude": "lon"})
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
        if v != "near_surface_air_temperature" and "geopotential" not in v and "wind" not in v:
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
        if nv in ['ua', 'va']:
            var0.plot(cmap='coolwarm', center=0)
        else:
            var0.plot(cmap='viridis')
        plt.title(f"ERA5 - {nv}")
        plt.show()

    # Reescribimos atributos
    lat_def = "cell centers at -89.5 to 89.5 (1° spacing)" if (v == "near_surface_air_temperature" or "geopotential" in v or "wind" in v) else "cell centers at -59.5 to 59.5 (1° spacing)"
    units_note = "Standardized"
    if "geopotential" in v:
        units_note = "Geopotential divided by g to obtain Geopotential Height (m)"
    elif "radiation" in v:
        units_note = "Sign inverted to match Outgoing Longwave Radiation (W m-2)"
    elif "temperature" in v:
        units_note = "Converted from Kelvin to degC"
    ds.attrs = {
        "title": f"ERA5 Reanalysis Processed Dataset - {v.replace('_', ' ').title()}",
        "source": "ECMWF ERA5 monthly averaged data",
        "variable_id": nv,
        "grid": "Regridded to regular 1.0° x 1.0° grid (bilinear interpolation)",
        "lat_definition": lat_def,
        "lon_definition": "cell centers at -179.5 to 179.5 (1° spacing)",
        "calendar": "proleptic_gregorian",
        "time_resolution": "monthly (middle of month ~ day 15)",
        "time_coverage": "1950-01 to 2014-12",
        "units": u,
        "units_note": units_note }

    # Salida
    aux = input(f'Dataset final info [y/n]  ')
    if aux == 'y':
        print(ds)

    out_dir = root_data / "data" / "processed" / "era5" / v
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_dir / f"ERA5_{nv}.nc")
