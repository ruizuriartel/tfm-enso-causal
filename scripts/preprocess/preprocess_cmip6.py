"""
preprocess_cmip6.py
Preprocesa los 7 modelos CMIP6 en los tres experimentos: nombres estándar, comprobación de unidades (error si no coinciden), calendario proleptic_gregorian, interpolación bilineal (xesmf) a 1° y recorte a 60°S-60°N en SST, OLR, PSL y PR. Concatena los modelos en un NetCDF por variable y experimento.
Entradas: data/raw/cmip6/<variable>/<modelo>/<experimento>/*.nc (CDS y Pangeo)
Salidas: data/processed/cmip6/<variable>/<nv>_<experimento>_allmodels.nc
Memoria: sección 4.4.
"""

import xarray as xr
import pandas as pd
import numpy as np
import xesmf as xe
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings("ignore")

root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
experimentos = ["historical", "ssp2_4_5", "ssp5_8_5"]
variables = ["sea_surface_temperature", "eastward_wind_200hPa", "eastward_wind_850hPa", "northward_wind_200hPa",
             "northward_wind_850hPa", "geopotential_height_200hPa", "precipitation",
             "toa_outgoing_longwave_radiation", "sea_level_pressure", "near_surface_air_temperature"]
units = ["degC", "m s-1", "m s-1", "m s-1", "m s-1", "m", "kg m-2 s-1", "W m-2", "Pa", "K"]
v_u = dict(zip(variables, units))

target_grid = xr.Dataset({
    "lat": (["lat"], np.arange(-89.5, 90.5, 1.0)),
    "lon": (["lon"], np.arange(-179.5, 180.5, 1.0))})

for p in experimentos:
    print(f"\n----{p}----\n")
    if p == "historical": start_year, end_year = "1950", "2014"
    else: start_year, end_year = "2015", "2100"
    for v, u in v_u.items():
        print(f"\n  - {v}\n")
        ds_list = []
        modelos_procesados = []
        for m in modelos:
            print(f"\n       - {m}\n")

            # Cargamos datos
            model_dir = root_data / "data" / "raw" / "cmip6" / v / m / p
            archivos_nc = sorted(list(model_dir.glob("*.nc")))
            if not archivos_nc:
                print(f"No se ha encontrado archivo para: {m} - {v} - {p}"); continue
            filename = archivos_nc[0]
            ds = xr.open_dataset(filename, engine="netcdf4")

            # Analizamos datos
            aux = input(f'{m} dataset initial info [y/n]  ')
            nv = ds.attrs['variable_id']
            lat_name = [c for c in ds.coords if 'lat' in c.lower()][0]
            lon_name = [c for c in ds.coords if 'lon' in c.lower()][0]
            if aux == 'y':
                print(ds)
                print("Fecha fin:", ds.time.values[-1])
                print(ds[nv].attrs)
                print(ds.time.encoding)
                print("Latitudes:",  ds[lat_name].min().item(), ds[lat_name].max().item())
                print("Longitudes:", ds[lon_name].min().item(), ds[lon_name].max().item())

            # Pasos: renombrado de variables, analisis de unidades, eliminar dimensiones de presión o altura,
            #        eje temporal común, rejilla común, limpieza del ds, ejecucion final.
            with tqdm(total=10, desc=f"Procesando", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}') as pbar:

                # Renombramos variable a nombre estándar del proyecto
                pbar.set_postfix_str("Preparando: Nombres")
                if nv == 'tos':
                    ds = ds.rename({'tos': 'sst'})        # Solo la SST necesita renombrarse; el resto ya usa el nombre CMIP6
                    nv = 'sst'
                pbar.update(1)

                # Análisis de unidades
                pbar.set_postfix_str("Preparando: Unidades")
                current_units = ds[nv].attrs.get('units')
                if current_units != u:
                    raise ValueError(f"La variable {v} del modelo {m} no tiene las unidades correctas ({current_units}).")
                if v == "near_surface_air_temperature":
                    with xr.set_options(keep_attrs=True):
                        ds[nv] = ds[nv] - 273.15
                    ds[nv].attrs['units'] = 'degC'
                pbar.update(1)

                # Eliminamos la dimension de presion y altura
                pbar.set_postfix_str("Preparando: Dimensiones")
                dims_to_remove = ['plev', 'height']
                for dim in dims_to_remove:
                    if dim in ds.dims:
                        ds = ds.squeeze(dim, drop=True)
                    elif dim in ds.coords:
                        ds = ds.drop_vars(dim)
                pbar.update(1)

                # Periodo temporal común y normalización de calendario
                pbar.set_postfix_str("Preparando: Calendario")
                ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False, align_on="year")
                ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))
                new_timestamps = pd.date_range(start=f"{start_year}-01-01", periods=ds.sizes["time"], freq="MS") + pd.Timedelta(days=14)
                ds = ds.assign_coords(time=new_timestamps.normalize())
                pbar.update(1)

                # Rejilla común (también pasamos de 0-360º a -180-180º)
                pbar.set_postfix_str("Preparando: Pesos Regrid ")
                lat_name = [c for c in ds.coords if 'lat' in c.lower()][0]
                lon_name = [c for c in ds.coords if 'lon' in c.lower()][0]
                weights_file = root_data / "data" / "processed" / "cmip6" / v
                if 'pangeo' in filename.name: weights_file = weights_file / f"{m}_pangeo_weights.nc"
                else: weights_file = weights_file / f"{m}_weights.nc"
                weights_file.parent.mkdir(parents=True, exist_ok=True)
                if lat_name == 'lat' and lon_name == 'lon':
                    ds_in = ds
                else:
                    dims = ds[lat_name].dims
                    ds_in = ds.assign_coords(
                        lat=(dims, ds[lat_name].values),
                        lon=(dims, ds[lon_name].values) )
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
                plt.title(f"{p} - {m} - {nv} - {v}")
                plt.show()

            # Salida
            aux = input(f'{m} dataset final info [y/n]  ')
            if aux == 'y': print(ds)
            ds_list.append(ds); modelos_procesados.append(m)

        # Ensamble: concatenar todos los modelos en dimensión 'model'
        ds_ensemble = xr.concat(ds_list, dim=pd.Index(modelos_procesados, name="model"))
        modelos_faltantes = [m for m in modelos if m not in modelos_procesados]
        str_presentes = ", ".join(modelos_procesados)
        str_faltantes = ", ".join(modelos_faltantes) if modelos_faltantes else "None"
        ds_ensemble.attrs = {
            "title": f"CMIP6 Processed Ensemble ({p}) - {v.replace('_', ' ').title()}",
            "source": f"Ensemble of {len(modelos_procesados)} CMIP6 models",
            "included_models": str_presentes,
            "missing_models": str_faltantes,
            "experiment_id": p,
            "variable_id": nv,
            "grid": "Regridded to regular 1.0° x 1.0° grid (bilinear interpolation)",
            "lat_definition": "cell centers at -89.5 to 89.5 (1° spacing)" if (v == "near_surface_air_temperature" or "geopotential" in v or  "wind" in v) else "cell centers at -59.5 to 59.5 (1° spacing)",
            "lon_definition": "cell centers at -179.5 to 179.5 (1° spacing)",
            "calendar": "proleptic_gregorian",
            "time_resolution": "monthly (middle of month ~ day 15)",
            "time_coverage": f"{start_year}-01 to {end_year}-12",
            "units": "degC" if (nv == "tas") else u}
        if v == "near_surface_air_temperature":
            ds_ensemble.attrs["units_note"] = "Temperature converted to degC"

        # Salida final
        aux = input(f'{v} dataset FINAL info [y/n]  ')
        if aux == 'y':
            print(ds_ensemble)

        out_dir = root_data / "data" / "processed" / "cmip6" / v
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_ensemble.to_netcdf(out_dir / f"{nv}_{p}_allmodels.nc")
