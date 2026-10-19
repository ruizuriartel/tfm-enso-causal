"""
seleccion_modelo_nino34.py
Calcula, para HadISST o para los 7 modelos CMIP6 (histórico 1950-2014), el índice Niño 3.4 (anomalía mensual sin tendencia y versión suavizada con filtro triangular de 5 meses) y el perfil zonal de SST del corredor ecuatorial (5°S-5°N, 150°E-90°W). Son las entradas de las métricas de Planton et al. (2021).
Se ejecuta dos veces, cambiando la variable datos ('hadisst' y 'cmip6').
Entradas: data/processed/hadisst/sea_surface_temperature/HadISST_sst.nc y data/processed/cmip6/sea_surface_temperature/sst_historical_allmodels.nc
Salidas: data/indices/nino34_seleccion_modelo/<modelo>_nino34_sst.nc
Memoria: sección 4.6 y Apéndice A.
"""

import xarray as xr
from pathlib import Path
import numpy as np
from scipy.signal import detrend
import os

# Ejecutar dos veces: primero 'hadisst' y luego 'cmip6'
datos = 'hadisst'

# Cargamos datos
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

if datos == 'hadisst':
    data_dir = root_data / "data" / "processed" / "hadisst" / "sea_surface_temperature"
    filename = data_dir / "HadISST_sst.nc"
    modelos = ["hadisst"]
elif datos == 'cmip6':
    data_dir = root_data / "data" / "processed" / "cmip6" / "sea_surface_temperature"
    filename = data_dir / "sst_historical_allmodels.nc"
    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
ds_entero = xr.open_dataset(filename, engine="netcdf4")

# Analizamos datos de entrada
aux = input('Dataset initial info [y/n]  ')
if aux == 'y':
    print(ds_entero)

# Definimos directorio de salida de todos los indices
out_dir = root_data / "data" / "indices" / "nino34_seleccion_modelo"
out_dir.mkdir(parents=True, exist_ok=True)

for m in modelos:
    print(f"\n----{m}----\n")
    ds = ds_entero.sel(model=m) if m != "hadisst" else ds_entero

    # ÍNDICE NIÑO34
        # Definición geométrica
    weights = np.cos(np.deg2rad(ds.latitude))
    ts = ds.sst.sel(latitude=slice(-5, 5), longitude=slice(-170, -120)).weighted(weights).mean(dim=("latitude", "longitude"))
        # Anomalías
            # eliminamos ciclo estacional
    p_base = slice("1950-01-01", "2014-12-31")
    clim = ts.sel(time=p_base).groupby("time.month").mean("time")
    anomaly = ts.groupby("time.month") - clim
            # eliminamos tendencia lineal
    anomaly = xr.apply_ufunc(detrend, anomaly.dropna(dim="time"),
        input_core_dims=[["time"]], output_core_dims=[["time"]])
        # Suavizado --> media móvil ponderada triangular de 5 meses para la selección de modelos
    pesos = xr.DataArray([1, 2, 3, 2, 1], dims=['window'])
    anomaly_s = anomaly.rolling(time=5, center=True).construct('window').dot(pesos) / 9.0

    # PERFIL ZONAL ECUATORIAL
    ds_eq = ds.sst.sel(latitude=slice(-5, 5)).where((ds.longitude >= 150) | (ds.longitude <= -90), drop=True)
    PZ_eq_tx = ds_eq.weighted(weights).mean(dim=("latitude"))

    # Salida
    ds_index = xr.Dataset(
        data_vars={
            "nino34_mensual": (("time",), anomaly.values.astype(np.float32),
                           {"long_name": "Índice Niño 3.4 bruto mensual", "units": "degC",
                            "description": "Anomalía mensual de SST en región Niño 3.4"}),

            "nino34_smooth": (("time",), anomaly_s.values.astype(np.float32),
                              {"long_name": "Índice Niño 3.4 suavizado", "units": "degC",
                               "description": "Anomalía mensual de SST en la región Niño 3.4, suavizada (filtro triangular de 5 meses)"}),

            "sst_zonal_eq": (("time", "longitude"), PZ_eq_tx.values.astype(np.float32),
                             {"long_name": "Evolución del Perfil Zonal SST del Pacífico Ecuatorial", "units": "degC",
                              "description": "SST promediada en banda ecuatorial (5S-5N) preservando tiempo y longitud"}),
        },
        coords={
            "time": anomaly.time,
            "longitude": PZ_eq_tx.longitude,
        },)

    ds_index.attrs['description'] = "Índice Niño 3.4 (mensual y suavizado) y evolución del perfil zonal del Pacífico Ecuatorial"
    ds_index.attrs['model'] = m
    ds_index.attrs['dataset'] = datos
    ds_index.attrs['climatology_period'] = "1950-2014"

    out_filename = out_dir / f"{m}_nino34_sst.nc"
    ds_index.to_netcdf(out_filename)

    aux = input(f'{m} dataset final info [y/n]  ')
    if aux == 'y':
        print(ds_index)
