"""
download_gpcc.py
Descarga y concatena la precipitación mensual de GPCC Precipitation Analysis v2025 (1°, 1946-2015) desde el repositorio abierto del DWD. https://opendata.dwd.de/climate_environment/GPCC/
                                                                                                                                         https://opendata.dwd.de/climate_environment/GPCC/html/gpcc_precipitation_analysis_climatology_v2025_doi_download.html
                                                                                                                                         https://climatedataguide.ucar.edu/climate-data/gpcc-global-precipitation-climatology-centre
Entradas: ninguna (acceso directo).
Salidas: data/raw/gpcc/precipitation/gpcc_monthly_1946_2015.nc
Memoria: sección 4.2.
"""

from pathlib import Path
import requests
import gzip
import shutil
import os
import xarray as xr
import pandas as pd

# Definimos directorio de salida
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

data_dir = root_data / "data" / "raw" / "gpcc" / "precipitation"
data_dir.mkdir(parents=True, exist_ok=True)

root_download = "https://opendata.dwd.de/climate_environment/GPCC/GPCC_Precipitation_Analysis_Monthly/Version_2025/100_degree/"

datasets = []
for start in range(1946, 2016, 5):
    end = start + 4

    url = root_download + f"gpcc_precipitation_analysis_monthly_{start}_{end}_v2025_100.nc.gz"
    gz_filename = data_dir / f"gpcc_precipitation_analysis_monthly_{start}_{end}_v2025_100.nc.gz"
    nc_filename = data_dir / f"gpcc_precipitation_analysis_monthly_{start}_{end}_v2025_100.nc"

    # Descarga
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(gz_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Descomprimir
    with gzip.open(gz_filename, 'rb') as f_in:
        with open(nc_filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Abrimos datasets
    ds = xr.open_dataset(nc_filename, engine="netcdf4", decode_times=False)
    ds.load()
    ds_c = ds.copy()
    ds.close()
    datasets.append(ds_c)

    # Borramos archivos
    os.remove(gz_filename)
    os.remove(nc_filename)

# Concatenamos datasets
ds_all = xr.concat(datasets, dim="time")

# Los ficheros se abren sin decodificar fechas y se sustituye el eje por uno mensual desde 1946-01 (840 pasos, 14 ficheros de 5 años)
ds_all["time"] = pd.date_range(start="1946-01-01", periods=len(ds_all.time), freq="MS")

# Dataset final
output_file = data_dir / "gpcc_monthly_1946_2015.nc"
ds_all.to_netcdf(output_file)
