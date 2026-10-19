"""
download_pangeo.py
Completa con el catálogo Pangeo (acceso anónimo) las cinco variables que CDS no ofrece: Z en ssp245 y viento en ssp585 para EC-Earth3-CC, y viento en ssp245 para HadGEM3-GC31-LL.
Entradas: ficheros ya descargados por download_cmip6.py (de ellos toma el variant_label) y catálogo de Pangeo.
Salidas: data/raw/cmip6/<variable>_<nivel>hPa/<modelo>/<experimento>/<modelo>_pangeo_*.nc
Requiere WSL o Linux. Memoria: sección 4.3.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import sys
import asyncio
from dask.diagnostics import ProgressBar

# Evitamos errores con asyncio
if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Versiones de los conjuntos de Pangeo usadas en la memoria (sección 4.3)
versiones_pangeo = {
    ("EC-Earth3-CC", "ssp2_4_5", "zg"): "20210113",
    ("EC-Earth3-CC", "ssp5_8_5", "ua"): "20210218",
    ("EC-Earth3-CC", "ssp5_8_5", "va"): "20210113",
    ("HadGEM3-GC31-LL", "ssp2_4_5", "ua"): "20190908",
    ("HadGEM3-GC31-LL", "ssp2_4_5", "va"): "20190908",
}

# Definimos directorio de salida
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

# Catalogo Pangeo
catalogo = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"
cat = pd.read_csv(catalogo)

def descarga(modelo, experimento, variable, miembro):

    # Definimos los niveles de presión y el nombre estándar de la variable climática
    if variable == "zg":
        niveles = [200]; nv = "geopotential_height"
    elif variable == "ua":
        niveles = [200, 850]; nv = "eastward_wind"
    elif variable == "va":
        niveles = [200, 850]; nv = "northward_wind"

    # Consultamos si la variable esta en el catálogo de Pangeo CMIP6
    v_pangeo = cat[
        (cat.source_id == modelo) &
        (cat.experiment_id == experimento.replace("_", "")) &
        (cat.variable_id == variable) &
        (cat.table_id == "Amon") &           # Amon -> Atmospheric Monthly
        (cat.member_id == miembro)]
    if v_pangeo.empty:
        print(f"\n{modelo} - {experimento} - {variable} - {miembro} NO disponible")
        return
    print(f"\n{modelo} - {experimento} - {variable} - {miembro} disponible")

    # Selección de rejilla
    rejillas = sorted(v_pangeo.grid_label.unique())
    if "gr" in rejillas:                                     # gr == rejilla regrideada al estandar del modelo
        v_pangeo = v_pangeo[v_pangeo.grid_label == "gr"]     # priorizamos gr para mantener coherencia con variables de CDS
    else:
        v_pangeo = v_pangeo[v_pangeo.grid_label == rejillas[0]]
    grid = v_pangeo.iloc[0].grid_label

    # Fijamos la versión usada en la memoria para que la descarga sea reproducible
    version = versiones_pangeo[(modelo, experimento, variable)]
    v_pangeo = v_pangeo[v_pangeo.version.astype(str) == version]
    if v_pangeo.empty:
        print(f"\n{modelo} - {experimento} - {variable} - versión {version} NO disponible")
        return
    zstore = v_pangeo.iloc[0].zstore
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds0 = xr.open_dataset(
        zstore,
        engine="zarr",
        backend_kwargs={"consolidated": True},
        storage_options={"token": "anon"},
        decode_times=time_coder,
        chunks={})
    cal = ds0.time.encoding.get("calendar")

    for l in niveles:
        m = modelo.lower().replace("-", "_")

        # Definimos lugar y nombre de descarga
        out_dir = root_data / "data" / "raw" / "cmip6" / f"{nv}_{l}hPa" / m / experimento
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{m}_pangeo_{experimento}_{nv}_{l}hPa.nc"

        try:
            # Selección del nivel de presion
            ds = ds0.sel(plev=float(l * 100), method="nearest")

            # Selección ventana temporal
            ds = ds.sel(time=slice("2015", "2100"))

            # Limpieza de coordenadas auxiliares
            ds = ds[[variable]]
            extra = [c for c in ds.coords if c not in ["time", "lat", "lon", "plev"]]
            if extra:
                ds = ds.drop_vars(extra)

            # Asignamos atributos que preprocess_cmip6.py necesita leer
            ds.attrs["variable_id"] = variable
            ds.attrs["source_id"] = modelo
            ds.attrs["experiment_id"] = experimento.replace("_", "")
            ds.attrs["variant_label"] = miembro
            ds.attrs["grid_label"] = grid
            ds.attrs["source"] = "Pangeo CMIP6"

            # Descarga
            print(f"  [{l} hPa] descargando ~{ds[variable].nbytes / 1e6:.0f} MB...", flush=True)
            with ProgressBar(): ds = ds.load()
            ds[variable] = ds[variable].astype(np.float32)
            if cal:
                ds.time.encoding.update({"calendar": cal, "units": "days since 2015-01-01"})
            ds.to_netcdf(out_file)
            print(ds)
            print(f"  calendario: {cal},  grid_label: {grid},  unidades: {ds[variable].attrs.get('units')}")

        except Exception as err:
            print(f"  [{l} hPa] ERROR: {err}")

if __name__ == '__main__':

    # Variables a descargar
    faltantes = [("EC-Earth3-CC", "ssp2_4_5", "zg"),
                 ("EC-Earth3-CC", "ssp5_8_5", "ua"),
                 ("EC-Earth3-CC", "ssp5_8_5", "va"),
                 ("HadGEM3-GC31-LL", "ssp2_4_5", "ua"),
                 ("HadGEM3-GC31-LL", "ssp2_4_5", "va")]

    for m, e, v in faltantes:
        mod = m.lower().replace("-", "_")
        archivos = sorted((root_data / "data" / "raw" / "cmip6").rglob(f"*/{mod}/{e}/*.nc"))
        a = xr.open_dataset(archivos[0], decode_times=False).attrs
        miembro = a.get("variant_label", a.get("member_id"))       # member_id para mantener coherencia con variables de CDS
        descarga(m, e, v, miembro)

