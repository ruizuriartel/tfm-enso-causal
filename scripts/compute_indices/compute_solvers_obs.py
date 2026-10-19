"""
compute_solvers_obs.py
Máscara de tierra (Natural Earth, rejilla común de 1°) y solvers EOF observacionales de las 11 variables candidatas (SST, OLR, PSL, PSI y CHI en 200 y 850 hPa, Z200 en PNA y PSA, PR_land y TAS_land), con las anomalías de octubre a marzo. Las funciones preproceso_pcs y calc_solver las reutilizan casi todos los demás scripts de la carpeta. Pregunta por pantalla antes de mostrar la máscara de tierra.
Entradas: data/processed/<hadisst|era5|gpcc|cru>/<variable>/ (psi y chi, de compute_wind_variables.py)
Salidas: data/indices/solvers_obs/mask_land.nc y solver_<variable>.pkl
Memoria: sección 5.2.
"""

import xarray as xr
import numpy as np
from eofs.xarray import Eof
import warnings
import os
from pathlib import Path
import pickle
import regionmask
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def build_mask_land(campo):
    # Genera una máscara binaria terrestre utilizando Natural Earth a partir de las coordenadas de un campo o rejilla dada

    lats = campo.latitude.values
    lons = campo.longitude.values
    land = regionmask.defined_regions.natural_earth_v5_1_2.land_110

    mask_land = (land.mask(lons, lats) == 0).rename({"lon": "longitude", "lat": "latitude"})
    return mask_land

def preproceso_pcs(var, lat='all', lon='all', mla=None, moc=False, rolling=False, p_base=slice("1981-01-01", "2010-12-31")):
    # Preprocesamiento de campos climáticos (partiendo de una rejilla común de 1x1°) para análisis EOF

    # Normalización de longitudes de 0° a 360° y reordenamiento espacial
    var = var.assign_coords(longitude=((var.longitude + 360) % 360)).sortby("longitude")
    if lat != 'all':
        var = var.sel(latitude=lat)
    if lon != 'all':
        var = var.sel(longitude=lon)

    # Cálculo de anomalías mensuales respecto al periodo base (estándar de la OMM (1981-2010) en histórico, la propia ventana en escenarios)
    clim = var.sel(time=p_base).groupby("time.month").mean("time")
    anom = var.groupby("time.month") - clim

    # Suavizado temporal opcional mediante media móvil trimestral centrada
    if rolling:
            anom = anom.rolling(time=3, center=True).mean().dropna(dim="time", how="all")

    # Aplicación de máscaras oceánicas o terrestres según corresponda
    if moc:
        anom = anom.where(anom.notnull().all(dim="time"))
    if mla is not None:
        mla = mla.assign_coords(longitude=((mla.longitude + 360) % 360)).sortby("longitude")
        anom = anom.where(mla)

    # Ponderación espacial por la raíz cuadrada del coseno de la latitud para equilibrar el área de las celdas
    weights = np.sqrt(np.cos(np.deg2rad(anom.latitude))).clip(0)
    return anom * weights

def calc_solver(anom_dom):
    # Cálculo del solver EOF con anomalías filtradas estacionalmente para retener únicamente los meses de octubre a marzo
    meses_validos = [10, 11, 12, 1, 2, 3]
    anom_de = anom_dom.where(anom_dom["time.month"].isin(meses_validos), drop=True)
    return Eof(anom_de, center=True)

if __name__ == '__main__':

    # Directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    pro_dir = root_data / "data" / "processed"
    solv_dir = root_data / "data" / "indices" / "solvers_obs"
    solv_dir.mkdir(parents=True, exist_ok=True)

    solvers={}

    # Generación y almacenamiento de la máscara de tierra basada en la rejilla objetivo común
        # misma rejilla común 1x1 que target_grid en preprocess_*.py, con los nombres ya renombrados (latitude/longitude)
    target_grid = xr.Dataset({
        "latitude": (["latitude"], np.arange(-89.5, 90.5, 1.0)),
        "longitude": (["longitude"], np.arange(-179.5, 180.5, 1.0))
    })
    mask_land = build_mask_land(target_grid)
    mask_land.to_netcdf(solv_dir / "mask_land.nc")

    # Visualización gráfica de comprobación de la máscara de tierra
    aux = input('Mostrar máscara de tierra [y/n]  ')
    if aux == 'y':
        plt.figure(figsize=(10, 5))
        mask_land.plot()
        plt.title("Máscara de tierra a partir de regionmask.defined_regions.natural_earth_v5_1_2.land_110 a target grid")
        plt.show()

    # Cálculo de los solvers para todas las variables observacionales y de reanálisis

        # SST - HadISST     (Temperatura superficial del mar)
    print('\nSST')
    campos = xr.open_dataset(pro_dir / "hadisst" / "sea_surface_temperature" / "HadISST_sst.nc", engine="netcdf4")
    ssta_dom = preproceso_pcs(campos.sst, lat=slice(-20, 20), lon=slice((120+360)%360, (-80+360)%360), moc=True)
    solvers['SST'] = calc_solver(ssta_dom)

        # OLR - ERA5        (Radiación de onda larga saliente en el tope de la atmósfera)
    print('OLR')
    campos = xr.open_dataset(pro_dir / "era5" / "toa_outgoing_longwave_radiation" / "ERA5_rlut.nc", engine="netcdf4")
    rluta_dom = preproceso_pcs(campos.rlut, lat=slice(-30, 30))
    solvers['OLR'] = calc_solver(rluta_dom)

        # PSL - ERA5        (Presión a nivel del mar)
    print('PSL')
    campos = xr.open_dataset(pro_dir / "era5" / "sea_level_pressure" / "ERA5_psl.nc", engine="netcdf4")
    psla_dom = preproceso_pcs(campos.psl, lat=slice(-20, 20), lon=slice((80+360)%360, (-80+360)%360))
    solvers['PSL'] = calc_solver(psla_dom)

        # PSI y CHI - ERA5   (Función corriente y potencial de velocidad por niveles de presión)
    for l in ['200', '850']:
        print(f"PSI{l}")
        campos_psi = xr.open_dataset(pro_dir / "era5" / f"streamfunction_{l}hPa" / "ERA5_psi.nc", engine="netcdf4")
        solvers[f"PSI{l}"] = calc_solver(preproceso_pcs(campos_psi.psi, lat=slice(-30, 30)))

        print(f"CHI{l}")
        campos_chi = xr.open_dataset(pro_dir / "era5" / f"velocity_potential_{l}hPa" / "ERA5_chi.nc", engine="netcdf4")
        solvers[f"CHI{l}"] = calc_solver(preproceso_pcs(campos_chi.chi, lat=slice(-30, 30)))

        # Z200 (PNA y PSA) - ERA5    (Altura geopotencial para análisis de patrones PNA y PSA)
    l = '200'
    campos = xr.open_dataset(pro_dir / "era5" / f"geopotential_height_{l}hPa" / "ERA5_zg.nc", engine="netcdf4")
    for region, lat_r in [('PNA', [20, 70]), ('PSA', [-70, -20])]:
        print(f"Z{l}_{region}")
        zg_dom = preproceso_pcs(campos.zg, lat=slice(lat_r[0], lat_r[1]), lon=slice((120+360)%360, (-60+360)%360))
        solvers[f"Z{l}_{region}"] = calc_solver(zg_dom)

        # PR_land - GPCC    (Precipitación terrestre)
    print('PR_land')
    campos = xr.open_dataset(pro_dir / "gpcc" / "precipitation" / "GPCC_pr.nc", engine="netcdf4")
    solvers['PR_land'] = calc_solver(preproceso_pcs(campos.pr, lat=slice(-30, 30), mla=mask_land))

        # TAS_land - CRU    (Temperatura superficial del aire sobre tierra)
    print('TAS_land')
    campos = xr.open_dataset(pro_dir / "cru" / "near_surface_air_temperature" / "CRU_tas.nc", engine="netcdf4")
    solvers['TAS_land'] = calc_solver(preproceso_pcs(campos.tas, lat=slice(20, 70), mla=mask_land)) # 20-70N: huella térmica del PNA sobre Norteamérica

    # Guardamos todos los objetos solver calculados
    for nombre, solver in solvers.items():
        with open(solv_dir / f"solver_{nombre}.pkl", "wb") as f: pickle.dump(solver, f)

