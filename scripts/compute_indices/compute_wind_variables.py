"""
compute_wind_variables.py
Descomposición de Helmholtz del viento horizontal global con armónicos esféricos (windspharm): función corriente psi (restando su media zonal) y potencial de velocidad chi, en 200 y 850 hPa, para ERA5 y los 7 modelos CMIP6 (histórico, SSP2-4.5 y SSP5-8.5). Pregunta una sola vez al inicio si se muestran las gráficas y los resúmenes.
Entradas: data/processed/era5/eastward_wind_<nivel>/ERA5_ua.nc y northward_wind_<nivel>/ERA5_va.nc; data/processed/cmip6/eastward_wind_<nivel>/ua_<experimento>_allmodels.nc y northward_wind_<nivel>/va_<experimento>_allmodels.nc
Salidas: data/processed/era5/velocity_potential_<nivel>/ERA5_chi.nc y streamfunction_<nivel>/ERA5_psi.nc; data/processed/cmip6/velocity_potential_<nivel>/chi_<experimento>_allmodels.nc y streamfunction_<nivel>/psi_<experimento>_allmodels.nc
Requiere WSL o Linux. Memoria: sección 4.5.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import gc
from windspharm.xarray import VectorWind


warnings.filterwarnings("ignore")

root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

levels = ["850hPa", "200hPa"]
experimentos = ["historical", "ssp2_4_5", "ssp5_8_5"]

def calc_chi_psi_lib(u, v):
    # Descomposición de Helmholtz del viento horizontal mediante windspharm (armónicos esféricos) sobre el campo global

    u = u.fillna(0.0)
    v = v.fillna(0.0)

    # windspharm requiere que las latitudes estén ordenadas de forma decreciente (de norte a sur); se reordenan si es necesario
    orig_ascending = u.latitude[0] < u.latitude[-1]
    if orig_ascending:
        u = u.sortby('latitude', ascending=False)
        v = v.sortby('latitude', ascending=False)

    w = VectorWind(u, v)
    psi, chi = w.sfvp()     # Función corriente (streamfunction) y potencial de velocidad (velocity potential) global

    if orig_ascending:
        psi = psi.sortby('latitude', ascending=True)
        chi = chi.sortby('latitude', ascending=True)

    # Restamos la media zonal explícitamente de la función corriente
    psi_smz = (psi - psi.mean(dim='longitude')).astype(np.float32).rename("psi")
    chi = chi.astype(np.float32).rename("chi")
    return chi, psi_smz

if __name__ == '__main__':

    mostrar = input('Mostrar gráficas y resúmenes [y/n]  ') == 'y'

    # Calculo de variables de reanálisis - ERA5
    print(f"\nERA5")
    for nv in levels:

        # Abrimos variables de viento
        pre_dir = root_data / "data" / "processed" / "era5"
        u_path = pre_dir / f"eastward_wind_{nv}" / "ERA5_ua.nc"
        v_path = pre_dir / f"northward_wind_{nv}" / "ERA5_va.nc"

        ds_u = xr.open_dataset(u_path, engine="netcdf4")
        ds_v = xr.open_dataset(v_path, engine="netcdf4")
        orig_attrs = ds_u.attrs.copy()

        with tqdm(total=3, desc=f"  {nv}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}') as pbar:
            pbar.set_postfix_str(r"Calculando chi y psi...")
            chi, psi = calc_chi_psi_lib(ds_u.ua, ds_v.va)
            pbar.update(1)

            pbar.set_postfix_str("Limpiando atributos...")
            chi = chi.astype(np.float32).rename("chi")
            psi = psi.astype(np.float32).rename("psi")
            chi.attrs = {"long_name": f"Velocity potential {nv}", "units": "m2 s-1"}
            psi.attrs = {"long_name": f"Streamfunction {nv}", "units": "m2 s-1"}
            pbar.update(1)

            pbar.set_postfix_str("Guardando...")
            out_chi = pre_dir / f"velocity_potential_{nv}"
            out_psi = pre_dir / f"streamfunction_{nv}"
            out_chi.mkdir(parents=True, exist_ok=True)
            out_psi.mkdir(parents=True, exist_ok=True)

            ds_chi_save = xr.Dataset({"chi": chi})
            ds_chi_save.attrs = orig_attrs.copy()
            ds_chi_save.attrs["title"] = f"{orig_attrs.get('title', 'ERA5')} - Velocity Potential {nv}"
            ds_chi_save.attrs["variable_id"] = "chi"
            ds_chi_save.attrs["units"] = "m2 s-1"

            ds_psi_save = xr.Dataset({"psi": psi})
            ds_psi_save.attrs = orig_attrs.copy()
            ds_psi_save.attrs["title"] = f"{orig_attrs.get('title', 'ERA5')} - Streamfunction {nv}"
            ds_psi_save.attrs["variable_id"] = "psi"
            ds_psi_save.attrs["units"] = "m2 s-1"

            ds_chi_save.to_netcdf(out_chi / "ERA5_chi.nc")
            ds_psi_save.to_netcdf(out_psi / "ERA5_psi.nc")
            pbar.update(1)
            pbar.set_postfix_str("¡Finalizado!")

        # Visualización gráfica opcional para el paso temporal t=0
        if mostrar:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            fig.suptitle(rf"ERA5 - {nv} (t=0)", fontsize=14, fontweight='bold')
            chi.isel(time=0).plot(ax=axes[0], cmap='RdBu_r')
            axes[0].set_title(r"$\chi$")
            psi.isel(time=0).plot(ax=axes[1], cmap='RdBu_r')
            axes[1].set_title(r"$\psi$")
            plt.tight_layout(); plt.show()

        # Información resumida por consola de los datasets generados
        if mostrar:
            print(f"    {nv}")
            print('\n   '+r'$\chi$')
            print(ds_chi_save)
            print('\n   '+r'$\psi$')
            print(ds_psi_save)

        ds_u.close()
        ds_v.close()
        del ds_u, ds_v, ds_chi_save, ds_psi_save
        gc.collect()

    # Cálculo de variables de los modelos climáticos - CMIP6
    print(f"\nCMIP6")
    for p in experimentos:
        for nv in levels:
            chi_list = []
            psi_list = []

            # Abrimos variables de viento
            pre_dir = root_data / "data" / "processed" / "cmip6"
            ua_all = pre_dir / f"eastward_wind_{nv}" / f"ua_{p}_allmodels.nc"
            va_all = pre_dir / f"northward_wind_{nv}" / f"va_{p}_allmodels.nc"

            ds_ua_all = xr.open_dataset(ua_all, engine="netcdf4")
            ds_va_all = xr.open_dataset(va_all, engine="netcdf4")
            orig_attrs_cmip6 = ds_ua_all.attrs.copy()

            modelos_ua = list(ds_ua_all.model.values)
            modelos_va = list(ds_va_all.model.values)

            if modelos_ua != modelos_va:
                raise ValueError(f'Error: Orden de modelos distinto entre ua y va en {p} {nv}:\n  ua: {modelos_ua}\n  va: {modelos_va}')

            for m in modelos_ua:
                ds_u = ds_ua_all.sel(model=m)
                ds_v = ds_va_all.sel(model=m)

                with tqdm(total=3, desc=f"  {p} {m} {nv}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}') as pbar:
                    pbar.set_postfix_str(r"Calculando chi y psi...")
                    chi_m, psi_m = calc_chi_psi_lib(ds_u.ua, ds_v.va)
                    pbar.update(1)

                    pbar.set_postfix_str("Limpiando atributos...")
                    chi_m = chi_m.astype(np.float32).rename("chi")
                    psi_m = psi_m.astype(np.float32).rename("psi")
                    chi_m.attrs = {"long_name": f"Velocity potential {nv}", "units": "m2 s-1"}
                    psi_m.attrs = {"long_name": f"Streamfunction {nv}", "units": "m2 s-1"}
                    pbar.update(1)

                    chi_list.append(xr.Dataset({"chi": chi_m}))
                    psi_list.append(xr.Dataset({"psi": psi_m}))
                    pbar.update(1)
                    pbar.set_postfix_str("¡Finalizado!")

                # Visualización gráfica opcional por modelo en t=0
                if mostrar:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                    fig.suptitle(rf"CMIP6 - {m} - {p} - {nv} (t=0)", fontsize=14, fontweight='bold')
                    chi_m.isel(time=0).plot(ax=axes[0], cmap='RdBu_r')
                    axes[0].set_title(r"$\chi$")
                    psi_m.isel(time=0).plot(ax=axes[1], cmap='RdBu_r')
                    axes[1].set_title(r"$\psi$")
                    plt.tight_layout(); plt.show()

                # Información resumida por consola de los datasets generados
                if mostrar:
                    print(f"    {p} {m} {nv}")
                    print('\n   ' + 'chi')
                    print(chi_m)
                    print('\n   ' + 'psi')
                    print(psi_m)

            # Ensamble de todos los modelos climáticos y guardado
            ds_chi = xr.concat(chi_list, dim=pd.Index(modelos_ua, name="model"))
            ds_psi = xr.concat(psi_list, dim=pd.Index(modelos_ua, name="model"))

            ds_chi.attrs = orig_attrs_cmip6.copy()
            ds_chi.attrs["title"] = f"CMIP6 Processed Ensemble ({p}) - Velocity Potential {nv}"
            ds_chi.attrs["variable_id"] = "chi"
            ds_chi.attrs["units"] = "m2 s-1"

            ds_psi.attrs = orig_attrs_cmip6.copy()
            ds_psi.attrs["title"] = f"CMIP6 Processed Ensemble ({p}) - Streamfunction {nv}"
            ds_psi.attrs["variable_id"] = "psi"
            ds_psi.attrs["units"] = "m2 s-1"

            out_chi = pre_dir / f"velocity_potential_{nv}"
            out_psi = pre_dir / f"streamfunction_{nv}"
            out_chi.mkdir(parents=True, exist_ok=True)
            out_psi.mkdir(parents=True, exist_ok=True)

            ds_chi.to_netcdf(out_chi / f"chi_{p}_allmodels.nc")
            ds_psi.to_netcdf(out_psi / f"psi_{p}_allmodels.nc")

            # Información resumida por consola de los ensambles
            if mostrar:
                print(f"    {p} {nv} (ensamble)")
                print('\n   ' + r'$\chi$')
                print(ds_chi)
                print('\n   ' + r'$\psi$')
                print(ds_psi)

            ds_ua_all.close()
            ds_va_all.close()
            del ds_ua_all, ds_va_all, ds_chi, ds_psi
            gc.collect()