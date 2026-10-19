"""
diagnostico_climatologia.py
Compara el sesgo climatológico entre el histórico (base fija 1981-2010) y cada ventana de escenario (base propia) con la desviación típica interna de cada nodo (con ddof=1), para ACCESS-CM2, HadGEM3-GC31-LL y NorESM2-MM.
Entradas: data/processed/cmip6/, data/indices/solvers_obs/mask_land.nc y data/indices/nodos_obs/nodos_finales.npy (de reduccion_nodos.py)
Salidas: resultados/resumen_diagnostico_climatologia.csv
Memoria: sección 5.1 y Apéndice B (Tabla B.1).
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
from pathlib import Path

import compute_nodos_obs as cno

def carga_dominio(pro_dir, folder, escenario, modelo, lat, lon, mla=None):
    # Calcula la serie 1D del dominio de un nodo: media espacial ponderada por sqrt(cos(lat)), solo sobre tierra si el nodo es terrestre

    # Abrimos el campo del modelo y escenario (el nombre de la variable sale del nombre del archivo)
    path = list((pro_dir / folder).glob(f"*{escenario}_allmodels.nc"))[0]
    nv = path.name.split('_')[0]
    campo = xr.open_dataset(path, engine="netcdf4")[nv].sel(model=modelo)

    # Longitudes de 0° a 360° y recorte al dominio del nodo (la longitud solo si el nodo la define)
    campo = campo.assign_coords(longitude=((campo.longitude + 360) % 360)).sortby("longitude")
    campo = campo.sel(latitude=lat)
    if lon is not None:
        campo = campo.sel(longitude=lon)

    # Nodos terrestres: nos quedamos solo con los puntos de tierra
    if mla is not None:
        mla_dom = mla.assign_coords(longitude=((mla.longitude + 360) % 360)).sortby("longitude")
        mla_dom = mla_dom.sel(latitude=lat)
        if lon is not None: mla_dom = mla_dom.sel(longitude=lon)
        campo = campo.where(mla_dom)

    # Media espacial ponderada: queda una serie 1D en el tiempo
    pesos = np.sqrt(np.cos(np.deg2rad(campo.latitude))).clip(0)
    return campo.weighted(pesos.fillna(0)).mean(dim=[d for d in campo.dims if d != "time"],skipna=True)          # serie 1D, promedio espacial ponderado

if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    pro_dir = root_data / "data" / "processed" / "cmip6"
    res_dir = root / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = root_data / "data" / "indices" / "nodos_obs"
    solv_dir = root_data / "data" / "indices" / "solvers_obs"

    # Cargamos los nodos finales seleccionados tras el proceso de reducción
    nodos_finales = list(np.load(obs_dir / "nodos_finales.npy", allow_pickle=True))

    # Cargamos máscara de tierra y nodos
    mask_land = xr.open_dataarray(solv_dir / "mask_land.nc")
    nodos = cno.get_nodos(mask_land)

    # Definimos modelos CMIP6 y periodos temporales
    meses_ventana = [10, 11, 12, 1, 2, 3]
    modelos = ["access_cm2", "hadgem3_gc31_ll", "noresm2_mm"]   # 3 modelos representativos del ensemble
    p_base_fijo = slice("1981-01-01", "2010-12-31")             # climatología de referencia del histórico
    periodos_esc = {"W_ssp245": ("ssp2_4_5", slice("2050-01-01", "2099-12-31")),            # ventanas de los escenarios,
                     "W_ssp585": ("ssp5_8_5", slice("2050-01-01", "2099-12-31"))}           # cada una con su propia climatología

    # Bucle principal iterando sobre cada nodo y cada modelo
    print(f"\n----DIAGNOSTICO DE ANOMALÍAS----\n")
    filas = []
    for nodo in nodos_finales:
        base = nodo.replace('_PC2','')
        print(base)

        if base == 'N34':
            folder = 'sea_surface_temperature'
            lat = slice(-5, 5)
            lon = slice(190, 240)
            mla = None

        else:
            datos = nodos.get(base, {})
            folder = datos['folder']
            dom = datos['dom']
            lat = dom['lat']
            lon = dom.get('lon')
            mla = dom.get('mla')

        for m in modelos:
            print(f"    {m}")

            # Anomalías del histórico respecto a la climatología fija 1981-2010
            serie_h = carga_dominio(pro_dir, folder, "historical", m, lat, lon, mla=mla)
            clim_ref = serie_h.sel(time=p_base_fijo).groupby("time.month").mean("time")
            anom_h = serie_h.sel(time=p_base_fijo).groupby("time.month") - clim_ref

            # std interna (ddof=1) restringida a los 6 meses de la ventana -> "ruido normal" del nodo dentro de oct-mar
            std_ref = float(anom_h.where(anom_h.time.dt.month.isin(meses_ventana), drop=True).std(ddof=1))

            # Comparamos con la climatología propia de cada ventana de escenario
            for esc, (p, p_base_esc) in periodos_esc.items():

                serie_e = carga_dominio(pro_dir, folder, p, m, lat, lon, mla=mla)
                clim_esc = serie_e.sel(time=p_base_esc).groupby("time.month").mean("time")  # climatología mensual del escenario

                # Los seis sesgos mensuales de la ventana oct-mar
                diff_ventana = (clim_esc - clim_ref).sel(month=meses_ventana).values
                    # A cada uno de los 6 le restamos la media de los 6 y calculamos su rango
                residuo = diff_ventana - diff_ventana.mean()
                rango_residuo = float(np.max(residuo) - np.min(residuo))
                    # Analizamos ese rango en unidades de la desviación típica interna
                ratio_rango_residuo = rango_residuo / std_ref if std_ref > 0 else np.nan

                filas.append({"nodo": nodo,
                              "modelo": m,
                              "escenario": esc,
                              "diff_media": round(float(np.mean(diff_ventana)), 5),     # sesgo medio de la ventana
                              "rango_residuo": round(rango_residuo, 5),                 # rango del sesgo mensual sin su media
                              "std_ref": round(std_ref, 5),
                              "ratio_rango_residuo": round(ratio_rango_residuo, 5)      # rango_residuo en unidades de std_ref
                              })

    # Guardamos y mostramos resultados
    df = pd.DataFrame(filas).sort_values("ratio_rango_residuo", ascending=False)
    df.to_csv(res_dir / "resumen_diagnostico_climatologia.csv", index=False)

    print(f"\n----RESULTADOS----\n")
    print(df.to_string(index=False))