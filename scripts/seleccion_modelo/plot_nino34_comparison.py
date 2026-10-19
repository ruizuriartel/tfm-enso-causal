"""
plot_nino34_comparison.py
Calcula las cuatro métricas de Planton et al. (2021) (ENSO_amplitude, ENSO_seasonality, ENSO_pattern, eq_SST_bias) de cada modelo frente a HadISST (ENSO_pattern se evalúa solo en los diciembres, como en el artículo), y dibuja el índice Niño 3.4 de cada uno junto al observado y el perfil zonal de SST del corredor ecuatorial.
Entradas: data/indices/nino34_seleccion_modelo/<modelo>_nino34_sst.nc (de seleccion_modelo_nino34.py)
Salidas: figures/seleccion_modelo/comp_nino34_hadisst_vs_<modelo>.png, comp_zonal_todos_los_modelos.png y resultados/resumen_metricas_planton.csv
Memoria: sección 4.6 y Apéndice A (Tabla A.1, Figuras A.1 y A.2).
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import pandas as pd

# Modelos y sus nombres oficiales (para las leyendas)
modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
nombres_oficiales = {"ec_earth3_cc": "EC-Earth3-CC", "access_cm2": "ACCESS-CM2", "cnrm_esm2_1": "CNRM-ESM2-1", "hadgem3_gc31_ll": "HadGEM3-GC31-LL", "miroc6": "MIROC6", "noresm2_mm": "NorESM2-MM", "ukesm1_0_ll": "UKESM1-0-LL"}

def ratio_seasonality(ts):                                           # ddof=1 -> dividir entre N-1 (varianza)
    std_ndj = ts.sel(time=ts['time.month'].isin([11, 12, 1])).std(ddof=1).item()
    std_mam = ts.sel(time=ts['time.month'].isin([3, 4, 5])).std(ddof=1).item()
    return std_ndj / std_mam if std_mam != 0 else np.nan

def regresion_OLS(x, y):   # Ordinary Least Squares por longitud
    cov = xr.cov(x, y, dim='time', ddof=1)
    var = x.var('time', ddof=1)
    return cov / var

def solo_diciembre(x, y):
    # ENSO_pattern (Planton et al., 2021): solo se evalúan los diciembres
    x = x.sel(time=x['time.month'] == 12).dropna('time')
    return x, y.sel(time=x.time)

def rmse(mod, had):
    return np.sqrt(((mod - had) ** 2).mean()).item()

def a_0_360(da):
    # Pasa la longitud de -180..180 a 0..360 y ordena, para que el corredor 150E-90W sea continuo
    return da.assign_coords(longitude=(da.longitude % 360)).sortby('longitude')


if __name__ == "__main__":

    # Cargamos datos
    root_fig = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root_fig))
    if not root_data.exists(): root_data = root_fig

    perfiles_modelos = {}; resultados = {}

    data_dir = root_data / "data" / "indices" / "nino34_seleccion_modelo"
    ds_had = xr.open_dataset(data_dir / "hadisst_nino34_sst.nc")
    ts_had_m = ds_had.nino34_mensual
    ts_had_s = ds_had.nino34_smooth
    sst_eq_h = ds_had.sst_zonal_eq

    # Cálculos sobre HadISST y funciones comunes para las métricas
        # ENSO_amplitude
    std_had = ts_had_m.std(ddof=1).item()
        # ENSO_seasonality
    seas_had = ratio_seasonality(ts_had_m)
        # ENSO_pattern (solo diciembres)
    clim_h = sst_eq_h.groupby("time.month").mean("time")
    ssta_eq_h = sst_eq_h.groupby("time.month") - clim_h
    x_had, y_had = solo_diciembre(ts_had_s, ssta_eq_h)
    beta_had = regresion_OLS(x_had, y_had)
    max_beta_had = beta_had.max().item()
        # eq_SST_bias
    perfil_medio_h = sst_eq_h.mean(dim=("time"))

    # Guardamos
    resultados["hadisst"] = {"modelo": "HadISST (obs.)", "amplitude_pct": np.nan, "sigma_mod": std_had,
                             "seasonality_pct": np.nan, "R_mod": seas_had, "pattern_rmse": np.nan,
                             "beta_max_mod": max_beta_had, "eq_sst_bias": np.nan}

    # Definimos directorio de salida de todas las figuras
    fig_dir = root_fig / "figures" / "seleccion_modelo"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print('\n')
    for m in modelos:
        ds_cmi = xr.open_dataset(data_dir / f"{m}_nino34_sst.nc")
        ts_cmi_m = ds_cmi.nino34_mensual
        ts_cmi_s = ds_cmi.nino34_smooth
        sst_eq_c = ds_cmi.sst_zonal_eq

        # Calculamos las métricas (estándar de Planton et al., 2021, CLIVAR)
            # ENSO_amplitude
        std_cmi = ts_cmi_m.std(ddof=1).item()
        enso_amplitude = (abs((std_cmi - std_had) / std_had) * 100)
            # ENSO_seasonality
        seas_cmi = ratio_seasonality(ts_cmi_m)
        enso_seasonality = abs((seas_cmi - seas_had) / seas_had) * 100
            # ENSO_pattern (solo diciembres)
        clim_c = sst_eq_c.groupby("time.month").mean("time")
        ssta_eq_c = sst_eq_c.groupby("time.month") - clim_c
        x_cmi, y_cmi = solo_diciembre(ts_cmi_s, ssta_eq_c)
        beta_cmi = regresion_OLS(x_cmi, y_cmi)
        enso_pattern = rmse(beta_cmi, beta_had)
        max_beta_cmi = beta_cmi.max().item()
            # eq_SST_bias
        perfil_medio_c = sst_eq_c.mean(dim=("time"))
        perfiles_modelos[m] = perfil_medio_c
        eq_sst_bias = rmse(perfil_medio_c, perfil_medio_h)

        # Guardamos resultados
        resultados[m] = {"modelo": nombres_oficiales[m],
                         "amplitude_pct": enso_amplitude, "sigma_mod": std_cmi,
                         "seasonality_pct": enso_seasonality, "R_mod": seas_cmi,
                         "pattern_rmse": enso_pattern, "beta_max_mod": max_beta_cmi,
                         "eq_sst_bias": eq_sst_bias}

        # Graficamos
        plt.figure(figsize=(11, 4))
        plt.axhline(0, color='gray', alpha=0.5)
        plt.plot(ts_had_m.time, ts_had_m, label='HadISST', color='black', linewidth=1)
        plt.plot(ts_cmi_m.time, ts_cmi_m, label=nombres_oficiales[m], color='red', alpha=0.8, linewidth=1)
        plt.ylabel("Anomalías SST (°C)")
        plt.xlabel("Años")
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()

        plt.savefig(fig_dir / f"comp_nino34_hadisst_vs_{m}.png", dpi=300, bbox_inches='tight')
        aux = input(f'Mostrar figura {m} [y/n]  ')
        if aux == 'y':
            plt.show()
        plt.close()

    # Comprobación de datos obs
    print(f"{'─' * 60}")
    aux = input('Comprobación métricas obs y multi-modelo [y/n]  ')
    if aux == 'y':
        print(f"\n{'-' * 45}")
        print(f"{'MÉTRICA (HadISST)':<25} | {'VALOR':<15}")
        print(f"{'-' * 45}")
        print(f"{'Amplitud (Desv. Est.)':<25} | {std_had:15.2f} °C")
        print(f"{'Estacionalidad (Ratio)':<25} | {seas_had:15.2f}")
        print(f"{'Máximo Beta (Patrón)':<25} | {max_beta_had:15.2f} °C/°C")
        print(f"{'-' * 45}\n")

    # Resumen resultados
    res_dir = root_fig / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultados.values()).round(6).to_csv(res_dir / "resumen_metricas_planton.csv", index=False)

    # Perfil zonal ecuatorial de todos los modelos
    plt.figure(figsize=(12, 6))
    for nombre, perfil in perfiles_modelos.items():
        perfil = a_0_360(perfil)
        plt.plot(perfil.longitude, perfil, color='royalblue', alpha=0.5, linewidth=1,
                 label='Modelos CMIP6' if nombre == modelos[0] else "")
    ensamble_mean = a_0_360(xr.concat(list(perfiles_modelos.values()), dim='model').mean('model'))
    plt.plot(ensamble_mean.longitude, ensamble_mean, color='blue', linewidth=2, label='Media ensemble CMIP6')
    perfil_medio_h_360 = a_0_360(perfil_medio_h)
    plt.plot(perfil_medio_h_360.longitude, perfil_medio_h_360, color='red', linewidth=2, label='HadISST (OBS)')

    plt.xlim(150, 270)
    plt.xticks([150, 180, 210, 240, 270], ['150°E', '180°', '150°W', '120°W', '90°W'])
    plt.xlabel("Longitud")
    plt.ylabel("SST media (°C)")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.savefig(fig_dir / "comp_zonal_todos_los_modelos.png", dpi=300, bbox_inches='tight')
    aux = input('Mostrar figura del perfil zonal [y/n]  ')
    if aux == 'y':
        plt.show()
    plt.close()
