"""
diagnostico_varianza_autocorr.py
Calcula la varianza y la autocorrelación de retardo 1 de cada nodo con los meses de octubre a marzo (emparejando solo meses consecutivos de una misma temporada), en la observación y en cada modelo y ventana, y el cambio relativo de varianza y el cambio de autocorrelación entre el histórico y cada escenario. En caida_relativa_varianza un valor positivo indica que la varianza aumenta. La varianza solo es informativa en N34, porque el resto de nodos están estandarizados en su ventana.
Entradas: data/indices/nodos_obs/ (de reduccion_nodos.py) y data/indices/matrices_cmip6/ (de compute_matrices_cmip6.py)
Salidas: resultados/resumen_varianza_autocorr.csv y resumen_varianza_autocorr_cambio.csv
Memoria: sección 5.9 y Tabla I.2.
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
from pathlib import Path

meses_temp = [10, 11, 12, 1, 2, 3]   # misma máscara estacional que pcmci_plus.py

def varianza_autocorr(serie, meses):
    # Calcula la varianza estacional y la autocorrelación de retardo 1 restringida a los meses de la temporada.

    en_temporada = np.isin(meses, meses_temp); idx = np.where(en_temporada)[0]
    var = float(np.nanvar(serie[idx], ddof=1))  # estimador muestral (N-1), como en el Apéndice B

    # Emparejamos pares consecutivos DENTRO de temporada (mismo criterio que ve la máscara de PCMCI+, sin saltar el hueco abril-septiembre)
    px, py = [], []
    for i in idx:
        if (i + 1) in idx and (meses[i + 1] - meses[i]) % 12 == 1:
            px.append(serie[i]); py.append(serie[i + 1])
    px, py = np.array(px), np.array(py)
    ac1 = float(np.corrcoef(px, py)[0, 1]) if len(px) > 2 else np.nan
    return var, ac1, len(idx), len(px)

if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    mod_dir = root_data / "data" / "indices" / "matrices_cmip6"
    obs_dir = root_data / "data" / "indices" / "nodos_obs"
    res_dir = root / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)

    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
    periodos = {"W_hist": "historical", "W_ssp245": "ssp2_4_5", "W_ssp585": "ssp5_8_5"}
    nodos_finales = list(np.load(obs_dir / "nodos_finales.npy", allow_pickle=True))

    filas = []

    # Calculamos métricas para la referencia observacional
    ds_obs = xr.open_dataset(obs_dir / "W_obs_matriz_780_reducida.nc"); meses_obs = ds_obs.time.dt.month.values
    for n in nodos_finales:
        var, ac1, n_temp, n_pares = varianza_autocorr(ds_obs[n].values, meses_obs)
        filas.append({
            "nodo": n,
            "modelo": "obs",
            "ventana": "W_obs",
            "varianza": round(var, 4),
            "autocorr_lag1": round(ac1, 3) if np.isfinite(ac1) else None,
            "n_temporada": n_temp,
            "n_pares": n_pares
        })

    # Calculamos métricas para las simulaciones de los modelos CMIP6
    for nm, p in periodos.items():
        for m in modelos:
            ruta = mod_dir / p / f"{nm}_{m}.nc"
            if not ruta.exists():
                print(f"    [FALTA] {nm} {m}")
                continue
            ds = xr.open_dataset(ruta)
            meses = ds.time.dt.month.values
            for n in nodos_finales:
                var, ac1, n_temp, n_pares = varianza_autocorr(ds[n].values, meses)
                filas.append({
                    "nodo": n,
                    "modelo": m,
                    "ventana": nm,
                    "varianza": round(var, 4),
                    "autocorr_lag1": round(ac1, 3) if np.isfinite(ac1) else None,
                    "n_temporada": n_temp,
                    "n_pares": n_pares
                })

    # Guardamos el resumen general de varianza y autocorrelación en un archivo CSV
    df = pd.DataFrame(filas)
    df.to_csv(res_dir / "resumen_varianza_autocorr.csv", index=False)

    # Análisis de cambio climático (Histórico vs. Escenarios Futuros): caida relativa de varianza y delta de autocorrelacion
    piv_var = df[df.modelo != "obs"].pivot_table(index=["nodo", "modelo"], columns="ventana", values="varianza")
    piv_ac = df[df.modelo != "obs"].pivot_table(index=["nodo", "modelo"], columns="ventana", values="autocorr_lag1")
    resumen = []

    for nodo, modelo in piv_var.index:
        v_h = piv_var.loc[(nodo, modelo)].get("W_hist", np.nan)
        a_h = piv_ac.loc[(nodo, modelo)].get("W_hist", np.nan)
        for esc in ("W_ssp245", "W_ssp585"):
            if esc not in piv_var.columns:
                continue
            v_e = piv_var.loc[(nodo, modelo)].get(esc, np.nan)
            a_e = piv_ac.loc[(nodo, modelo)].get(esc, np.nan)

            caida_var = (v_e - v_h) / v_h if pd.notna(v_h) and v_h > 0 else None

            resumen.append({
                "nodo": nodo, "modelo": modelo, "escenario": esc,
                "varianza_hist": round(v_h, 4) if pd.notna(v_h) else None,
                "varianza_esc": round(v_e, 4) if pd.notna(v_e) else None,
                "caida_relativa_varianza": round(caida_var, 3) if caida_var is not None else None,
                "autocorr_hist": a_h, "autocorr_esc": a_e,
                "delta_autocorr": round(a_e - a_h, 3) if pd.notna(a_h) and pd.notna(a_e) else None
            })

    # Guardamos el resumen del cambio en un archivo CSV
    df_resumen = pd.DataFrame(resumen)
    df_resumen.to_csv(res_dir / "resumen_varianza_autocorr_cambio.csv", index=False)

    print(f"\n----MAYORES CAÍDAS DE VARIANZA----\n")
    print(df_resumen.sort_values("caida_relativa_varianza").head(15).to_string(index=False))

    print(f"\n----MAYORES SUBIDAS DE AUTOCORRELACIÓN DE RETARDO 1----\n")
    print(df_resumen.sort_values("delta_autocorr", ascending=False).head(15).to_string(index=False))
