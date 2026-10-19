"""
pcmci_plus.py
Ejecuta PCMCI+ (ParCorr, tau_min=0, tau_max=3, pc_alpha=0.05) con máscara estacional de octubre a marzo sobre las 22 ventanas (observacional y 7 modelos en histórico, SSP2-4.5 y SSP5-8.5). La corrección de Benjamini-Hochberg se calcula sobre los enlaces con desfase y se guarda como marcador, nunca como filtro. El resumen cuenta las aristas significativas solo entre los enlaces con desfase y sin autoenlaces. Las funciones calcula_grafo y enlaces_a_df las reutilizan el bootstrap y las figuras.
Entradas: data/indices/nodos_obs/W_obs_matriz_780_reducida.nc (de reduccion_nodos.py) y data/indices/matrices_cmip6/ (de compute_matrices_cmip6.py)
Salidas: data/indices/pcmci_plus/<periodo>/grafo_<ventana>.pkl y enlaces_<ventana>.csv; resultados/resumen_aristas_grafos.csv
Memoria: sección 5.5.
"""

import numpy as np
import xarray as xr
import pandas as pd
import pickle
import os
import warnings
from pathlib import Path
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

warnings.filterwarnings("ignore")

# Definimos parámetros globales
meses_temp = [10, 11, 12, 1, 2, 3]
tau_max = 3
pc_alpha = 0.05
fdr_alpha = 0.05
fdr_method = 'fdr_bh'

def calcula_grafo(datos, nombres, meses, tau_max=tau_max, pc_alpha=pc_alpha, fdr_method=fdr_method, meses_temp=meses_temp):
    # Ejecuta el algoritmo PCMCI+ sobre los datos enmascarando los meses fuera de la temporada definida.

    # Construimos la máscara para excluir los meses que no pertenecen a la temporada de interés (solo meses de octubre a marzo)
    mascara_1d = ~np.isin(meses, meses_temp)      # True = enmascarado en tigramite --> con mask_type='y' solo se evalúan los enlaces cuyo destino cae en octubre-marzo
    mask = np.tile(mascara_1d[:, None], (1, len(nombres)))

    # Inicializamos y ejecutamos PCMCI+
    dataframe = pp.DataFrame(datos, var_names=nombres, mask=mask)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(mask_type='y'), verbosity=0)
    resultados = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)

    # Corrección de p-valores mediante control de tasa de falsos positivos (FDR) --> (los enlaces contemporáneos no se corrigen y conservan su p-valor)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=resultados['p_matrix'], fdr_method=fdr_method)

    return {
        "graph": resultados['graph'],
        "val_matrix": resultados['val_matrix'],
        "p_matrix": resultados['p_matrix'],
        "q_matrix": q_matrix,
        "nombres": list(nombres),
        "T": datos.shape[0],
        "T_temporada": int((~mascara_1d).sum()),
        "tau_max": tau_max,
        "pc_alpha": pc_alpha,
        "fdr_method": fdr_method
    }

def enlaces_a_df(g, fdr_alpha=fdr_alpha):
    # Convierte la matriz de adyacencia y métricas del grafo de Tigramite en un DataFrame tabular para guardar la información.

    nombres = g["nombres"]
    graph = g["graph"]
    N, _, t_lags = graph.shape
    filas = []

    for i in range(N):
        for j in range(N):
            for tau in range(t_lags):
                enlace = graph[i, j, tau]

                # Omitir entradas sin enlace detectado
                if enlace in ("", None):
                    continue

                # Evitar duplicar enlaces contemporáneos (simetría i >= j para tau = 0)
                if tau == 0 and i >= j:
                    continue

                filas.append({
                    "origen": nombres[i],
                    "destino": nombres[j],
                    "tau": tau,
                    "tipo": "contemporaneo" if tau == 0 else "dirigido",
                    "enlace": enlace,                                                           # cadena de tigramite (-->, o-o, ...)
                    "dirigido": bool(tau >= 1 and enlace == "-->"),
                    "signo": int(np.sign(float(g["val_matrix"][i, j, tau]))),
                    "val": float(g["val_matrix"][i, j, tau]),
                    "p": float(g["p_matrix"][i, j, tau]),
                    "q": float(g["q_matrix"][i, j, tau]),
                    "fdr_significativo": bool(float(g["q_matrix"][i, j, tau]) < fdr_alpha)      # marca visual, NO filtro (en los contemporáneos q = p, porque no se corrigen)
                })

    return pd.DataFrame(filas)

if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    obs_dir = root_data / "data" / "indices" / "nodos_obs"
    mod_dir = root_data / "data" / "indices" / "matrices_cmip6"
    out_dir = root_data / "data" / "indices" / "pcmci_plus"; out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = root / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)

    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
    periodos = {"W_hist": "historical", "W_ssp245": "ssp2_4_5", "W_ssp585": "ssp5_8_5"}

    # Construimos de la lista completa de tareas (observaciones + modelos CMIP6): etiqueta, subcarpeta de salida, ruta de la matriz
    tareas = [("W_obs", "obs", obs_dir / "W_obs_matriz_780_reducida.nc")]
    for nm, p in periodos.items():
        for model in modelos:
            tareas.append((f"{nm}_{model}", p, mod_dir / p / f"{nm}_{model}.nc"))

    resumen = []

    # Procesamiento iterativo de cada matriz de nodos
    for etiqueta, sub, ruta in tareas:

        # Cargamos matriz de nodos
        out = out_dir / sub; out.mkdir(parents=True, exist_ok=True)
        ds = xr.open_dataset(ruta)
        df = ds.to_dataframe()
        datos = df.values
        nombres = list(df.columns)
        meses = ds.time.dt.month.values
        tiempos = ds.time.values

        # Cálculo del grafo causal mediante PCMCI+
        g = calcula_grafo(datos, nombres, meses)
        g.update({
            "etiqueta": etiqueta,
            "t_ini": str(tiempos[0])[:7],
            "t_fin": str(tiempos[-1])[:7],
            "meses_temporada": meses_temp,
            "fdr_alpha": fdr_alpha,
            "N": len(nombres)
        })

        # Guardamos resultados
        with open(out / f"grafo_{etiqueta}.pkl", "wb") as f:
            pickle.dump(g, f)

        df = enlaces_a_df(g)
        df.to_csv(out / f"enlaces_{etiqueta}.csv", index=False)

        n_cont = int((df.tau == 0).sum())
        n_dir = int((df.tau >= 1).sum())
        n_fdr = int(df[(df.tau >= 1) & (df.origen != df.destino)].fdr_significativo.sum())  # solo enlaces con desfase y sin autoenlaces, los únicos a los que se aplica la corrección

        print(f"{etiqueta:<28} T={g['T']:<4} temporada={g['T_temporada']:<4} "
              f"aristas={len(df):<3} (contemporaneas={n_cont}, dirigidas={n_dir})  FDR (desfase) q<{fdr_alpha}={n_fdr}")

        auto = int((df.origen == df.destino).sum())
        resumen.append({
            "ventana": etiqueta,
            "T": g["T"],
            "T_temporada": g["T_temporada"],
            "aristas": len(df),
            "autoenlaces": auto,
            "aristas_sin_auto": len(df) - auto,
            "contemporaneas": n_cont,
            "dirigidas": n_dir,
            "fdr_significativas_desfase": n_fdr,
            "periodo": f"{g['t_ini']} a {g['t_fin']}"
        })

    pd.DataFrame(resumen).to_csv(res_dir / "resumen_aristas_grafos.csv", index=False)