"""
compute_matrices_cmip6.py
Matrices W_hist, W_ssp245 y W_ssp585 de cada uno de los 7 modelos para los 8 nodos finales: N34 como índice de área y el resto de series proyectando las anomalías de cada campo sobre la EOF observacional. Las anomalías se calculan respecto a la climatología de referencia de cada ventana (1981-2010 en el histórico y 2050-2099 en cada escenario).
Entradas: data/processed/cmip6/, data/indices/solvers_obs/ (de compute_solvers_obs.py) y data/indices/nodos_obs/nodos_finales.npy (de reduccion_nodos.py)
Salidas: data/indices/matrices_cmip6/<experimento>/<W>_<modelo>.nc
Memoria: secciones 5.1 a 5.3.
"""

import xarray as xr
import numpy as np
import pickle
import os
from pathlib import Path
from tqdm import tqdm
import warnings
import compute_nodos_obs as cno
import compute_solvers_obs as cso

warnings.filterwarnings("ignore")

def PC_sB(anom, solver_obs, npc=1):
    # Calcula la Componente Principal mediante la estrategia B (proyección directa sobre el EOF observacional).
    anom_t = anom.transpose('time', ...)
    mode_idx = npc - 1
    pc = solver_obs.projectField(anom_t, neofs=npc, eofscaling=1).isel(mode=mode_idx)
    return (pc - pc.mean()) / pc.std()

if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    pro_dir = root_data / "data" / "processed" / "cmip6"
    obs_dir = root_data / "data" / "indices" / "nodos_obs"
    solv_dir = root_data / "data" / "indices" / "solvers_obs"
    out_dir = root_data / "data" / "indices" / "matrices_cmip6"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargamos los nodos finales seleccionados tras el proceso de reducción
    nodos_finales = list(np.load(obs_dir / "nodos_finales.npy", allow_pickle=True))

    # Cargamos solvers precalculados para los datos observacionales
    solvers = {}
    for n in nodos_finales:
        if n == 'N34': continue
        with open(solv_dir / f"solver_{n.replace('_PC2','')}.pkl", "rb") as f:
            solvers[n] = pickle.load(f)

    # Cargamos máscara de tierra y nodos
    mask_land = xr.open_dataarray(solv_dir / "mask_land.nc")
    nodos = cno.get_nodos(mask_land)

    # Definimos modelos CMIP6 y experimentos temporales
    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
    periodos = {"W_hist": ("historical", slice("1950", "2014"), slice("1981-01-01", "2010-12-31")),
                "W_ssp245": ("ssp2_4_5", slice("2050", "2099"), slice("2050-01-01", "2099-12-31")),
                "W_ssp585": ("ssp5_8_5", slice("2050", "2099"), slice("2050-01-01", "2099-12-31"))}     # recorte futuro a la hora de construir matrices

    # Bucle principal iterando sobre cada experimento, periodo y modelo climático
    for nm, (p, ventana, p_base) in periodos.items():
        print(f"{nm} ({p})")
        for model in modelos:
            series_m = {}
            pbar2 = tqdm(nodos_finales, desc=f"  {model}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')

            for nodo in pbar2:
                pbar2.set_postfix_str(f"{nodo}")

                # Carga de datos procesados del modelo actual para el nodo correspondiente
                datos = nodos.get(nodo.replace('_PC2',''), {})
                folder = datos['folder']
                dir_var = pro_dir / folder
                path_var = list(dir_var.glob(f"*{p}_allmodels.nc"))[0]
                nv = path_var.name.split('_')[0]
                campo = xr.open_dataset(path_var, engine="netcdf4")
                campo_modelo = campo[nv].sel(model=model)

                # Cálculo del índice N34
                if nodo == 'N34':
                    serie = cno.indice_n34(campo_modelo, p_base=p_base).sel(time=ventana)
                    series_m[nodo] = serie
                    continue

                # Estrategia B: Extracción de anomalías y proyección espacial sobre el EOF observacional
                dom = datos['dom']
                anom = cso.preproceso_pcs(campo_modelo, p_base=p_base, **dom).sel(time=ventana)  # ventana DESPUÉS de las anomalías
                if '_PC2' not in nodo:
                    serie = PC_sB(anom, solvers[nodo], npc=1)
                else:
                    serie = PC_sB(anom, solvers[nodo], npc=2)
                series_m[nodo] = serie
            pbar2.set_postfix_str("¡Finalizado!")

            # Construimos y guardamos matriz
            t_m = series_m[nodos_finales[0]].time.values
            for n in nodos_finales[1:]:
                t_m = np.intersect1d(t_m, series_m[n].time.values)

            ds_m = xr.Dataset({n: xr.DataArray(series_m[n].sel(time=t_m).values, coords=[t_m], dims=['time']) for n in nodos_finales},
                attrs={
                    "descripcion": f"Matriz final {nm} para el modelo {model}.",
                    "modelo": model,
                    "periodo": p,
                    "nodos": ", ".join(nodos_finales),
                    "T": len(t_m), "N": len(nodos_finales),
                    "estrategia": "Los nodos se han conseguido mediante la strategy B, es decir hemos calculando las PCs de cada variable proyectando el campo sobre la EOF observacional."
                })
            (out_dir / p).mkdir(parents=True, exist_ok=True)
            ds_m.to_netcdf(out_dir / p / f"{nm}_{model}.nc")
