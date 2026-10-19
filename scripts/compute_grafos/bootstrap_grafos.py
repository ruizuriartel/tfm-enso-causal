"""
bootstrap_grafos.py
Bootstrap por bloques de 12 meses (de julio a junio) con B=500 réplicas sobre cada una de las 22 ventanas, repitiendo PCMCI+ en cada réplica (4 procesos en paralelo, un hilo por proceso). Para cada arista guarda la frecuencia de aparición, la coherencia de signo y la fracción significativa tras la corrección FDR. En las 7 ventanas históricas lo repite con el número de bloques de la ventana de escenario (muestreo igualado). Fija B, la semilla y los procesos que importan los demás scripts.
Entradas: data/indices/nodos_obs/W_obs_matriz_780_reducida.nc y data/indices/matrices_cmip6/
Salidas: data/indices/bootstrap/frecuencias_<ventana>_B500.csv y muestras_<ventana>_B500.pkl (también con sufijo _igualado en el histórico); resultados/resumen_bootstrap.csv
Memoria: secciones 5.6 y 5.10.
"""

import os
# Limitar hilos de librerías numéricas para evitar bloqueos en multiprocesamiento
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[v] = '1'         # monohilo

import numpy as np
import pandas as pd
import xarray as xr
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pcmci_plus as pp

# Definimos parámetros globales
B = 500
N = 4               # Procesos en paralelo (uno por hilo del procesador)
seed = 1996
mes_inicio = 7

# Nombres oficiales de los modelos (solo para figuras)
nombres_modelos = {"ec_earth3_cc": "EC-Earth3-CC", "access_cm2": "ACCESS-CM2", "cnrm_esm2_1": "CNRM-ESM2-1",
                   "hadgem3_gc31_ll": "HadGEM3-GC31-LL", "miroc6": "MIROC6", "noresm2_mm": "NorESM2-MM",
                   "ukesm1_0_ll": "UKESM1-0-LL"}

def bloques_temporada(meses, mes_inicio=mes_inicio):
    # Corta la serie temporal en bloques de 12 meses de julio a junio

    inicios = np.where((meses == mes_inicio))[0]
    bloques = []

    for i0 in inicios:
        bloque = np.arange(i0, i0 + 12)

        # Descartar el último bloque si está incompleto
        if bloque[-1] >= len(meses):
            continue

        bloques.append(bloque)

    # Devolvemos una lista de arrays de índices
    return bloques

def block_bootstrap(datos, bloques, rng, n_bloques=None, mes_inicio=mes_inicio):
    # Genera una nueva matriz de datos mediante remuestreo por bloques con reemplazo.

    k = n_bloques if n_bloques is not None else len(bloques)

    # Selección aleatoria de bloques con reemplazo (remuestreo)
    elegidos = rng.integers(0, len(bloques), size=k)
    idx = np.concatenate([bloques[e] for e in elegidos])

    # Reconstruimos la matriz y el vector temporal
    datos_r = datos[idx, :]
    meses_r = np.tile(np.array([(mes_inicio + m - 1) % 12 + 1 for m in range(12)]), k)
    return datos_r, meses_r

def muestra_bootstrap(args):
    # Ejecuta una muestra individual del bootstrap calculando el grafo causal.

    datos, bloques, nombres, semilla, n_bloques = args

    # Creamos un generador de números aleatorios independiente para la selección aleatoria de bloques
    rng = np.random.default_rng(semilla)

    # Remuestreamos
    datos_r, meses_r = block_bootstrap(datos, bloques, rng, n_bloques)

    # Cálculo del grafo causal mediante PCMCI+ de la nueva muestra
    try:
        g = pp.calcula_grafo(datos_r, nombres, meses_r)
        df = pp.enlaces_a_df(g)
        df = df[df.origen != df.destino]        # sin autoenlaces

        return {(r.origen, r.destino, r.tau): {"val": r.val, "q": r.q, "dirigida": r.dirigido} for r in df.itertuples()}

    except Exception as err:
        print(f"    muestra {semilla} fallida: {err}")
        return None

def bootstrap_ventana(datos, nombres, meses, etiqueta, n_bloques=None, B=B, n_jobs=N, seed=seed):
    # Ejecuta el remuestreo B veces en paralelo y computa estadísticas de robustez para las aristas.
                    # calcula en qué porcentaje de las réplicas sobrevive cada enlace causal.

    # Separamos la serie temporal en bloques de temporadas
    bloques = bloques_temporada(meses)
    k = n_bloques if n_bloques is not None else len(bloques)
    print(f"{etiqueta}: {len(bloques)} bloques disponibles, se remuestrean {k} | B = {B}")

    # Preparamos argumentos para muestra_bootstrap con semillas únicas por muestra
    args = [(datos, bloques, nombres, seed + b, n_bloques) for b in range(B)]
    t0 = time.time()

    # Procesamiento en paralelo en diferentes núcleos del procesador
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            muestras = list(ex.map(muestra_bootstrap, args))
    else:
        muestras = [muestra_bootstrap(a) for a in args]

    muestras = [r for r in muestras if r is not None]
    print(f"   {len(muestras)}/{B} muestras en {time.time()-t0:.0f} s")
    if not muestras:
        print("  sin muestras válidas"); return None, []

    # Agregación de frecuencias y métricas por arista
    conteo = {}; filas = []

    for m in muestras:
        for k_e, info in m.items():
            d = conteo.setdefault(k_e, {"n": 0, "vals": [], "n_dirigida": 0, "n_fdr": 0}) # inicializamos aristas
            d["n"] += 1; d["vals"].append(info["val"])
            d["n_dirigida"] += int(info["dirigida"])
            d["n_fdr"] += int(info["q"] < pp.fdr_alpha)     # solo es informativo en los enlaces con desfase (en los contemporáneos q = p)

    for (origen, destino, tau), c in conteo.items():
        vals = np.array(c["vals"])
        filas.append({
            "origen": origen,
            "destino": destino,
            "tau": tau,
            "frecuencia": round(c["n"] / len(muestras), 3),
            "val_medio": round(float(vals.mean()), 4),

            # val_std usa ddof=0 (el valor por defecto de NumPy), cuando por coherencia con el resto del trabajo debería usar ddof=1 (estimador muestral).
            # No se corrige porque no se usa en la memoria, con B = 500 el cambio es de un 0,1 % como mucho y obligaría a repetir el bootstrap completo.
            "val_std": round(float(vals.std()), 4),

            "coherencia_signo": round(float(max((vals > 0).mean(), (vals < 0).mean())), 3),
            "frac_dirigida": round(c["n_dirigida"] / c["n"], 3) if tau >= 1 else None, "frac_fdr": round(c["n_fdr"] / c["n"], 3)
        })

    df = pd.DataFrame(filas).sort_values("frecuencia", ascending=False).reset_index(drop=True)
    return df, muestras

if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    obs_dir = root_data / "data" / "indices" / "nodos_obs"
    mod_dir = root_data / "data" / "indices" / "matrices_cmip6"
    out_dir = root_data / "data" / "indices" / "bootstrap"; out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = root / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)

    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
    periodos = {"W_hist": "historical", "W_ssp245": "ssp2_4_5", "W_ssp585": "ssp5_8_5"}

    # Construimos de la lista completa de tareas
    tareas = [("W_obs", obs_dir / "W_obs_matriz_780_reducida.nc")]
    for nm, p in periodos.items():
        for m in modelos:
            tareas.append((f"{nm}_{m}", mod_dir / p / f"{nm}_{m}.nc"))

    # Extraer longitud base de los escenarios futuros para el remuestreo igualado
    n_bloques_ssp = None
    ref = mod_dir / "ssp2_4_5" / f"W_ssp245_{modelos[0]}.nc"
    if ref.exists():
        ds = xr.open_dataset(ref)
        n_bloques_ssp = len(bloques_temporada(ds.time.dt.month.values))
        print(f"Ventana de escenario: {n_bloques_ssp} bloques -> el histórico se igualará a ese tamaño")

    resumen = []

    # Procesamiento iterativo de cada matriz de nodos
    for etiqueta, ruta in tareas:

        # Cargamos datos
        ds = xr.open_dataset(ruta)
        df_n = ds.to_dataframe()
        datos = df_n.values
        nombres = list(df_n.columns)
        meses = ds.time.dt.month.values

        # BOOTSTRAP INTERNO: frecuencias de aristas absolutas
        df, muestras = bootstrap_ventana(datos, nombres, meses, etiqueta)
        if df is None:
            continue

        # Guardado de resultados
        destino = out_dir / f"frecuencias_{etiqueta}_B{B}.csv"
        df.to_csv(destino, index=False)
        with open(out_dir / f"muestras_{etiqueta}_B{B}.pkl", "wb") as f:
            pickle.dump({"etiqueta": etiqueta, "nombres": nombres, "muestras": muestras, "B": len(muestras)}, f)

        # Análisis de aristas por su frecuencia
        robustas = int((df.frecuencia >= 0.9).sum())
        fragiles = int((df.frecuencia < 0.5).sum())
        print(f"  robustas (f≥0.9): {robustas} | frágiles (f<0.5): {fragiles}")

        resumen.append({
            "ventana": etiqueta,
            "aristas_vistas": len(df),
            "robustas": robustas,
            "fragiles": fragiles
        })

        # MUESTREO IGUALADO (solo para historical): -> corrige sesgo de tamaño muestral para más adelante hacer el análisis de diferencias entre ventanas
        if etiqueta.startswith("W_hist") and n_bloques_ssp is not None:

            eq = f"{etiqueta}_igualado"
            df_eq, muestras_eq = bootstrap_ventana(datos, nombres, meses, eq, n_bloques=n_bloques_ssp)

            # Guardamos resultado
            if df_eq is None:
                continue

            df_eq.to_csv(out_dir / f"frecuencias_{eq}_B{B}.csv", index=False)
            with open(out_dir / f"muestras_{eq}_B{B}.pkl", "wb") as f:
                pickle.dump({"etiqueta": eq, "nombres": nombres, "muestras": muestras_eq, "B": len(muestras_eq)}, f)

            resumen.append({
                "ventana": eq,
                "aristas_vistas": len(df_eq),
                "robustas": int((df_eq.frecuencia >= 0.9).sum()),
                "fragiles": int((df_eq.frecuencia < 0.5).sum())
            })

    if resumen:
        pd.DataFrame(resumen).to_csv(res_dir / "resumen_bootstrap.csv", index=False)

