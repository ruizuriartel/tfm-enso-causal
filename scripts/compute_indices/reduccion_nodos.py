"""
reduccion_nodos.py
Reduce los nodos candidatos a los 8 finales con la correlación de Pearson (descarte directo si |r| > 0.85, zona de inspección entre 0.70 y 0.85) y el VIF iterativo (umbral 10), con N34 como ancla exenta de descarte y reglas de desempate físicas. Descarta SST_PC2 a mano tras el criterio de North. Trabaja sobre la serie de octubre a marzo (390 meses). Pregunta por pantalla antes de mostrar la figura de patrones.
Entradas: data/indices/nodos_obs/W_obs_matriz_780_candidata.nc, series_pc1_pc2.pkl y patrones_pc1_pc2.pkl (de compute_matriz_obs.py y compute_nodos_obs.py); data/indices/solvers_obs/mask_land.nc; data/processed/hadisst/ (para el patrón de N34)
Salidas: data/indices/nodos_obs/W_obs_matriz_780_reducida.nc, nodos_finales.npy y log_reduccion.txt; figures/nodos_obs/heatmap_corr_vif.png y patrones_finales.png
Memoria: sección 5.3 (Figura 5.1) y Apéndice D.
"""

import xarray as xr
import numpy as np
import warnings
import os
from pathlib import Path
import pandas as pd
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pickle
import compute_nodos_obs as cno
import compute_solvers_obs as cso

warnings.filterwarnings("ignore")

def calc_vif(X, nombres):
    # Calcula el Factor de Inflación de Varianza (VIF) para un conjunto de variables.
    df = pd.DataFrame(X, columns=nombres).dropna()
    Xc = add_constant(df)
    return pd.Series([variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])], index=Xc.columns).drop("const")

def plot_corr_vif_heatmap(matriz_datos, nombres, orden = None):
    # Genera y exporta un mapa de calor combinado con las correlaciones de Pearson y los valores VIF iniciales.

    df_full = pd.DataFrame(matriz_datos, columns=nombres)

    # Matriz de correlación absoluta y cálculo de VIF excluyendo la variable ancla (N34)
    corr_matrix = df_full.corr().abs()
    nombres_sin_ancla = [n for n in nombres if n != 'N34']
    vif_values = calc_vif(df_full[nombres_sin_ancla].values, nombres_sin_ancla)
    vif_values['N34'] = np.nan
    vif_values = vif_values.reindex(nombres)
    df_vif = vif_values.to_frame(name='VIF')

    # Reordenamiento opcional de las variables en las matrices
    if orden is not None:
        corr_matrix = corr_matrix.reindex(index=orden, columns=orden); df_vif = df_vif.reindex(orden)
    else: orden = nombres

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[10, 1.2, 0.3], wspace=0.05)

    # Graficamos matriz de correlación
    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(corr_matrix, ax=ax1, annot=True, fmt=".2f", cmap='coolwarm', vmin=0, vmax=1, cbar=True,
                square=True, annot_kws={"size": 8})
    for i in range(len(orden)):
        for j in range(len(orden)):
            if i == j: continue
            val = corr_matrix.iloc[i, j]
            if val > 0.85:
                rect = patches.Rectangle((j, i), 1, 1, fill=False, edgecolor='forestgreen', lw=1.5, zorder=5)
                ax1.add_patch(rect)
            elif val >= 0.7:
                rect = patches.Rectangle((j, i), 1, 1, fill=False, edgecolor='yellowgreen', lw=1.5, zorder=4)
                ax1.add_patch(rect)

    # Graficamos valores VIF
    ax2 = fig.add_subplot(gs[1]); ax3 = fig.add_subplot(gs[2])
    sns.heatmap(df_vif, ax=ax2, annot=True, fmt=".1f", cmap='coolwarm', vmin=0, vmax=15, cbar_ax=ax3, yticklabels=False)
    for i in range(len(orden)):
        val = df_vif.iloc[i, 0]
        if val > 10:
            rect = patches.Rectangle((0, i), 1, 1, fill=False, edgecolor='forestgreen', lw=1.5, zorder=5)
            ax2.add_patch(rect)
        elif val >= 5:
            rect = patches.Rectangle((0, i), 1, 1, fill=False, edgecolor='yellowgreen', lw=1.5, zorder=4)
            ax2.add_patch(rect)

    plt.tight_layout(); plt.savefig(fig_dir / 'heatmap_corr_vif.png', dpi=300, bbox_inches='tight')

def desempate(na, nb, vif):
    # Decide cuál de los dos nodos colineales se descarta (memoria, sección 5.3).
    # El ancla N34 nunca llega aquí: pv_reduction la protege antes de llamar a esta función.
    # Reglas por orden de prioridad:

    # 1. Entre dos niveles de un mismo campo se conserva el de 200 hPa y se descarta el de 850 hPa
    #    (el nivel de 200 hPa es el más relevante para la respuesta atmosférica a ENSO)
    if '850' in na and '200' in nb: return na
    if '850' in nb and '200' in na: return nb

    # 2. Si interviene la PSL, se descarta la PSL (así se retienen más nodos dinámicos)
    if 'PSL' in nb: return nb
    if 'PSL' in na: return na

    # 3. Si ninguna de las anteriores aplica, se descarta el nodo con mayor VIF
    return na if vif[na] >= vif[nb] else nb

def pv_reduction(matriz_390, nombres, ancla='N34'):
    # Ejecuta el proceso secuencial de reducción de variables basado en umbrales de Pearson y VIF iterativo

    nodos = list(nombres); activos = list(nombres); inspeccion = []
    df = pd.DataFrame(matriz_390, columns=nombres).dropna()

    # Paso 1: Diagnóstico de correlación de Pearson por pares
    print("\n Diagnóstico de Pearson por pares")
    corr = df.corr().abs()
    for i in range(len(nodos)):
        for j in range(i + 1, len(nodos)):
            na, nb = nodos[i], nodos[j]
            if na not in activos or nb not in activos: continue
            r = corr.loc[na, nb]
            if r < 0.7: continue
            elif r <= 0.85: inspeccion.append((na, nb)); continue
            activos_sin_ancla = [n for n in activos if n != ancla]
            vif = calc_vif(df[activos_sin_ancla].values, activos_sin_ancla)
            if na == ancla: desc = nb
            elif nb == ancla: desc = na
            else: desc = desempate(na, nb, vif)
            activos.remove(desc)
            msg = f"  Pearson |r({na},{nb})|={r:.3f} --> DESCARTA {desc}"; log.append(msg); print(msg)
            inspeccion = [par for par in inspeccion if desc not in par]

    # Paso 2: Reducción iterativa de VIF dentro de la zona de inspección ambigua
    print("\n Reducción iterativa de VIF de la zona de inspección")
    inspeccion = list({nodo for par in inspeccion for nodo in par})
    print(f"  Nodos inspección: {inspeccion}")
    while True:
        activos_sin_ancla = [n for n in activos if n != ancla]
        vif = calc_vif(df[activos_sin_ancla].values, activos_sin_ancla)
        vif_ins = vif[vif.index.isin(inspeccion)]
        if vif_ins.empty or vif_ins.max() <= 10: print("  VIF máx en inspección ≤ 10. Fin."); break
        candidato = vif_ins.idxmax()
        desc = None
        for otro in activos:
            if otro in (candidato, ancla): continue
            d = desempate(candidato, otro, vif)
            if d != candidato: desc = d; break
        if desc is None: desc = candidato
        activos.remove(desc)
        if desc in inspeccion:
            inspeccion.remove(desc)
        msg = f"  VIF {desc}={vif[desc]:.2f} --> DESCARTADO"; log.append(msg); print(msg)

    return activos, log

def patron_n34(pro_dir, mask_land, n34):
        # Patrón de N34: regresión de las anomalías globales de SST sobre la serie N34 estandarizada,
        # con el mismo preprocesado y el mismo recorte octubre-marzo que el resto de patrones (PC_sA)
        datos = cno.get_nodos(mask_land)['SST']
        dir_var = pro_dir / datos['db'].lower() / datos['folder']
        path_var = list(dir_var.glob(f"{datos['db']}*.nc"))[0]
        nv = path_var.stem.split('_')[1]
        campo = xr.open_dataset(path_var, engine="netcdf4")[nv]
        anom_glob = cso.preproceso_pcs(campo, moc=True)
        meses_validos = [10, 11, 12, 1, 2, 3]
        anom_ge = anom_glob.where(anom_glob["time.month"].isin(meses_validos), drop=True)
        n34_n = (n34 - n34.mean()) / n34.std()
        t = np.intersect1d(anom_ge.time.values, n34_n.time.values)
        reg = xr.cov(anom_ge.sel(time=t), n34_n.sel(time=t), dim='time')
        return reg.where(abs(reg.latitude) <= 50)


if __name__ == '__main__':

    # Determinamos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    solv_dir = root_data / "data" / "indices" / "solvers_obs"; solv_dir.mkdir(parents=True, exist_ok=True)
    data_dir = root_data / "data" / "indices" / "nodos_obs"; data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / "figures" / "nodos_obs"; fig_dir.mkdir(parents=True, exist_ok=True)

    log = []

    # Subimos matriz candidata
    ds_candidata = xr.open_dataset(data_dir / "W_obs_matriz_780_candidata.nc")
    nombres = list(ds_candidata.data_vars)
    t = ds_candidata.time.values

    # Comprobación de la regla de North para PC2 --> eliminamos SST_PC2
    nombres = [n for n in nombres if n != 'SST_PC2']
    msg_sst = "  Eliminación manual: SST_PC2 descartada tras comprobar el criterio de North."
    log.append(msg_sst); print(f"\n{msg_sst}")

    # Matriz candidata
    matriz_780 = np.column_stack([ds_candidata[n].values for n in nombres])

    # Aplicamos máscara estacional
    meses_validos = [10, 11, 12, 1, 2, 3]
    idx_mascara = np.isin(pd.DatetimeIndex(t).month, meses_validos)
    matriz_390 = matriz_780[idx_mascara, :]

    # Ejecución del diagnóstico visual y del algoritmo de reducción de variables
    plot_corr_vif_heatmap(matriz_390, nombres)
    nodos_finales, log = pv_reduction(matriz_390, nombres)

    # Cálculo de los factores de inflación de varianza (VIF) finales para el subconjunto superviviente
    nodos_finales_sa = [n for n in nodos_finales if n != 'N34']
    df_final_sa = pd.DataFrame(matriz_390, columns=nombres).dropna()[nodos_finales_sa]
    vif_final = calc_vif(df_final_sa.values, nodos_finales_sa)

    # Guardamos los resultados procesados y metadatos
        # W_obs
    ds_780_reducida = ds_candidata[nodos_finales]
    ds_780_reducida.attrs.update({
        "descripcion": "Matriz final observacional W_obs reducida por Pearson+VIF",
        "nodos": ", ".join(nodos_finales), "T": len(t), "N": len(nodos_finales)})
    ds_780_reducida.to_netcdf(data_dir / "W_obs_matriz_780_reducida.nc")
        # Nodos finales
    np.save(data_dir / "nodos_finales.npy", np.array(nodos_finales))
        # Archivo de registro (log) con el resumen del proceso
    fin = f"\n\nNodos supervivientes ({len(nodos_finales)}): {nodos_finales}\n"; print(fin)
    with open(data_dir / "log_reduccion.txt", "w") as f:
        f.write(f"Nodos candidatos: {nombres}\n\n")
        f.write("\n".join(log))
        f.write(f"\n\nNodos supervivientes ({len(nodos_finales)}): {nodos_finales}\n")
        f.write(f"Descartados: {[n for n in nombres if n not in nodos_finales]}\n")
        f.write(f"\nVIF final: { {k: f'{v:.2f}' for k, v in vif_final.items()} }")

    # Imprimimos los patrones de los nodos finales
    pro_dir = root_data / "data" / "processed"
    with open(data_dir / "patrones_pc1_pc2.pkl", "rb") as f:
        patrones_todos = pickle.load(f)
    with open(data_dir / "series_pc1_pc2.pkl", "rb") as f:
        series_todas = pickle.load(f)
    mask_land = xr.open_dataarray(solv_dir / "mask_land.nc")

    patrones_finales = {}
    for nodo in nodos_finales:
        etiqueta = nodo.replace('_PC2', '')
        if nodo == 'N34':
            # N34 no es una EOF: sin varianza explicada y con el recuadro en la región Niño 3.4
            eof = patron_n34(pro_dir, mask_land, series_todas['N34'])
            patrones_finales[etiqueta] = (eof, None, "°C", -5, 5, 190, 240)
        else:
            clave = nodo if nodo.endswith('_PC2') else f"{nodo}_PC1"
            patrones_finales[etiqueta] = patrones_todos[clave]

    cno.plot_pattern_grid(patrones_finales, fig_dir / "patrones_finales.png", columnas=4)