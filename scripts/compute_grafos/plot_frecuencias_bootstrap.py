"""
plot_frecuencias_bootstrap.py
Mapas de calor de la frecuencia de aparición de cada arista en el bootstrap por bloques, para la observación y los 7 modelos en cada ventana, con las adyacencias contemporáneas (τ=0) arriba y los enlaces dirigidos (τ≥1, todos los desfases de una misma dirección agregados) abajo. Es la misma forma de contar que la capa de topología del F1 y el criterio binario de cambio. En el histórico usa el bootstrap completo, no el igualado. Solo se muestran las aristas con frecuencia de al menos 0.3 en alguna columna.
Entradas: data/indices/bootstrap/muestras_<ventana>_B500.pkl (de bootstrap_grafos.py)
Salidas: figures/grafos_causales/frecuencias_W_hist.png, frecuencias_W_ssp245.png y frecuencias_W_ssp585.png
Memoria: Apéndice G.
"""

import itertools
import pickle
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import bootstrap_grafos as boot

modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
ventanas = ["W_hist", "W_ssp245", "W_ssp585"]

umbral_filas = 0.3    # None = sin filtro; con un valor (p. ej. 0.5) solo se muestran las aristas con f >= umbral en alguna columna
alto_fila = 0.28      # altura de cada fila en pulgadas

def frecuencias_agregadas(ruta_muestras):
    # Frecuencia de cada adyacencia (tau=0) y de cada enlace dirigido (cualquier tau>=1), contando en cuantas réplicas aparece la arista al menos una vez

    with open(ruta_muestras, "rb") as f:
        d = pickle.load(f)
    muestras, nombres = d["muestras"], d["nombres"]
    n = len(muestras)

    cont, dirig = {}, {}
    for a, b in itertools.combinations(sorted(nombres), 2):
        par = frozenset((a, b))
        cont[f"{a}-{b}"] = sum(any(tau == 0 and frozenset((o, dd)) == par for (o, dd, tau) in m) for m in muestras) / n
    for a, b in itertools.permutations(sorted(nombres), 2):
        dirig[f"{a}→{b}"] = sum(any(tau >= 1 and o == a and dd == b for (o, dd, tau) in m) for m in muestras) / n
    return cont, dirig

def tabla_frecuencias(boot_dir, vent, modelos, B=boot.B):
    # Construye las dos tablas (tau=0 y tau>=1) con la observación en la primera columna. En la ventana histórica se usa el bootstrap completo, no el igualado.

    columnas = {"obs": boot_dir / f"muestras_W_obs_B{B}.pkl"}
    columnas.update({m: boot_dir / f"muestras_{vent}_{m}_B{B}.pkl" for m in modelos})

    t0, t1 = {}, {}
    for col, ruta in columnas.items():
        if not ruta.exists():
            print(f"[FALTA] {ruta.name}"); continue
        t0[col], t1[col] = frecuencias_agregadas(ruta)

    tablas = []
    for t in (pd.DataFrame(t0), pd.DataFrame(t1)):
        if "obs" not in t.columns:
            return None, None
        if umbral_filas is not None:
            t = t[t.max(axis=1) >= umbral_filas]
        t = t.sort_values("obs", ascending=False)
        t.index = [k.replace("_PC2", "") for k in t.index]
        tablas.append(t)
    return tablas

def dibuja_tabla(ax, tabla, titulo, etiquetas_x):
    # Mapa de calor con el valor de cada celda; celda en blanco = la arista no aparece en ninguna réplica

    im = ax.imshow(tabla.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(tabla.index))); ax.set_yticklabels(tabla.index, fontsize=8)
    ax.set_xticks(range(len(tabla.columns)))
    ax.set_xticklabels([boot.nombres_modelos.get(c, "OBS") for c in tabla.columns] if etiquetas_x else [],
                       rotation=45, ha="right", fontsize=8)

    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            v = tabla.values[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="black" if v > 0.4 else "white")
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="none"))

    ax.axvline(0.5, color="black", linewidth=1.2)   # separa la observación de los modelos
    ax.set_title(titulo, fontsize=10, loc="left")
    return im

# Script principal
if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    boot_dir = root_data / "data" / "indices" / "bootstrap"
    fig_dir = root / "figures" / "grafos_causales"; fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n----FRECUENCIAS POR VENTANA----\n")
    for vent in ventanas:
        t0, t1 = tabla_frecuencias(boot_dir, vent, modelos)
        if t0 is None: continue

        fig = plt.figure(figsize=(9, alto_fila * (len(t0) + len(t1)) + 2.5))
        gs = gridspec.GridSpec(2, 2, height_ratios=[max(len(t0), 1), max(len(t1), 1)], width_ratios=[30, 1],
                               hspace=0.15, wspace=0.05)
        im = dibuja_tabla(fig.add_subplot(gs[0, 0]), t0, "Adyacencias contemporáneas (τ = 0)", etiquetas_x=False)
        dibuja_tabla(fig.add_subplot(gs[1, 0]), t1, "Enlaces dirigidos (τ ≥ 1, desfases agregados)", etiquetas_x=True)
        fig.colorbar(im, cax=fig.add_subplot(gs[:, 1]), label=f"frecuencia de aparición (B = {boot.B})")

        fig.savefig(fig_dir / f"frecuencias_{vent}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"    - {vent}: {len(t0)} adyacencias y {len(t1)} enlaces dirigidos")
