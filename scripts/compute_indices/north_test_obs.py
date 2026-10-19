"""
north_test_obs.py
Criterio de North con grados de libertad efectivos por autocorrelación de lag 1 (Bretherton et al., 1999) para los tres primeros modos de cada variable observacional. La función plot_north_test_neff también la usa diagnostico_nodos_cmip6.py. Pregunta por pantalla antes de mostrar las figuras.
Entradas: data/indices/solvers_obs/solver_<variable>.pkl (de compute_solvers_obs.py)
Salidas: figures/nodos_obs/north_test_plots_neff.png
Memoria: sección 5.3 y Apéndice D.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

def plot_north_test_neff(solver, ax = None, titulo = None, num_modos=3):
    # Calcula y grafica el criterio de North ajustado, utilizando grados de libertad efectivos (neff) basados en la autocorrelación temporal.

    # Porcentaje de varianza explicada y autovalores
    var_frac = solver.varianceFraction(neigs=num_modos).values * 100
    eigs = solver.eigenvalues(neigs=num_modos).values

    # Componentes principales (PC) sin normalizar para calcular la autocorrelación
    pcs = solver.pcs(npcs=num_modos, pcscaling=0)
    n   = pcs.shape[0]

    # Grados de libertad efectivos por autocorrelación temporal entre pcs (neff) para cada modo
    neff = []
    for k in range(num_modos):
        pc = pcs[:, k]
        pc = pc[~np.isnan(pc)]
        r1 = np.corrcoef(pc[:-1], pc[1:])[0, 1]        # autocorrelación de lag-1
        neff.append(max(n * (1 - r1) / (1 + r1), 2))   # Bretherton et al. (1999)
    neff = np.array(neff)

    # Calculamos el margen de error según el criterio de North usando neff
    factor = np.sqrt(2.0 / np.array(neff)) / solver.eigenvalues().sum().item()
    errores = eigs * factor * 100

    # Graficamos
    if ax is not None:
        for i in range(num_modos):
            y_min = var_frac[i] - errores[i]
            y_max = var_frac[i] + errores[i]
            ax.axhspan(y_min, y_max, color='red', alpha=0.15, edgecolor='none')

        ax.errorbar(range(1, num_modos + 1), var_frac, yerr=errores,
                    fmt='o', capsize=5, capthick=2, color='black', markerfacecolor='red')

        ax.set_title(titulo, fontsize=9)
        ax.set_xlabel("PC", fontsize=8); ax.set_ylabel("Var. explicada (%)", fontsize=8)
        ax.set_xticks(range(1, num_modos + 1)); ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.6)

    return var_frac, errores, neff

if __name__ == '__main__':

    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    solv_dir = root_data / "data" / "indices" / "solvers_obs"
    fig_dir = root / "figures" / "nodos_obs" ; fig_dir.mkdir(parents=True, exist_ok=True)

    variables = ['SST', 'OLR', 'PSL', 'CHI200', 'CHI850', 'PSI200', 'PSI850', 'Z200_PSA', 'Z200_PNA', 'PR_land', 'TAS_land']

    cols = 4
    rows = int(np.ceil(len(variables) / cols))
    fig = plt.figure(figsize=(2.5 * cols, 3 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.35)

    for i, var in enumerate(variables):

        # Cargamos solvers
        with open(solv_dir / f"solver_{var}.pkl", "rb") as f:
            solver = pickle.load(f)

        # Graficamos
        ax = fig.add_subplot(gs[i])
        _, _, _ = plot_north_test_neff(solver, ax, titulo=var)

    fig.tight_layout()
    fig.savefig(fig_dir / "north_test_plots_neff.png", dpi=200, bbox_inches="tight")
    aux = input('Mostrar gráficas [y/n]  ')
    if aux == 'y':
        plt.show()
    plt.close(fig)

    nodos_eliminados = ['SST_PC2']
    print(f"\nEliminamos los nodos {nodos_eliminados} por no cumplir el criterio de North.")