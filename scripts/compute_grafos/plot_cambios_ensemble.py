"""
plot_cambios_ensemble.py
Vista agregada del cambio estructural del ensemble bajo calentamiento, con la misma disposición de nodos que el grafo observacional. Solo dibuja el esqueleto persistente (gris) y los cambios robustos del criterio binario (naranja si aparecen, morado si desaparecen), etiquetados con el rango de su fracción ponderada bajo los tres esquemas. Las aristas dirigidas agrupan todos los desfases de una misma dirección.
Entradas: resultados/resumen_esqueleto_persistente.csv y resumen_cambio_binario_robusto_con_magnitud.csv (de comparacion_escenarios.py) y data/indices/pcmci_plus/ (para los nombres de los nodos, de pcmci_plus.py)
Salidas: figures/cambio_estructural/cambios_ensemble.png
Memoria: Figura 6.2.
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import plot_grafos_causales as pgc

esquemas = ["democracia", "solo_calidad", "calidad_independencia"]
escenarios = [("W_ssp245", "SSP2-4.5"), ("W_ssp585", "SSP5-8.5")]

# Colores: gris para el esqueleto y una pareja naranja/morado para los cambios, distinta de la
# escala roja/azul de los grafos (que allí codifica el signo de la correlación parcial)
col_esq = "#8C8C8C"
col_tipo = {"aparece": "#E08214", "desaparece": "#8073AC"}
estilo_tipo = {"aparece": "solid", "desaparece": "solid"}
curva_cambios = 0.08   # curvatura de las aristas de cambio (signo = lado hacia el que se curvan)

def es_dirigida(tau):
    # En comparacion_escenarios los enlaces con desfase se agregan sobre todos los tau>=1 (tau = "-")
    return str(tau).strip() != "0"

def dibuja_cambios(ax, nombres, esqueleto, cambios, titulo):
    pos = pgc.layout_cadena(nombres)
    r_nodo = 0.155

    aristas = [dict(i=r.origen, j=r.destino, dirigida=es_dirigida(r.tau), tipo="esqueleto", etiqueta=None)
               for r in esqueleto.itertuples()]
    for r in cambios.itertuples():
        fr = [getattr(r, f"frac_{r.tipo}_{e}") for e in esquemas]
        lo, hi = f"{min(fr):.2f}", f"{max(fr):.2f}"
        aristas.append(dict(i=r.origen, j=r.destino, dirigida=es_dirigida(r.tau), tipo=r.tipo, etiqueta=lo if lo == hi else f"{lo}-{hi}"))

    for e in aristas:
        p0, p1 = np.array(pos[e["i"]]), np.array(pos[e["j"]])
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        a, b = p0 + u * r_nodo, p1 - u * r_nodo
        L_ab = np.linalg.norm(b - a)

        # Curvamos la arista si pasa por encima de otro nodo, igual que en los grafos de PCMCI+
        par = tuple(sorted((e["i"], e["j"])))
        lado = pgc.lado_a_evitar(np.array(pos[par[0]]), np.array(pos[par[1]]), pos, excluir=set(par))
        sep = 0.13 * 0.55 * (-lado) if lado != 0 else 0.0
        # Los cambios sin nodo en medio se curvan ligeramente para no montarse sobre el esqueleto
        if e["tipo"] != "esqueleto" and lado == 0:
            sep = curva_cambios
        if (e["i"], e["j"]) != par:
            sep = -sep
        rad = 2 * sep / L_ab

        if e["tipo"] == "esqueleto":
            color, estilo, lw, z = col_esq, "solid", 2.2, 1
        else:
            color, estilo, lw, z = col_tipo[e["tipo"]], estilo_tipo[e["tipo"]], 3.2, 2

        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>" if e["dirigida"] else "-", mutation_scale=18,
                                     linewidth=lw, linestyle=estilo, color=color, zorder=z,
                                     shrinkA=0, shrinkB=0))

        # Etiqueta con el rango de la fracción ponderada en el punto medio de la arista
        if e["etiqueta"]:
            n_der = np.array([u[1], -u[0]])
            c = 0.5 * (a + b) + sep * n_der
            ax.text(c[0], c[1], e["etiqueta"], ha="center", va="center", fontsize=8, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, linewidth=1.2))

    for n in nombres:
        x, y = pos[n]
        ax.add_patch(Circle((x, y), r_nodo, facecolor="#DEDEDE", edgecolor="#555555", linewidth=1.2, zorder=5))
        ax.text(x, y, n.replace('_PC2', '').replace('_land', '').replace('Z200_', ''),
                ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)

    if cambios.empty:
        ax.text(0.2, -0.77, "sin cambios robustos", ha="center", va="center", fontsize=9, style="italic", color="#555555")

    ax.set_xlim(-1.35, 1.45); ax.set_ylim(-0.82, 0.75)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(titulo, fontsize=13)

if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    pcmci_dir = root_data / "data" / "indices" / "pcmci_plus"
    res_dir = root / "resultados"
    fig_dir = root / "figures" / "cambio_estructural"; fig_dir.mkdir(parents=True, exist_ok=True)

    # Nombres de los nodos, tomados del grafo observacional para usar exactamente la misma disposición
    ruta_obs = next(pcmci_dir.glob("*/grafo_W_obs.pkl"))
    with open(ruta_obs, "rb") as f:
        nombres = pickle.load(f)["nombres"]

    esqueleto = pd.read_csv(res_dir / "resumen_esqueleto_persistente.csv")
    robustos = pd.read_csv(res_dir / "resumen_cambio_binario_robusto_con_magnitud.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for ax, (esc, titulo) in zip(axes, escenarios):
        dibuja_cambios(ax, nombres, esqueleto, robustos[robustos.escenario == esc], titulo)

    # Los dos tipos de cambio van siempre en la leyenda, aunque alguno no aparezca en la figura
    ley = [Line2D([0], [0], color=col_esq, lw=2.2, label="esqueleto persistente")]
    for tipo in ("aparece", "desaparece"):
        ley.append(Line2D([0], [0], color=col_tipo[tipo], lw=3.2, linestyle=estilo_tipo[tipo], label=f"cambio robusto: {tipo}"))
    ley += [Line2D([0], [0], color="#444444", lw=2, marker=">", markersize=7, label="dirigida (τ≥1)"),
            Line2D([0], [0], color="#444444", lw=2, label="sin dirección (τ=0)")]
    fig.legend(handles=ley, loc="lower center", ncol=len(ley), fontsize=9, frameon=False, handlelength=3.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.08, wspace=0.02)

    fig.savefig(fig_dir / "cambios_ensemble.png", dpi=250, bbox_inches="tight"); plt.close(fig)
