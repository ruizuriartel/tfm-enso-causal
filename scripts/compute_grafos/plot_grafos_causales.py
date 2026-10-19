"""
plot_grafos_causales.py
Dibuja los grafos causales de PCMCI+: el grafo observacional (grosor según la frecuencia de aparición en el bootstrap, trazo continuo o discontinuo según la significancia FDR y color según la correlación parcial relativa al máximo de su clase) y un panel por ventana con la observación y los 7 modelos, con grosor constante. La disposición de los nodos (layout_cadena) y el trazado de las aristas los reutiliza plot_cambios_ensemble.py.
Entradas: data/indices/pcmci_plus/<periodo>/grafo_<ventana>.pkl y enlaces_<ventana>.csv (de pcmci_plus.py) y data/indices/bootstrap/frecuencias_<ventana>_B500.csv (de bootstrap_grafos.py)
Salidas: figures/grafos_causales/grafo_W_obs.png, grafos_W_hist.png, grafos_W_ssp245.png y grafos_W_ssp585.png
Memoria: Figura 6.1 y Apéndice F.
"""

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pcmci_plus as pp
import bootstrap_grafos as boot

# Definición de capas espaciales para organizar visualmente los nodos de la red causal
capas = [
    (-1.15, ["N34"]),
    (-0.35, ["CHI200", "PSI200"], 1.1),
    (0.05, ["OLR"]),
    (0.55, ["Z200_PNA", "Z200_PSA"], 0.85),
    (1.20, ["TAS_land", "PR_land"], 0.55)]

def layout_cadena(nombres, capas=capas):
    # Calcula las coordenadas (x, y) de cada nodo en función de las capas predefinidas para estructurar el gráfico en forma de cadena causal.

    base = {n: n.replace('_PC2', '') for n in nombres}
    pos = {}

    # Posicionamiento estructurado según las capas definidas
    for x, capa, *d in capas:
        dy_capa = d[0] if d else 0.95
        presentes = sorted([n for n in nombres if base[n] in capa], key=lambda n: capa.index(base[n]))
        for iy, n in enumerate(presentes):
            y = 0.0 if len(presentes) == 1 else dy_capa * ((len(presentes) - 1) / 2 - iy)
            pos[n] = (x, y)

    # Posicionamiento de respaldo para nodos que no estén en la lista
    faltan = [n for n in nombres if n not in pos]
    for k, n in enumerate(faltan):
        ang = 2 * np.pi * k / max(len(faltan), 1); pos[n] = (1.2 * np.cos(ang), 1.2 * np.sin(ang))

    return pos

def carga_aristas(res_dir, sub, etiqueta, boot_dir, filtro_frecuencia = None):
    # Carga los enlaces causales desde el CSV y les cruza la frecuencia de aparición obtenida previamente mediante el análisis de bootstrap.

    # Cargamos CSV
    df = pd.read_csv(res_dir / sub / f"enlaces_{etiqueta}.csv")
    df = df[df.origen != df.destino]
    ruta_frec = boot_dir / f"frecuencias_{etiqueta}_B{boot.B}.csv"
    frec = {}
    if ruta_frec.exists():
        df_frec = pd.read_csv(ruta_frec)
        frec = {((min(r.origen, r.destino), max(r.origen, r.destino), 0) if r.tau == 0
                 else (r.origen, r.destino, int(r.tau))): float(r.frecuencia) for r in df_frec.itertuples()}

    out = []
    for r in df.itertuples():
        # Estandarizamos la clave de la arista según su retardo temporal (tau)
        k = (min(r.origen, r.destino), max(r.origen, r.destino), 0) if r.tau == 0 else (r.origen, r.destino, int(r.tau))
        fr = frec.get(k, 1.0)

        # Filtro opcional por umbral de frecuencia mínima de bootstrap
        if filtro_frecuencia is not None and fr < filtro_frecuencia: continue
        out.append({
            "i": r.origen,
            "j": r.destino,
            "tau": int(r.tau),
            "val": float(r.val),
            "q": float(r.q),
            "dirigida": bool(r.dirigido),
            "frec": fr
        })
    return out

def maximos_por_clase(grafos):
    # Calcula el valor absoluto máximo de las correlaciones parciales separadas por clases
    #     temporales (tau=0 contemporáneas vs tau>=1 retardadas) para estandarizar la escala de colores en los gráficos.
    m0, m1 = 0.0, 0.0
    for g in grafos:
        for e in g["aristas"]:
            v = abs(e["val"])
            if e["tau"] == 0:
                m0 = max(m0, v)
            else:
                m1 = max(m1, v)
    return (m0 if m0 > 0 else 1.0), (m1 if m1 > 0 else 1.0)

def lado_a_evitar(p0, p1, pos, excluir, radio=0.24):
    # Verifica si algún nodo intermedio interfiere con el trazado recto de una arista en el grafo
    #     para decidir de qué lado curvarla y evitar solapamientos visuales.
    d = p1 - p0; L = np.linalg.norm(d)
    if L < 1e-9:
        return 0
    u = d / L
    n = np.array([u[1], -u[0]])   # normal al segmento
    mejor_dist = radio
    lado = 0
    for nombre, p in pos.items():
        if nombre in excluir:
            continue
        v = np.array(p) - p0
        proj = np.dot(v, u)
        if not (0.1 * L < proj < 0.9 * L):
            continue
        dist = abs(np.dot(v, n))
        if dist < mejor_dist:
            mejor_dist = dist
            lado = np.sign(np.dot(v, n))
    return lado

def dibuja_grafo(g, ax, max0, max1, titulo="", grosor_por_frecuencia=True):
    # Dibuja los nodos, adyacencias (tau=0) y conexiones dirigidas (tau>=1):

    nombres = g["nombres"]
    aristas = g["aristas"]
    pos = layout_cadena(nombres)
    curvas = {}

    r_num, t_num = 0.055, 0.34
    r_nodo=0.155
    paso_sep = 0.13

    # Agrupamos aristas por pares de nodos
    for e in aristas:
        par = tuple(sorted((e["i"], e["j"])))
        curvas.setdefault(par, []).append(e)

    for par, lista in curvas.items():
        lista_ord = sorted(lista, key=lambda x: (x["tau"], x["i"]))
        n = len(lista_ord)
        p0_par = np.array(pos[par[0]])
        p1_par = np.array(pos[par[1]])
        lado_evitar = lado_a_evitar(p0_par, p1_par, pos, excluir=set(par))

        for k, e in enumerate(lista_ord):
            p0 = np.array(pos[e["i"]])
            p1 = np.array(pos[e["j"]])
            d = p1 - p0
            L = np.linalg.norm(d)
            u = d / L

            # Recortar los extremos de la línea en el borde de los discos
            a, b = p0 + u * r_nodo, p1 - u * r_nodo
            L_ab = np.linalg.norm(b - a)

            # Calculamos curvatura
            if lado_evitar != 0:
                sep = paso_sep * 0.55 * (k + 1) * (-lado_evitar)
            else:
                sep = 0.0 if n == 1 else paso_sep * (k - (n - 1) / 2)
            if (e["i"], e["j"]) != par:
                sep = -sep
            rad = 2 * sep / L_ab

            # Mapeo de color para las aristas según la magnitud relativa de la correlación parcial
            mx = max0 if e["tau"] == 0 else max1
            intensidad = np.clip(e["val"] / mx, -1, 1)
            color = plt.cm.RdBu_r(Normalize(-1, 1)(intensidad))

            # Configuración de grosor y estilo de trazo basados en el bootstrap y FDR
            fr = e.get("frec", 1.0)
            lw_min, lw_max = 0.8, 4.0
            grosor = (lw_min + (lw_max - lw_min) * fr) if grosor_por_frecuencia else 2.0
            escala_flecha = (10 + 16 * fr) if grosor_por_frecuencia else 13
            estilo = "solid" if e["q"] < pp.fdr_alpha else (0, (5.0 / grosor, 4.5 / grosor))

            # Dibujamos aristas
            flecha = "-|>" if e["dirigida"] else "-"
            ax.add_patch(FancyArrowPatch(
                a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle=flecha,
                mutation_scale=escala_flecha, linewidth=grosor, linestyle=estilo,
                color=color, zorder=1, shrinkA=0, shrinkB=0))

            # Dibujamos tau en la arista con retardo
            if e["tau"] >= 1:
                n_der = np.array([u[1], -u[0]])
                c = (1 - t_num) * a + t_num * b + 4 * t_num * (1 - t_num) * sep * n_der
                ax.add_patch(Circle(c, r_num, facecolor="white", edgecolor=color, linewidth=1.6, zorder=3))
                ax.text(c[0], c[1], str(e["tau"]), ha="center", va="center", fontsize=7, zorder=4)

    # Dibujamos los nodos
    for n in nombres:
        x, y = pos[n]
        ax.add_patch(Circle((x, y), r_nodo, facecolor="#DEDEDE", edgecolor="#555555", linewidth=1.2, zorder=5))
        ax.text(x, y, n.replace('_PC2', '').replace('_land', '').replace('Z200_', ''), ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)
    ax.set_xlim(-1.45, 1.55); ax.set_ylim(-0.95, 0.95)
    ax.set_aspect("equal"); ax.axis("off")
    if titulo:
        ax.set_title(titulo, fontsize=15)

def leyenda(fig, axes, max0, max1, ax_caja=None, sep=3.4, escala=1.0, mostrar_frecuencia=True):
    # Construye y ubica las barras de color y la leyenda.

    sm = ScalarMappable(norm=Normalize(-1, 1), cmap=plt.cm.RdBu_r)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=list(axes), orientation="horizontal", fraction=0.035, pad=0.03, aspect=60)

    fs = 8 * escala
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.tick_params(labelsize=fs)
    cb.set_label(f"correlación parcial relativa con signo (val / máximo global de su clase; τ=0: {max0:.3f}, τ≥1: {max1:.3f})", fontsize=fs)

    if mostrar_frecuencia:
        axg = cb.ax.inset_axes([0, -sep, 1, 1.0])
        xs = np.linspace(0, 1, 80 + 1)
        for x0, x1 in zip(xs[:-1], xs[1:]):
            f = (x0 + x1) / 2
            lw_min, lw_max = 0.8, 4.0
            axg.plot([x0, x1 + 1.5 / 80], [0, 0], color="#444444", solid_capstyle="butt", linewidth=lw_min + (lw_max - lw_min) * f)

        axg.set_xlim(0, 1)
        axg.set_ylim(-1, 1)
        axg.set_yticks([])
        axg.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axg.tick_params(labelsize=fs, length=2)
        for s in axg.spines.values():
            s.set_visible(False)
        axg.set_xlabel(f"frecuencia de aparición de la arista en el bootstrap por bloques (B = {boot.B})", fontsize=fs)

    elem_ley = [
        Line2D([0], [0], color="#444444", lw=2.5, linestyle="solid", label=f"FDR significativo (q < {pp.fdr_alpha})"),
        Line2D([0], [0], color="#444444", lw=2.5, linestyle=(0, (1.6, 1.0)), label=f"FDR no significativo (q ≥ {pp.fdr_alpha})"),
        Line2D([0], [0], color="#444444", lw=2.5, marker=">", markersize=7, label="dirigida τ≥1"),
        Line2D([0], [0], color="#444444", lw=2.5, label="sin dirección τ=0 (adyacencia)")
    ]

    destino = ax_caja if ax_caja is not None else list(axes)[0]; solo = ax_caja is not None and not destino.has_data()
    destino.legend(
        handles=elem_ley, loc="center" if solo else "lower left", ncol=1 if solo else 2,
        fontsize=fs * (1.3 if solo else 1.0), handlelength=3.2, labelspacing=0.8,
        frameon=True, edgecolor="#999999", framealpha=1.0, borderpad=1.0)

if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    res_dir = root_data / "data" / "indices" / "pcmci_plus"
    boot_dir = root_data / "data" / "indices" / "bootstrap"
    fig_dir = root / "figures" / "grafos_causales"; fig_dir.mkdir(parents=True, exist_ok=True)

    # Carga de todos los grafos causales guardados en formato pickle (.pkl)
    rutas = sorted(res_dir.glob("*/grafo_*.pkl")); grafos = {}
    for r in rutas:
        with open(r, "rb") as f:
            g = pickle.load(f)
        etiqueta = g.get("etiqueta", r.stem.replace("grafo_", ""))
        g["aristas"] = carga_aristas(res_dir, r.parent.name, etiqueta, boot_dir)
        grafos[etiqueta] = g

    # Calculamos máximos globales para normalizar colores equitativamente
    max0, max1 = maximos_por_clase(grafos.values())
    print(f"{len(grafos)} grafos | máximo |val| τ=0: {max0:.3f} | τ≥1: {max1:.3f}")

    # GRAFO OBSERVACIONAL
    # Generamos la figura individual para el grafo observacional de referencia (W_obs)
    if "W_obs" in grafos:
        fig, ax = plt.subplots(figsize=(8, 7))
        dibuja_grafo(grafos["W_obs"], ax, max0, max1)
        leyenda(fig, [ax], max0, max1, ax_caja=ax, sep=5.0)
            # Nota: el color codifica magnitud relativa dentro de cada clase (τ=0 vs τ≥1), no magnitud absoluta comparable entre clases.
                # Z200_PSA_PC2 aparece sin adyacencias. Es una propiedad real de la estructura causal observada, no un fallo de representación.
        fig.savefig(fig_dir / "grafo_W_obs.png", dpi=250, bbox_inches="tight"); plt.close(fig)

    # GRAFOS DE MODELOS
    # Generamos paneles de grafos comparando modelos CMIP6 frente a la observación por ventana climática
    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
    for vent in ["W_hist", "W_ssp245", "W_ssp585"]:
        presentes = [(m, grafos[f"{vent}_{m}"]) for m in modelos if f"{vent}_{m}" in grafos]
        if not presentes:
            continue

        ncols = 3; nrows = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.2 * nrows), layout="constrained")
        axes = np.array(axes).reshape(-1)

        # Dibujar la referencia observacional en la primera celda y los modelos en las siguientes
        dibuja_grafo(grafos["W_obs"], axes[0], max0, max1, titulo="Observacional (referencia)", grosor_por_frecuencia=False)
        for ax, (m, g) in zip(axes[1:], presentes):
            dibuja_grafo(g, ax, max0, max1, titulo=boot.nombres_modelos[m], grosor_por_frecuencia=False)

        for ax in axes[len(presentes) + 1:]:
            ax.axis("off")

        # Guardamos
        leyenda(fig, axes[:8], max0, max1, ax_caja=axes[8], escala=1.6, mostrar_frecuencia=False)
        fig.savefig(fig_dir / f"grafos_{vent}.png", dpi=200, bbox_inches="tight"); plt.close(fig)

