"""
diagnostico_nodos_cmip6.py
EOF propias de cada modelo para comprobar si cada nodo mide lo mismo que en observaciones: congruencia con el patrón observacional, criterio de North, varianza de la proyección, control sin tendencia lineal en los escenarios y congruencia intra-modelo entre el histórico y cada escenario.
Entradas: data/processed/cmip6/, data/indices/solvers_obs/ y data/indices/nodos_obs/ (de compute_solvers_obs.py y reduccion_nodos.py)
Salidas: resultados/resumen_diagnostico_nodos_modelos.csv y resumen_congruencia_intra_modelo.csv; figures/diagnostico_nodos/congruencia_<ventana>.png, varproy_<ventana>.png, caida_relativa_<escenario>.png y congruencia_intra_<escenario>.png
Memoria: secciones 5.4, 6.2 y Apéndice E.
"""

import xarray as xr
import numpy as np
import pandas as pd
import pickle
import warnings
import os
from pathlib import Path
import matplotlib.pyplot as plt

import compute_nodos_obs as cno
import compute_solvers_obs as cso
import north_test_obs as north

warnings.filterwarnings("ignore")

n_eofs = 5; umbral_congruencia= 0.6; umbral_congruencia_sens = 0.8

def var_proyeccion(anom, solver_obs, npc):
    # Calcula la fracción de varianza del campo del modelo recogida por la proyección sobre el patrón observacional

    pc = solver_obs.projectField(anom, neofs=npc, eofscaling=0).isel(mode=npc - 1)  # sin escalado
    patron = solver_obs.eofs(neofs=npc, eofscaling=0).isel(mode=npc - 1)

    t = np.intersect1d(anom.time.values, pc.time.values)
    a = anom.sel(time=t)
    recon = pc.sel(time=t) * patron

    var_total = float(((a - a.mean(dim="time")) ** 2).sum(skipna=True))
    var_resid = float(((a - recon) ** 2).sum(skipna=True))
    return (1 - var_resid / var_total) if var_total > 0 else np.nan

def congruencia(a, b):
    # Calcula el coeficiente de congruencia (correlación espacial) entre dos patrones bidimensionales

    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()   # aplanamos vectores a formato unidimensional
    land_mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[land_mask], b[land_mask]

    norm = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
    return float(np.sum(a * b) / norm) if norm > 0 else np.nan   # los patrones ya están ponderados en preproceso_pcs por sqrt(cos(lat))

def north_separado(solver, k, num_modos=n_eofs):
    # Aplica el test de North para evaluar si un modo EOF está estadísticamente bien separado de sus vecinos
    var_frac, errores, _ = north.plot_north_test_neff(solver, num_modos=num_modos)
    sup = (var_frac[k-1] - var_frac[k]) if k > 0 else np.inf
    inf = (var_frac[k] - var_frac[k+1]) if k+1 < num_modos else np.inf
    margen = min(sup, inf) / errores[k]             # adim., por eso debe ser mayor a 1
    return float(margen), bool(margen > 1.0)

def quita_tendencia(anom):
    # Elimina la tendencia lineal temporal de un campo anómalo mediante ajuste polinómico de grado 1
    coef = anom.polyfit(dim="time", deg=1, skipna=True)
    tend = xr.polyval(anom.time, coef.polyfit_coefficients)
    return anom - tend

def cota_intra(g1,g2):
    # Calcula la cota geométrica de consistencia intra-modelo entre dos escenarios
    return float(g1 * g2 - np.sqrt(max(0.0, (1 - g1 ** 2) * (1 - g2 ** 2))))

if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    pro_dir = root_data / "data" / "processed"
    obs_dir = root_data / "data" / "indices" / "nodos_obs"
    solv_dir = root_data / "data" / "indices" / "solvers_obs"
    fig_dir = root / "figures" / "diagnostico_nodos"; fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir = root / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)

    # Cargamos nodos finales, máscara de tierra y directorio con información de nodos
    nodos_finales = list(np.load(obs_dir / "nodos_finales.npy", allow_pickle=True))
    mask_land = xr.open_dataarray(solv_dir / "mask_land.nc")
    nodos = cno.get_nodos(mask_land)

    # Definimos modelos CMIP6 y las tres ventanas temporales a comparar contra la observación
    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
    ventanas = {"W_hist":   ("historical", slice("1950", "2014"), slice("1981-01-01", "2010-12-31"), False),
                "W_ssp245": ("ssp2_4_5",   slice("2050", "2099"), slice("2050-01-01", "2099-12-31"), True),
                "W_ssp585": ("ssp5_8_5",   slice("2050", "2099"), slice("2050-01-01", "2099-12-31"), True)}
        # el último valor de cada ventana (control_tend) marca si se hace también el chequeo sin tendencia,
        # solo tiene sentido en los escenarios (el histórico no lleva forzamiento de calentamiento fuerte)

    filas = []; patrones_modelo = {}

    # Bucle principal de diagnóstico cruzado entre modelos y observaciones por cada nodo en cada ventana
    print('\n---- ANÁLISIS DE CONGRUENCIA Y NORTH ----')
    for nodo in nodos_finales:
        print(f"\n {nodo} ")
        if nodo == 'N34':
            continue        # el ancla es un índice de área, no una PC, no tiene EOF que comparar
        base = nodo.replace('_PC2', '')
        npc = 2 if nodo.endswith('_PC2') else 1
        datos = nodos.get(base, {})
        folder = datos['folder']
        dom = datos['dom']

        # Patron de referencia -> la EOF observacional
        with open(solv_dir / f"solver_{base}.pkl", "rb") as f: solver_obs = pickle.load(f)
        patron_obs = solver_obs.eofs(neofs=npc, eofscaling=0).isel(mode=npc-1)

        # Análisis de los patrones de cada modelo CMIP6, en las tres ventanas
        for nm, (p, ventana, p_base, control_tend) in ventanas.items():
            print(f"    {nm}")
            for m in modelos:
                try:
                    # Cargamos el campo del modelo y calculamos sus propias EOF (n_eofs)
                    dir_var = pro_dir / "cmip6" / folder; path_var = list(dir_var.glob(f"*{p}_allmodels.nc"))[0]
                    nv = path_var.name.split('_')[0]
                    campo = xr.open_dataset(path_var, engine="netcdf4")
                    campo_modelo = campo[nv].sel(model=m)
                    anom = cso.preproceso_pcs(campo_modelo, p_base=p_base, **dom).sel(time=ventana)           # ventana DESPUÉS de las anomalías
                    solver_mod = cso.calc_solver(anom)
                    eofs_mod = solver_mod.eofs(neofs=n_eofs, eofscaling=0)

                    # Evaluación de congruencia con el patrón observacional y test de separación de North
                    gs = [congruencia(patron_obs.values, eofs_mod.isel(mode=k).values) for k in range(n_eofs)]
                    k_best = int(np.argmax(np.abs(gs)))     # Analizamos el que más se parece al observacional
                    g_signo = float(gs[k_best])
                    g = abs(g_signo)
                    margen, separado = north_separado(solver_mod, k_best)
                    var_exp = float(solver_mod.varianceFraction(neigs=n_eofs).values[k_best] * 100)

                    vp = var_proyeccion(anom, solver_obs, npc)
                    g_hist = patrones_modelo.get((nodo, m, "W_hist"), {}).get("g")
                    caida_relativa = ((g_hist - g) / g_hist) if (g_hist is not None and g_hist > 0 and nm != "W_hist") else None
                    signo_fix = np.sign(g_signo) if g_signo != 0 else 1.0
                    patrones_modelo[(nodo, m, nm)] = {
                        "patron": eofs_mod.isel(mode=k_best).values * signo_fix,
                        "eof": k_best + 1,
                        "var_exp": var_exp,
                        "g": g}

                    # Análisis adicional sin tendencia lineal si el escenario lo requiere
                        # analizamos si una congruencia baja es por la tendencia de calentamiento o por una deriva real del patron
                    g_dt = np.nan
                    if control_tend:
                        solver_dt = cso.calc_solver(quita_tendencia(anom))
                        eofs_dt = solver_dt.eofs(neofs=n_eofs, eofscaling=0)
                        g_dt = max(abs(congruencia(patron_obs.values, eofs_dt.isel(mode=k).values)) for k in range(n_eofs))

                    # Resultados
                    if g < umbral_congruencia:
                        conf = f"baja (g<{umbral_congruencia})"
                    elif not separado:
                        conf = "media - falla North"
                    else:
                        conf = "alta"

                    aviso = ""
                    if control_tend and np.isfinite(g_dt):
                        aviso = f"  g_dt={g_dt:.2f}"
                        if g < umbral_congruencia <= g_dt:
                            aviso = f"  g_dt={g_dt:.2f} | caída por tendencia (base válida)"
                        elif g < umbral_congruencia and g_dt < umbral_congruencia:
                            aviso = f"  g_dt={g_dt:.2f} | DERIVA REAL del patrón"

                    print(f"       {m:<17} g={g:.2f}   EOF{k_best+1} del modelo ({var_exp:.1f}%)   margen North={margen:.2f}  {conf}{aviso}")

                    filas.append({
                        "nodo": nodo, "ventana": nm, "periodo": p, "modelo": m,
                        "congruencia": round(g, 3),
                        "congruencia_sin_tendencia": round(g_dt, 3) if np.isfinite(g_dt) else None,
                        "eof_modelo": k_best + 1,
                        "eof_obs": npc,
                        "var_explicada_pct": round(var_exp, 1),
                        "margen_north": round(margen, 2),
                        "north_separado": separado,
                        "orden_intercambiado": (k_best + 1) != npc,
                        "confianza": conf,
                        "var_proyeccion_pct": round(vp * 100, 1),
                        "caida_relativa_congruencia": round(caida_relativa, 3) if caida_relativa is not None else None,
                        "alta_confianza_0.8": bool(g >= umbral_congruencia_sens)
                    })
                except Exception as err:
                    print(f"       {m:<17} ERROR: {err}")
                    filas.append({
                        "nodo": nodo, "ventana": nm, "periodo": p, "modelo": m,
                        "congruencia": None,
                        "confianza": "error"
                    })
    df = pd.DataFrame(filas)
    df.to_csv(res_dir / "resumen_diagnostico_nodos_modelos.csv", index=False)

    # Sensibilidades y análisis de resultados
    print(f"\n  - Sensibilidad del umbral de congruencia")
    n_06 = int((df.congruencia >= umbral_congruencia).sum())
    n_08 = int((df.congruencia >= umbral_congruencia_sens).sum())
    print(f"\n      >= 0.6: {n_06} de {len(df)}   |   >= 0.8: {n_08} de {len(df)}")

    print(f"\n  - Nodos de baja confianza (g < {umbral_congruencia})\n")
    bajos = df[df.confianza.str.startswith("baja", na=False)]
    print(bajos[["nodo", "ventana", "modelo", "congruencia"]].to_string(index=False) if not bajos.empty else "  ninguno")

    print(f"\n  - Casos con intercambio de orden (el homólogo no es la EOF de igual indice)\n")
    swap = df[df.orden_intercambiado == True]
    print(swap[["nodo", "ventana", "modelo", "eof_obs", "eof_modelo", "congruencia"]].to_string(index=False) if not swap.empty else "  ninguno")

    # Mapa de calor de congruencia modelo-obs, uno por ventana: filas = nodos, columnas = modelos
    for vent in ventanas:
        sub = df[(df.ventana == vent) & df.congruencia.notnull()]
        if sub.empty: continue
        tabla = sub.pivot(index="nodo", columns="modelo", values="congruencia")
        norte = sub.pivot(index="nodo", columns="modelo", values="north_separado")
        g_dt = sub.pivot(index="nodo", columns="modelo", values="congruencia_sin_tendencia")
        ancho = 1.1 * len(tabla.columns) + 2
        alto = 0.9 * len(tabla.index) + 2
        fig, ax = plt.subplots(figsize=(ancho, alto))
        im = ax.imshow(tabla.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(tabla.columns)))
        ax.set_xticklabels([cno.nombres_modelos[c] for c in tabla.columns], rotation=45, ha="right", fontsize=16)
        ax.set_yticks(range(len(tabla.index)))
        ax.set_yticklabels([n.replace('_PC2', '') for n in tabla.index], fontsize=16)

        for i in range(tabla.shape[0]):
            for j in range(tabla.shape[1]):
                g = tabla.values[i, j]
                if not np.isfinite(g): continue
                marca = "" if norte.values[i, j] else " +"  # modos no separados (North)
                d = g_dt.values[i, j]
                if np.isfinite(d) and g < umbral_congruencia:
                    marca += " *" if d >= umbral_congruencia else " **"  # * tendencia, ** deriva real

                ax.text(j, i, f"{g:.2f}{marca}", ha="center", va="center", fontsize=15,
                        color="black" if g > 0.4 else "white")

        fig.colorbar(im, ax=ax, label="congruencia |g|")
        fig.tight_layout()
        fig.savefig(fig_dir / f"congruencia_{vent}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Mapa de calor de la fracción de varianza explicada por la proyeccion, uno por ventana: filas = nodos, columnas = modelos
    for vent in ventanas:
        sub = df[(df.ventana == vent) & df.var_proyeccion_pct.notnull()]
        if sub.empty: continue
        tabla = sub.pivot(index="nodo", columns="modelo", values="var_proyeccion_pct")
        ancho = 1.1 * len(tabla.columns) + 2
        alto = 0.9 * len(tabla.index) + 2
        fig, ax = plt.subplots(figsize=(ancho, alto))
        im = ax.imshow(tabla.values, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

        ax.set_xticks(range(len(tabla.columns)))
        ax.set_xticklabels([cno.nombres_modelos[c] for c in tabla.columns], rotation=45, ha="right", fontsize=16)
        ax.set_yticks(range(len(tabla.index)))
        ax.set_yticklabels([n.replace('_PC2', '') for n in tabla.index], fontsize=16)

        for i in range(tabla.shape[0]):
            for j in range(tabla.shape[1]):
                vp = tabla.values[i, j]
                if not np.isfinite(vp): continue
                ax.text(j, i, f"{vp:.0f}", ha="center", va="center", fontsize=15,
                        color="black" if vp > 40 else "white")

        fig.colorbar(im, ax=ax, label="varianza de la proyección (%)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"varproy_{vent}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Mapa de calor de la caida relativa de congruencia histórico -> escenario, uno por escenario: filas = nodos, columnas = modelos
    for esc in ("W_ssp245", "W_ssp585"):
        sub = df[(df.ventana == esc) & df.caida_relativa_congruencia.notnull()]
        if sub.empty: continue
        tabla = sub.pivot(index="nodo", columns="modelo", values="caida_relativa_congruencia") * 100   # en %
        ancho = 1.1 * len(tabla.columns) + 2
        alto = 0.9 * len(tabla.index) + 2
        fig, ax = plt.subplots(figsize=(ancho, alto))
        im = ax.imshow(tabla.values, cmap="RdBu_r", vmin=-50, vmax=50, aspect="auto")   # rojo: la congruencia cae, azul: sube

        ax.set_xticks(range(len(tabla.columns)))
        ax.set_xticklabels([cno.nombres_modelos[c] for c in tabla.columns], rotation=45, ha="right", fontsize=16)
        ax.set_yticks(range(len(tabla.index)))
        ax.set_yticklabels([n.replace('_PC2', '') for n in tabla.index], fontsize=16)

        for i in range(tabla.shape[0]):
            for j in range(tabla.shape[1]):
                c = tabla.values[i, j]
                if not np.isfinite(c): continue
                ax.text(j, i, f"{c:.1f}", ha="center", va="center", fontsize=15,
                        color="white" if abs(c) > 30 else "black")

        fig.colorbar(im, ax=ax, label="caída relativa de la congruencia (%)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"caida_relativa_{esc}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Evaluación de la congruencia intra-modelo (comparación entre patrón histórico y escenarios futuros)
        # el modo que calculo para el modelo en el histórico, ¿se parece al modo que calculo para ese mismo modelo en el escenario de calentamiento?
    print('\n \n---- ANÁLISIS DE CONGRUENCIA INTRA-MODELO ----\n')
    filas_intra = []
    for nodo in nodos_finales:
        if nodo == 'N34':
            continue
        for m in modelos:
            if (nodo, m, "W_hist") not in patrones_modelo: continue
            h = patrones_modelo[(nodo, m, "W_hist")]
            for esc in ("W_ssp245", "W_ssp585"):
                if (nodo, m, esc) not in patrones_modelo: continue
                e = patrones_modelo[(nodo, m, esc)]

                g_intra = congruencia(h["patron"], e["patron"])  # con signo, patrones alineados a obs con fix sign
                cota = cota_intra(h["g"], e["g"])                # test de consistencia (cota geometrica)
                filas_intra.append(
                    {"nodo": nodo, "modelo": m, "escenario": esc,
                     "congruencia_intra": round(g_intra, 3),
                     "eof_hist": h["eof"], "eof_esc": e["eof"],
                     "cambio_indice_eof": h["eof"] != e["eof"],
                     "var_exp_hist": round(h["var_exp"], 1), "var_exp_esc": round(e["var_exp"], 1),
                     "g_obs_hist": round(h["g"], 3), "g_obs_esc": round(e["g"], 3),
                     "cota_geometrica": round(cota, 3),
                     "viola_cota": bool(g_intra < cota)
                     })
    df_intra = pd.DataFrame(filas_intra)
    df_intra.to_csv(res_dir / "resumen_congruencia_intra_modelo.csv", index=False)

    bajos_intra = df_intra[df_intra.congruencia_intra < umbral_congruencia]
    print(bajos_intra.to_string(index=False) if not bajos_intra.empty else "  todos por encima del umbral")

    print(f"\n  - Test de consistencia geométrica\n")
    violaciones = df_intra[df_intra.viola_cota]
    if violaciones.empty: print("     ninguna celda viola la cota")
    else: print(violaciones[["nodo", "modelo", "escenario", "congruencia_intra", "cota_geometrica"]].to_string(index=False))

    # Mapa de calor de congruencia intra-modelo, uno por escenario: filas = nodos, columnas = modelos
    for esc in ("W_ssp245", "W_ssp585"):
        sub = df_intra[df_intra.escenario == esc]
        if sub.empty: continue
        tabla = sub.pivot(index="nodo", columns="modelo", values="congruencia_intra")
        cambio = sub.pivot(index="nodo", columns="modelo", values="cambio_indice_eof")
        viola = sub.pivot(index="nodo", columns="modelo", values="viola_cota")
        g_h = sub.pivot(index="nodo", columns="modelo", values="g_obs_hist")
        g_e = sub.pivot(index="nodo", columns="modelo", values="g_obs_esc")
        ancho = 1.1 * len(tabla.columns) + 2
        alto = 0.9 * len(tabla.index) + 2
        fig, ax = plt.subplots(figsize=(ancho, alto))
        im = ax.imshow(tabla.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(tabla.columns)))
        ax.set_xticklabels([cno.nombres_modelos[c] for c in tabla.columns], rotation=45, ha="right", fontsize=16)
        ax.set_yticks(range(len(tabla.index)))
        ax.set_yticklabels([n.replace('_PC2', '') for n in tabla.index], fontsize=16)

        for i in range(tabla.shape[0]):
            for j in range(tabla.shape[1]):
                g = tabla.values[i, j]
                if not np.isfinite(g):
                    continue
                marca = " (idx)" if cambio.values[i, j] else ""           # cambia el índice de EOF ganador respecto al histórico
                marca += " *" if bool(viola.values[i, j]) else ""
                extra = f"\n({g_h.values[i, j]:.2f}|{g_e.values[i, j]:.2f})" if g < umbral_congruencia else ""

                ax.text(j, i, f"{g:.2f}{marca}{extra}", ha="center", va="center",
                        fontsize=12 if extra else 15, color="black")

        fig.colorbar(im, ax=ax, label="congruencia intra con signo (g)")
        fig.tight_layout()
        fig.savefig(fig_dir / f"congruencia_intra_{esc}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
