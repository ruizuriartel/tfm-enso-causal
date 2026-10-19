"""
comparacion_escenarios.py
Contraste de cambio estructural entre el histórico y cada escenario: (a) contraste global por modelo con la probabilidad de que el F1 entre ventanas quede por debajo del ruido interno de la ventana peor muestreada, con histórico igualado y sin igualar; (b) criterio binario arista por arista con la fracción ponderada de modelos bajo los tres esquemas y barrido de umbrales; (c) esqueleto persistente, formado por las aristas cuyo comportamiento mayoritario es persistir en los dos escenarios; (d) magnitud del cambio (delta_f) y cruce de modelos entre escenarios.
Entradas: data/indices/bootstrap/muestras_*.pkl (de bootstrap_grafos.py) y resultados/resumen_pesos_EM.csv (de comparacion_obs_modelos_hist.py)
Salidas: resultados/resumen_cambio_estructural_general.csv, resumen_cambio_binario_agregado.csv, resumen_cambio_binario_robusto_con_magnitud.csv, resumen_cambio_arista_por_arista.csv, resumen_detalle_modelo_casos_interes.csv, resumen_esqueleto_persistente.csv y resumen_cruce_modelos_ssp245_vs_ssp585.csv; figures/cambio_estructural/contraste_<escenario>.png y parejas_cambio_estructural.png
Memoria: secciones 5.8, 5.9, 6.4 y Apéndice I.
"""

import numpy as np
import pandas as pd
import pickle
import itertools
import os
from pathlib import Path
import matplotlib.pyplot as plt
import bootstrap_grafos as boot
import comparacion_obs_modelos_hist as comh
from matplotlib.patches import Patch

# Definimos parámetros globales
umbral_p = 0.95                         # Umbral de P(F1_entre < F1_intra) que declara cambio estructural global.
umbral_presencia = 0.5                  # Umbral de frecuencia que declara si una arista está presente o no en el grafo
umbral_frac_primario = 0.5              # Umbral de fracción ponderada que declara cambio robusto en el criterio binario.
umbrales_frac = [0.4, 0.5, 0.6, 0.7]    # Barrido de umbrales de fracción ponderada para el análisis de sensibilidad.

# Nombres oficiales de los modelos
nombres_modelo = boot.nombres_modelos

def estado_cambio(f_hist, f_esc, umbral=umbral_presencia):
    # Determina si una arista desaparece, aparece, persiste o está ausente entre ventanas.
    h = f_hist >= umbral
    e = f_esc >= umbral
    if h and not e: return "desaparece"
    if e and not h: return "aparece"
    if h and e: return "persiste"
    return "ausente"

def frecuencia_arista(muestras, a, b, dirigida):
    # Calcula la frecuencia de aparición de una arista en las muestras de bootstrap.
    if dirigida:
        cond = lambda o, d, tau: tau >= 1 and o == a and d == b
    else:
        par = frozenset((a, b))
        cond = lambda o, d, tau: tau == 0 and frozenset((o, d)) == par
    presente = sum(1 for m in muestras if any(cond(o, d, tau) for (o, d, tau) in m.keys()))
    return presente / len(muestras)

def distribucion_f1(muestras_a, muestras_b, rng, misma_ventana=False):
    # Genera una distribución de valores de F1 emparejando muestras al azar para evaluar la variabilidad.
    vals = []
    n_pares = 500   # pares de muestras por distribucion
    for _ in range(n_pares):
            # Elegimos dos muestras al azar
        i = rng.integers(0, len(muestras_a)); j = rng.integers(0, len(muestras_b))
        if misma_ventana:
            while j == i: j = rng.integers(0, len(muestras_b))
        r = comh.analisis_enlaces(comh.normalizar_enlaces(muestras_a[i]), comh.normalizar_enlaces(muestras_b[j]))
        vals.append(r["topologia"]["f1"])
    return np.array(vals)

def prob_menor(dif, ruido, rng, n=20000):
    # Estima la probabilidad de que la diferencia entre ventanas sea menor o igual que el ruido interno.
    a = rng.choice(dif, n); b = rng.choice(ruido, n)
    return float((a < b).mean() + 0.5 * (a == b).mean())   # Los empates cuentan medio punto

def detalle_arista(origen, destino, tau, esc):
    # Extrae el detalle del comportamiento de una arista específica por modelo para un escenario dado.
    sub = df_ar[(df_ar.origen == origen) & (df_ar.destino == destino) & (df_ar.tau == tau) & (df_ar.escenario == esc)]
    return sub[["modelo", "f_hist", "f_esc", "delta_f", "estado"]].sort_values("delta_f")

def frec_pareja(o, d, tau, m, esc, df):
    # Frecuencia histórica y de escenario de una pareja concreta, para un modelo dado
    filas = df[(df.modelo == m) & (df.escenario == esc) & (df.origen == o) & (df.destino == d) & (df.tau == tau)]
    f_h = float(filas.f_hist.iloc[0]) if len(filas) else 0.0
    f_e = float(filas.f_esc.iloc[0]) if len(filas) else 0.0
    return f_h, f_e

if __name__ == '__main__':

    # Para este análisis usamos solo la capa de topología.

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    boot_dir = root_data / "data" / "indices" / "bootstrap"
    pcmci_dir = root_data / "data" / "indices" / "pcmci_plus"
    res_dir = root / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / "figures" / "cambio_estructural"; fig_dir.mkdir(parents=True, exist_ok=True)

    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
    escenarios = ["W_ssp245", "W_ssp585"]

    # Creamos un generador de números aleatorios con la misma semilla que en bootstrap
    rng = np.random.default_rng(boot.seed)

    # ANÁLISIS GLOBAL DE CAMBIO ESTRUCTURAL POR MODELO Y ESCENARIO: ¿Cambia el grafo entero bajo calentamiento?
    print('\n---- ANÁLISIS GLOBAL DE CAMBIO ESTRUCTURAL POR MODELO Y ESCENARIO ----\n')

    filas = []
    distribuciones = {}
    muestras_dict = {}   # (m, esc) -> (muestras_hist_eq, muestras_esc)
    for m in modelos:
        print(f"\n  {m}")

        # Cargamos las muestras de bootstrap del histórico (con y sin igualar el tamaño muestral al escenario)
        m_path = boot_dir / f"muestras_W_hist_{m}_B{boot.B}.pkl"
        with open(m_path, "rb") as f: m_hist = pickle.load(f)["muestras"]
        m_path = boot_dir / f"muestras_W_hist_{m}_igualado_B{boot.B}.pkl"
        with open(m_path, "rb") as f: m_hist_eq = pickle.load(f)["muestras"]

        # Ruido interno del histórico: distribución de F1 comparando el propio histórico consigo mismo
        ruido_int_h = distribucion_f1(m_hist_eq, m_hist_eq, rng, misma_ventana=True)

        for esc in escenarios:
            print(f"        {esc}")

            m_path = boot_dir / f"muestras_{esc}_{m}_B{boot.B}.pkl"
            with open(m_path, "rb") as f: m_esc = pickle.load(f)["muestras"]

            # Ruido interno del escenario y diferencia entre ventanas
            ruido_int_esc = distribucion_f1(m_esc, m_esc, rng, misma_ventana=True)
            ruido_int = np.concatenate([ruido_int_h, ruido_int_esc])  # ruidos juntos, solo como comparación con el criterio anterior
            dif_vent_eq = distribucion_f1(m_hist_eq, m_esc, rng)  # con muestreo igualado
            dif_vent_raw = distribucion_f1(m_hist, m_esc, rng)  # sin igualar, como control de artefacto de potencia

            # Contraste frente a cada ruido interno por separado. Nos quedamos con la menor probabilidad, que es la que da
            # el ruido de la ventana peor muestreada (la de menor F1 interno), es decir, el criterio más conservador.
            p_eq_h = prob_menor(dif_vent_eq, ruido_int_h, rng)
            p_eq_e = prob_menor(dif_vent_eq, ruido_int_esc, rng)
            p_raw_h = prob_menor(dif_vent_raw, ruido_int_h, rng)
            p_raw_e = prob_menor(dif_vent_raw, ruido_int_esc, rng)
            p_eq, ventana_ref = (p_eq_h, "hist") if p_eq_h <= p_eq_e else (p_eq_e, "esc")
            p_raw = min(p_raw_h, p_raw_e)
            p_eq_conjunto = prob_menor(dif_vent_eq, ruido_int, rng)  # criterio anterior, ruidos juntos
            cambio = p_eq >= umbral_p

            print(f"          - ruido interno: mediana hist={np.median(ruido_int_h):.3f}, mediana esc={np.median(ruido_int_esc):.3f}")
            print(f"          - dif entre ventanas: mediana={np.median(dif_vent_eq):.3f}  IQR=[{np.percentile(dif_vent_eq, 25):.3f}, {np.percentile(dif_vent_eq, 75):.3f}]  (igualado)")
            print(f"          - P frente a ruido hist={p_eq_h:.3f}, frente a ruido esc={p_eq_e:.3f} -> referencia {ventana_ref}")
            print(f"          - P(diferencia entre ventanas < ruido interno) = {p_eq:.3f} igualado | {p_raw:.3f} sin igualar | {p_eq_conjunto:.3f} ruidos juntos (criterio anterior)  ->  {'CAMBIO ESTRUCTURAL' if cambio else 'indistinguible del ruido'}")
            if p_raw >= umbral_p > p_eq:
                print("               - AVISO: el cambio aparece sin igualar y desaparece al igualar -> artefacto de potencia")

            filas.append({
                "modelo": m, "escenario": esc,
                "ruido_int_mediana": round(float(np.median(ruido_int)), 3),
                "ruido_hist_mediana": round(float(np.median(ruido_int_h)), 3),
                "ruido_esc_mediana": round(float(np.median(ruido_int_esc)), 3),
                "ruido_int_p25": round(float(np.percentile(ruido_int, 25)), 3),
                "ruido_int_p75": round(float(np.percentile(ruido_int, 75)), 3),
                "dif_igualado_mediana": round(float(np.median(dif_vent_eq)), 3),
                "dif_sin_igualar_mediana": round(float(np.median(dif_vent_raw)), 3),
                "P_hist_igualado": round(p_eq_h, 3),
                "P_esc_igualado": round(p_eq_e, 3),
                "ventana_referencia": ventana_ref,
                "P_menor_igualado": round(p_eq, 3),
                "P_menor_sin_igualar": round(p_raw, 3),
                "P_menor_igualado_conjunto": round(p_eq_conjunto, 3),
                "cambio_estructural": bool(cambio),
                "artefacto_potencia": bool(p_raw >= umbral_p > p_eq)
            })
            distribuciones[(m, esc)] = (ruido_int_h, ruido_int_esc, dif_vent_eq, dif_vent_raw)
            muestras_dict[(m, esc)] = (m_hist_eq, m_esc)
    df = pd.DataFrame(filas)
    df.to_csv(res_dir / "resumen_cambio_estructural_general.csv", index=False)

    # ANÁLISIS ARISTA A ARISTA: ¿qué relaciones concretas cambian bajo calentamiento?
    print('\n---- ANÁLISIS ARISTA A ARISTA ----\n')

    cambios = []
    nombres = None
    for m in modelos:
        with open(boot_dir / f"muestras_W_hist_{m}_igualado_B{boot.B}.pkl", "rb") as f:
            nombres = pickle.load(f)["nombres"]; break

    # Enlistamos aristas posibles
    pares_sin_dir = list(itertools.combinations(sorted(nombres), 2))   # contemporáneas,
    pares_con_dir = list(itertools.permutations(sorted(nombres), 2))   # desfasadas, con las dos direcciones

    for m in modelos:
        for esc in escenarios:
            m_hist_eq, m_esc = muestras_dict[(m, esc)]

            # Relaciones contemporáneas (tau=0, sin dirección resoluble)
            for a, b in pares_sin_dir:
                f_h = frecuencia_arista(m_hist_eq, a, b, dirigida=False)
                f_e = frecuencia_arista(m_esc, a, b, dirigida=False)
                if f_h == 0.0 and f_e == 0.0:
                    continue
                cambios.append({
                    "modelo": m,
                    "escenario": esc,
                    "origen": a,
                    "destino": b,
                    "tau": 0,
                    "f_hist": round(f_h, 3),
                    "f_esc": round(f_e, 3),
                    "delta_f": round(f_e - f_h, 3)
                })

            # Relaciones desfasadas (tau>=1, dirigidas)
            for o, d in pares_con_dir:
                f_h = frecuencia_arista(m_hist_eq, o, d, dirigida=True)
                f_e = frecuencia_arista(m_esc, o, d, dirigida=True)
                if f_h == 0.0 and f_e == 0.0:
                    continue
                cambios.append({
                    "modelo": m,
                    "escenario": esc,
                    "origen": o,
                    "destino": d,
                    "tau": "-",
                    "f_hist": round(f_h, 3),
                    "f_esc": round(f_e, 3),
                    "delta_f": round(f_e - f_h, 3)
                })

    df_ar = pd.DataFrame(cambios)
    df_ar["estado"] = [estado_cambio(r.f_hist, r.f_esc) for r in df_ar.itertuples()]

    # CRITERIO BINARIO DE CAMBIO: fracción ponderada de los modelos que "cambian" (aparece o desaparece), bajo los tres esquemas de peso.
    # El tipo dominante se decide por el esquema democracia.
    # Robusto si la fracción ponderada del tipo dominante supera el umbral bajo los TRES esquemas a la vez.
    print('\n---- CRITERIO BINARIO DE CAMBIO POR ARISTA Y ESCENARIO (fracción ponderada de modelos) ----\n')

    # Cargamos pesos para el cálculo del ensamble mean (EM). Triple reporte (democracia, solo calidad, calidad por independencia)
    ruta_pesos = res_dir / "resumen_pesos_EM.csv"; pesos = pd.read_csv(ruta_pesos, index_col=0)
    esquemas = ["democracia", "solo_calidad", "calidad_independencia"]

    agregado_bin = []
    for esc in escenarios:
        print(f'\n  {esc}')
        sub = df_ar[df_ar.escenario == esc]
        for clave, g in sub.groupby(["origen", "destino", "tau"]):
            o, d, tau = clave
            fila = {"escenario": esc, "origen": o, "destino": d, "tau": tau, "n_modelos": len(g)}
            frac = {"aparece": {}, "desaparece": {}}

            # Calculamos la fracción ponderada (según cada esquema de pesos) de aparición o desaparición
            for e in esquemas:
                w_todos = np.array([pesos.loc[mm, e] if mm in pesos.index else 0.0 for mm in g["modelo"]])
                for tipo in ("aparece", "desaparece"):
                    w_tipo = np.array([pesos.loc[mm, e] if mm in pesos.index else 0.0 for mm in g.loc[g["estado"] == tipo, "modelo"]])
                    frac[tipo][e] = float(w_tipo.sum() / w_todos.sum()) if w_todos.sum() > 0 else 0.0
                    fila[f"frac_{tipo}_{e}"] = round(frac[tipo][e], 3)

            # Determinamos el cambio dominante y evaluamos si cumple la robustez en todos los esquemas
            tipo_dom = "aparece" if frac["aparece"]["democracia"] >= frac["desaparece"]["democracia"] else "desaparece"
            hay_cambio = (frac["aparece"]["democracia"] + frac["desaparece"]["democracia"]) > 0
            fila["tipo"] = tipo_dom if hay_cambio else None

            for u in umbrales_frac:
                fila[f"robusto_{u}"] = bool(hay_cambio and all(frac[tipo_dom][e] > u for e in esquemas))

            fila["robusto"] = fila[f"robusto_{umbral_frac_primario}"]
            agregado_bin.append(fila)

    df_bin = pd.DataFrame(agregado_bin).sort_values("frac_aparece_democracia", ascending=False)
    df_bin.to_csv(res_dir / "resumen_cambio_binario_agregado.csv", index=False)

    print(f"       - Cambios binarios robustos (capa binaria, umbral primario {umbral_frac_primario})")
    for u in umbrales_frac:
        rob = df_bin[df_bin[f"robusto_{u}"]]; print(f"\n              - umbral {u}: {len(rob)} cambios robustos")
        if not rob.empty:
            print(rob[["escenario", "origen", "destino", "tau", "tipo", "frac_aparece_democracia", "frac_desaparece_democracia"]].to_string(index=False))

    # ANÁLISIS POR MODELO DE LAS ARISTAS IMPORTANTES:  umbral 0.4 (si sale en un escenario en el otro también es interesante)
    print('\n---- ANÁLISIS POR MODELO DE LAS ARISTAS DE INTERÉS ----\n')

    interesantes = df_bin[df_bin[f"robusto_{min(umbrales_frac)}"]][["origen", "destino", "tau"]].drop_duplicates()
    interesantes = interesantes.merge(pd.DataFrame({"escenario": escenarios}), how="cross")

    detalles_interes = []
    for _, r in interesantes.iterrows():
        d = detalle_arista(r.origen, r.destino, r.tau, r.escenario)
        d.insert(0, "escenario", r.escenario)
        d.insert(1, "origen", r.origen)
        d.insert(2, "destino", r.destino)
        d.insert(3, "tau", r.tau)
        detalles_interes.append(d)

    if detalles_interes:
        df_detalle_interes = pd.concat(detalles_interes, ignore_index=True)
        df_detalle_interes.to_csv(res_dir / "resumen_detalle_modelo_casos_interes.csv", index=False)

    # ESQUELETO PERSISTENTE: relaciones cuyo comportamiento mayoritario es persistir en los dos escenarios (no implica que persistan en todos los modelos)
    print('\n---- ESQUELETO PERSISTENTE ----\n')

    esqueleto = []
    for clave, g in df_ar.groupby(["origen", "destino", "tau"]):
        o, d, tau = clave
        fila = {"origen": o, "destino": d, "tau": tau}
        ok = True

        # Verificamos si el enlace persiste en todos los escenarios
        for esc in escenarios:
            ge = g[g.escenario == esc]
            if ge.empty:
                ok = False
                continue
            persiste = ge["estado"] == "persiste"
            fracs = []

            # Calculamos la fracción ponderada de persistencia para cada esquema de peso
            for e in esquemas:
                w = np.array([pesos.loc[mm, e] if mm in pesos.index else 0.0 for mm in ge["modelo"]])
                w_p = np.array([pesos.loc[mm, e] if mm in pesos.index else 0.0 for mm in ge.loc[persiste, "modelo"]])
                f = float(w_p.sum() / w.sum()) if w.sum() > 0 else 0.0
                fila[f"frac_persiste_{esc}_{e}"] = round(f, 3)
                fracs.append(f)

            # Exigir que todos los esquemas superen el umbral primario de robustez
            if min(fracs) <= umbral_frac_primario:
                ok = False

        fila["esqueleto"] = bool(ok)
        esqueleto.append(fila)

    df_esq = pd.DataFrame(esqueleto)
    df_esq[df_esq.esqueleto].to_csv(res_dir / "resumen_esqueleto_persistente.csv", index=False)
    print(f"      - Esqueleto persistente: {int(df_esq.esqueleto.sum())} relaciones con comportamiento mayoritario de persistir en los dos escenarios.")
    if df_esq.esqueleto.sum():
        print(df_esq[df_esq.esqueleto][["origen", "destino", "tau"]].to_string(index=False))

    # REFINAMIENTO DE MAGNITUD (delta_f continuo): en vez del binario aparece/desaparece, se mira
    # cuánto se mueve la frecuencia de la arista, con barrido de sensibilidad del umbral mínimo.
    print('\n---- REFINAMIENTO DE MAGNITUD (delta_f continuo) ----\n')

    agregado = []
    umbrales = [0.1, 0.15, 0.2, 0.25]           # sensibilidad del umbral de magnitud mínima
    clave_grupo = ["origen", "destino", "tau"]

    print(f"      - Cambios agregados robustos bajo los tres esquemas de ponderación")
    for esc in escenarios:
        sub = df_ar[df_ar.escenario == esc]
        for clave, g in sub.groupby(clave_grupo):
            o, d, tau = clave
            fila = {"escenario": esc, "origen": o, "destino": d, "tau": tau, "n_modelos": len(g)}

            # Calculamos los valores promedio ponderados de cambio y frecuencias para cada esquema de peso
            for e in esquemas:
                w = np.array([pesos.loc[mm, e] if mm in pesos.index else 0.0 for mm in g["modelo"]])
                fila[f"delta_f_{e}"] = round(float(np.sum(w * g["delta_f"].values) / w.sum()), 3) if w.sum() > 0 else np.nan
                fila[f"f_hist_{e}"] = round(float(np.sum(w * g["f_hist"].values) / w.sum()), 3) if w.sum() > 0 else np.nan
                fila[f"f_esc_{e}"] = round(float(np.sum(w * g["f_esc"].values) / w.sum()), 3) if w.sum() > 0 else np.nan

            # Validamos robustez: exige que los tres esquemas coincidan en el signo y superen el umbral mínimo
            signos = [np.sign(fila[f"delta_f_{e}"]) for e in esquemas]
            magnitudes = [abs(fila[f"delta_f_{e}"]) for e in esquemas]
            mismo_signo = len(set(signos)) == 1

            for u in umbrales:
                fila[f"robusto_{u}"] = bool(mismo_signo and min(magnitudes) > u)

            # Aplicamos umbral de cambio de frecuencia en 0.2 (20% de cambio de frecuencia). El umbral nos quita los cambios pequeños tomándolos como ruido.
            fila["robusto"] = fila["robusto_0.2"]
            agregado.append(fila)

    df_ag = pd.DataFrame(agregado).sort_values("delta_f_democracia")

    # Fusionamos resultados con el detalle individual (análisis por arista)
    piv = df_ar.pivot_table(index=["escenario", "origen", "destino", "tau"], columns="modelo", values="delta_f")
    piv.columns = [f"delta_f_{c}" for c in piv.columns]
    piv = piv.reset_index()
    df_ag = df_ag.merge(piv, on=["escenario", "origen", "destino", "tau"], how="left").sort_values("delta_f_democracia")
    df_ag.to_csv(res_dir / "resumen_cambio_arista_por_arista.csv", index=False)

    for u in umbrales:
        rob = df_ag[df_ag[f"robusto_{u}"]]; print(f"\n              - umbral {u}: {len(rob)} cambios robustos")
        if not rob.empty:
            print(rob[["escenario", "origen", "destino", "tau", "f_hist_democracia", "f_esc_democracia",
                    "delta_f_democracia", "delta_f_solo_calidad", "delta_f_calidad_independencia"]].to_string(index=False))

    # Fusion los resultados del criterio binario con los del continuo: añade su delta_f de magnitud, como refinamiento del veredicto.
    print('\n---- FUSIÓN BINARIO + CONTINUO ----\n')

    df_bin_robusto = df_bin[df_bin.robusto].merge(
        df_ag[["escenario", "origen", "destino", "tau", "delta_f_democracia", "delta_f_solo_calidad", "delta_f_calidad_independencia"]],
        on=["escenario", "origen", "destino", "tau"], how="left")
    df_bin_robusto.to_csv(res_dir / "resumen_cambio_binario_robusto_con_magnitud.csv", index=False)
    print(f"      - Cambios robustos por criterio binario (umbral {umbral_frac_primario}), con magnitud de refinamiento.")
    if len(df_bin_robusto): print(df_bin_robusto[["escenario", "origen", "destino", "tau", "tipo", "frac_aparece_democracia",
                        "frac_desaparece_democracia", "delta_f_democracia"]].to_string(index=False))

    # CRUCE DE MODELOS ENTRE ESCENARIOS: para cada arista con cambio robusto, qué modelos concretos sostienen ese cambio en ssp245 y en ssp585
    print('\n---- CRUCE DE MODELOS POR ARISTA (SSP2-4.5 vs SSP5-8.5) ----\n')

    aristas_robustas = df_bin[df_bin.robusto][["origen", "destino", "tau", "tipo"]].drop_duplicates()
    filas_cruce = []

    for _, ar in aristas_robustas.iterrows():
        fila = {"origen": ar.origen, "destino": ar.destino, "tau": ar.tau, "tipo": ar.tipo}

        for esc in escenarios:
            det = detalle_arista(ar.origen, ar.destino, ar.tau, esc).set_index("modelo")
            sostienen = [f"{m} ({det.loc[m, 'delta_f']:+.3f})" for m in modelos
                         if m in det.index and det.loc[m, "estado"] == ar.tipo]
            fila[f"modelos_{esc}"] = ", ".join(sostienen) if sostienen else "ninguno"
        filas_cruce.append(fila)

    df_cruce = pd.DataFrame(filas_cruce)
    df_cruce.to_csv(res_dir / "resumen_cruce_modelos_ssp245_vs_ssp585.csv", index=False)
    print(f"      - Cruce de modelos por arista (criterio binario oficial).")
    print(df_cruce.to_string(index=False))

    # FIGURAS
    print('\n---- FIGURAS ----\n')

    parejas = sorted(set((r.origen, r.destino, r.tau) for r in df_bin[df_bin.robusto].itertuples()))
    ventanas = ["histórico", "SSP2-4.5", "SSP5-8.5"]
    colores_v = ["#7C7C7C", "#7FB2D5", "#2E6F9E"]

    # Figura 1: barras de frecuencia por modelo y ventana, una subfigura por pareja robusta
    x = np.arange(len(modelos))
    w = 0.27
    if parejas:
        fig, axes = plt.subplots(1, len(parejas), figsize=(4.2 * len(parejas), 4.6), sharey=True)
        for ax, (o, d, tau) in zip(np.atleast_1d(axes), parejas):
            par = (o, d)
            h = [frec_pareja(o, d, tau, m, "W_ssp245", df_ar)[0] for m in modelos]
            s245 = [frec_pareja(o, d, tau, m, "W_ssp245", df_ar)[1] for m in modelos]
            s585 = [frec_pareja(o, d, tau, m, "W_ssp585", df_ar)[1] for m in modelos]
            for k, (vals, nom, col) in enumerate(zip([h, s245, s585], ventanas, colores_v)):
                ax.bar(x + (k - 1) * w, vals, width=w, color=col, label=nom if (o, d, tau) == parejas[0] else None)
            ax.set_xticks(x)
            ax.set_xticklabels([nombres_modelo[m] for m in modelos], fontsize=6.5, rotation=0)
            conector = "→" if tau == "-" else "-" #  →
            o_t = o.replace("_PC2", ""); d_t = d.replace("_PC2", "")
            ax.set_title(f"{o_t} {conector} {d_t}", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.axhline(0.5, color="#AA0000", linestyle=":", linewidth=1)  # umbral de fragilidad
        np.atleast_1d(axes)[0].set_ylabel("frecuencia de aparición", fontsize=9)
        fig.legend(loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"Robustez de las parejas con cambio estructural (criterio binario, umbral {umbral_frac_primario}) por modelo y ventana   "
                     f"(histórico con muestreo igualado, B = {boot.B})", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / "parejas_cambio_estructural.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Figura 2: violín de las distribuciones F1 (ruido histórico, ruido escenario, diferencia entre ventanas), una por escenario
    for esc in escenarios:
        sub = [(m, distribuciones[(m, esc)]) for m in modelos if (m, esc) in distribuciones]
        if not sub: continue
        fig, axes = plt.subplots(2, 4, figsize=(9, 6.2), sharey=True)
        axes = axes.ravel()
        for ax, (m, (i_h, i_e, e_eq, _)) in zip(axes, sub):
            v = ax.violinplot([i_h, i_e, e_eq], positions=[0, 1, 2], showmedians=True)

            # los dos ruidos internos en gris, la diferencia entre ventanas en azul
            for k, cuerpo in enumerate(v["bodies"]):
                cuerpo.set_facecolor("#BBBBBB" if k < 2 else "#2E6F9E")
                cuerpo.set_alpha(0.75)
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["ruido\nhistórico", "ruido\nescenario", "entre\nventanas"], fontsize=9)
            ax.tick_params(axis="y", labelsize=11)
            p = float(df[(df.modelo == m) & (df.escenario == esc)]["P_menor_igualado"].iloc[0])
            ax.set_title(f"{nombres_modelo[m]}\nP = {p:.2f}", fontsize=12, color="tab:red" if p >= umbral_p else "black")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
        for ax in (axes[0], axes[4]):
            ax.set_ylabel("F1 (topología)", fontsize=12)

        # el octavo panel queda libre para la leyenda
        axes[7].axis("off")
        axes[7].legend(handles=[Patch(facecolor="#BBBBBB", alpha=0.75, label="ruido interno"),
                                Patch(facecolor="#2E6F9E", alpha=0.75, label="entre ventanas")],
                       loc="center", fontsize=12, frameon=False)

        fig.tight_layout()
        fig.savefig(fig_dir / f"contraste_{esc}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
