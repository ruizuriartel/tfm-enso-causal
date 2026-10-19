"""
comparacion_obs_modelos_hist.py
Calcula el F1 de cada modelo en el histórico frente al grafo observacional en tres capas (topología, topología y desfase, topología, desfase y signo; la primera es la principal), su distribución sobre las réplicas del bootstrap y la probabilidad de separación entre cada par de modelos. Incluye la sensibilidad sin la adyacencia CHI200-PSI200 y calcula los pesos del ensemble (democracia, solo calidad, calidad e independencia por familia) que usa comparacion_escenarios.py.
Entradas: data/indices/pcmci_plus/ (de pcmci_plus.py) y data/indices/bootstrap/muestras_W_hist_<modelo>_B500.pkl (de bootstrap_grafos.py)
Salidas: resultados/resumen_general_mod_vs_obs.csv, resumen_enlaces_mod_vs_obs.csv, resumen_f1_sin_chi_psi.csv, resumen_solapamiento_distribucion_F1_mod_vs_obs.csv y resumen_pesos_EM.csv; figures/grafos_causales/f1_capas_hist_vs_obs.png
Memoria: secciones 5.7, 5.8 y Apéndice H.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import bootstrap_grafos as boot

# Definimos parámetros globales
autoenlaces = False
tolerancia_tau = 0
fdr = False
B = boot.B

# Calculamos el F1 (grafo de referencia vs. modelos) utilizando tres criterios distintos para definir las aristas emparejadas:
capas = [
    ("topologia", dict(usar_tau=False, usar_signo=False)),                  # topología (origen, destino y dirección)
    ("topologia_desfase", dict(usar_tau=True,  usar_signo=False)),          # topología + desfase
    ("topologia_desfase_signo", dict(usar_tau=True,  usar_signo=True))      # topología + desfase + signo
]

    # La capa primaria es la de topología: la respuesta atmosférica a ENSO es dominantemente contemporánea y las aristas
        # dirigidas con desfase son débiles y dependientes de la configuración, asi que exigir coincidencia de desfase penaliza
        # diferencias que no son estructurales. Las otras dos se reportan como comprobaciones de robustez.

def normalizar_enlaces(fuente, incluir_autoenlaces=autoenlaces, solo_fdr=fdr):
    # Estandariza el formato de los enlaces causales a partir de un diccionario o DataFrame

    # Cargamos datos crudos
    if isinstance(fuente, dict):
        crudo = [(o, d, int(tau), float(info["val"]), True) for (o, d, tau), info in fuente.items()]
    else:
        df = fuente if incluir_autoenlaces else fuente[fuente["origen"] != fuente["destino"]]
        if solo_fdr:
            df = df[df["fdr_significativo"] == True]
        crudo = [(r.origen, r.destino, int(r.tau), float(r.val), True) for r in df.itertuples()]

    enlaces = []
    vistos = set()
    for o, d, tau, val, _ in crudo:

        # Descarte opcional de autoenlaces
        if not incluir_autoenlaces and o == d:
            continue

        # Enlaces contemporáneos no tienen dirección
        par = tuple(sorted([o, d])) if tau == 0 else (o, d)

        # Quitamos duplicados
        k = (par, tau, val > 0)
        if k in vistos:
            continue
        vistos.add(k)

        # Construimos diccionario normalizado de salida
        enlaces.append({
            "par": par,
            "tau": tau,
            "signo": bool(val > 0),
            "contemporaneo": tau == 0
        })
    return enlaces

def emparejar(obs, mod, usar_tau, usar_signo, tolerancia=tolerancia_tau):
    # Empareja los enlaces entre el grafo de observaciones y el de un modelo según la capa calculando TP, FP y FN.

    candidatos = []
    for i, a in enumerate(obs):
        for j, b in enumerate(mod):
                # Separamos contemporáneas de retardadas
            if a["contemporaneo"] != b["contemporaneo"]:
                continue
                # Mismo origen y destino para direccionadas
            if a["contemporaneo"]:
                if set(a["par"]) != set(b["par"]):
                    continue
            elif a["par"] != b["par"]:
                continue
                # Análisis de signo
            if usar_signo and a["signo"] != b["signo"]:
                continue
                # Tolerancia opcional del retardo para enlazar en la capa de desfase
            dtau = abs(a["tau"] - b["tau"])
            if usar_tau and dtau > tolerancia:
                continue
                # Guardamos los candidatos a emparejar
            candidatos.append((dtau, i, j))
    candidatos.sort(key=lambda x: x[0])

    # Emparejamos las aristas uno con otra para evitar duplicados
    obs_usado, mod_usado, pares = set(), set(), []
    for dtau, i, j in candidatos:
        if i in obs_usado or j in mod_usado:
            continue
        obs_usado.add(i)
        mod_usado.add(j)
        pares.append((i, j, dtau))

    return len(pares), len(mod) - len(mod_usado), len(obs) - len(obs_usado), pares  # TP, FP, FN, pares de TP

def calcular_f1(tp, fp, fn):
    # Calcula la precisión, exhaustividad (recall) y el F1-score a partir de TP, FP y FN.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    exhaustividad = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * exhaustividad / (precision + exhaustividad) if (precision + exhaustividad) > 0 else 0.0)
    return precision, exhaustividad, f1

def colapso(enlaces, usar_signo):
    # Colapsa / fusiona los enlaces no contemporáneos para eliminar redundancias al evaluar capas sin desfase.
    vistos = set()
    out = []
    for e in sorted(enlaces, key=lambda x: x["tau"]):
        k = (e["contemporaneo"], e["par"], e["signo"] if usar_signo else None)
        if k in vistos:
            continue
        vistos.add(k)
        out.append(e)
    return out

def excluir_par(enlaces, nodos=("CHI200", "PSI200"), tau=0):
    # Filtra y excluye un par específico de nodos (ej. CHI200-PSI200) para análisis de sensibilidad.
    par = frozenset(nodos)
    return [e for e in enlaces if not (e["contemporaneo"] and e["tau"] == tau and set(e["par"]) == par)]

def analisis_enlaces(obs, mod, capas=capas):
    # Ejecuta la evaluación completa de enlaces para todas las capas definidas.
    out = {}
    for nombre, cfg in capas:
        a, b = (obs, mod) if cfg["usar_tau"] else (colapso(obs, cfg["usar_signo"]), colapso(mod, cfg["usar_signo"]))
        tp, fp, fn, pares = emparejar(a, b, **cfg)
        p, r, f1 = calcular_f1(tp, fp, fn)
        out[nombre] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "pares": pares,
            "obs_usado": a,
            "mod_usado": b
        }
    return out

def detalle_enlaces(obs, mod, pares, modelo, capa):
    # Genera una lista detallada con el estado de cada enlace (TP, FP, FN) para su exportación.
    usados_obs = {i for i, _, _ in pares}
    usados_mod = {j for _, j, _ in pares}
    filas = []

    # Verdaderos positivos
    for i, j, dtau in pares:
        a, b = obs[i], mod[j]
        filas.append({"modelo": modelo, "capa": capa, "tipo": "TP",
                      "nodo_a": b["par"][0], "nodo_b": b["par"][1],
                      "tipo_enlace": "contemporaneo" if b["contemporaneo"] else "dirigido",
                      "tau_obs": a["tau"], "tau_modelo": b["tau"], "dtau": dtau,
                      "signo_obs": "+" if a["signo"] else "-", "signo_modelo": "+" if b["signo"] else "-"})

    # Falsos positivos
    for j, b in enumerate(mod):
        if j in usados_mod: continue
        filas.append({"modelo": modelo, "capa": capa, "tipo": "FP",
                      "nodo_a": b["par"][0], "nodo_b": b["par"][1],
                      "tipo_enlace": "contemporaneo" if b["contemporaneo"] else "dirigido",
                      "tau_obs": None, "tau_modelo": b["tau"], "dtau": None,
                      "signo_obs": None, "signo_modelo": "+" if b["signo"] else "-"})

   # Falsos negativos
    for i, a in enumerate(obs):
        if i in usados_obs: continue
        filas.append({"modelo": modelo, "capa": capa, "tipo": "FN",
                      "nodo_a": a["par"][0], "nodo_b": a["par"][1],
                      "tipo_enlace": "contemporaneo" if a["contemporaneo"] else "dirigido",
                      "tau_obs": a["tau"], "tau_modelo": None, "dtau": None,
                      "signo_obs": "+" if a["signo"] else "-", "signo_modelo": None})
    return filas

def plot_f1_capas(df, fig_path, dist=None):
    # Genera un gráfico de barras horizontales comparando el F1 de las tres capas, incluyendo barras de error del bootstrap.

    nombres = [c for c, _ in capas]
    orden = df[df.capa == nombres[0]].sort_values("f1", ascending=True)["modelo"].tolist()

    y = np.arange(len(orden))
    h = 0.26

    fig, ax = plt.subplots(figsize=(10, 0.85 * len(orden) + 2.5))
    colores = ["#2E6F9E", "#7FB2D5", "#C9DCEA"]

    for k, capa in enumerate(nombres):
        sub = df[df.capa == capa].set_index("modelo").loc[orden]
        yk = y + (1 - k) * h

        ax.barh(yk, sub["f1"], height=h, color=colores[k], label=capa.replace("topologia", "topología").replace("_", " + "))

        # Añadir opcionalmente barras de error
        if dist is not None and all(m in dist for m in orden):
            med = np.array([np.median(dist[m][capa]) for m in orden])
            p25 = np.array([np.percentile(dist[m][capa], 25) for m in orden])
            p75 = np.array([np.percentile(dist[m][capa], 75) for m in orden])

            ax.errorbar(med, yk, xerr=[med - p25, p75 - med], fmt="D", ms=3.5, color="#1A1A1A",
                        ecolor="#1A1A1A", elinewidth=1.1, capsize=2.5, zorder=5)

        # Añadimos etiquetas
        for yi, (f1, tp, fp, fn) in zip(y + (1 - k) * h, zip(sub["f1"], sub["TP"], sub["FP"], sub["FN"])):
            ax.text(0.02, yi, f"{f1:.2f}  ({tp}/{fp}/{fn})", va="center", fontsize=7)

    # Leyenda
    if dist is not None:
        ax.errorbar([], [], xerr=[[]], fmt="D", ms=3.5, color="#1A1A1A", ecolor="#1A1A1A",
                    elinewidth=1.1, capsize=2.5, label=f"mediana y rango intercuartílico (B = {B})")

    ax.set_yticks(y)
    ax.set_yticklabels([boot.nombres_modelos[m] for m in orden], fontsize=9)
    ax.set_xlim(0, 1.12); ax.set_xlabel("F1-score")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    pcmci_dir = root_data / "data" / "indices" / "pcmci_plus"
    fig_dir = root / "figures" / "grafos_causales"; fig_dir.mkdir(parents=True, exist_ok=True)
    periodo, ventana = "historical", "W_hist"
    boot_dir = root_data / "data" / "indices" / "bootstrap"
    res_dir = root / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)

    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]

    # Comparación entre el grafo observacional de referencia y los modelos en el periodo histórico
    obs_path = pcmci_dir / "obs" / "enlaces_W_obs.csv"
    obs = normalizar_enlaces(pd.read_csv(obs_path))
    n_cont = sum(1 for e in obs if e["contemporaneo"])
    print(f"\n----REFERENCIA OBSERVACIONAL----\n")
    print(f"    {len(obs)} aristas ({n_cont} contemporáneas, {len(obs)-n_cont} dirigidas)"
          f"{'  [solo FDR]' if fdr else ''}\n")

    filas, detalles, filas_sens = [], [], []
    for m in modelos:
        ruta = pcmci_dir / periodo / f"enlaces_{ventana}_{m}.csv"
        mod = normalizar_enlaces(pd.read_csv(ruta))
        res = analisis_enlaces(obs, mod)

        for capa, r in res.items():
            filas.append({"modelo": m, "capa": capa,
                          "n_aristas_modelo": len(r["mod_usado"]),
                          "TP": r["TP"], "FP": r["FP"], "FN": r["FN"],
                          "precision": round(r["precision"], 3),
                          "recall": round(r["recall"], 3),
                          "f1": round(r["f1"], 3)
                          })
            detalles.extend(detalle_enlaces(r["obs_usado"], r["mod_usado"], r["pares"], m, capa))

        # Análisis de sensibilidad excluyendo el par CHI200-PSI200 (solo en topología)
        res_sin = analisis_enlaces(excluir_par(obs), excluir_par(mod), capas=[capas[0]])
        r_sin = res_sin["topologia"]
        filas_sens.append({"modelo": m,
                           "TP": r_sin["TP"], "FP": r_sin["FP"], "FN": r_sin["FN"],
                           "precision": round(r_sin["precision"], 3),
                           "recall": round(r_sin["recall"], 3),
                           "f1": round(r_sin["f1"], 3)
                           })

    df = pd.DataFrame(filas)
    for capa, _ in capas:
        print(f"\n----F1 DE LA CAPA: {capa}----\n")
        print(df[df.capa == capa].sort_values("f1", ascending=False).to_string(index=False))

    df.to_csv(res_dir / "resumen_general_mod_vs_obs.csv", index=False)
    pd.DataFrame(detalles).to_csv(res_dir / "resumen_enlaces_mod_vs_obs.csv", index=False)
    df_sens = pd.DataFrame(filas_sens).sort_values("f1", ascending=False)
    df_sens.to_csv(res_dir / "resumen_f1_sin_chi_psi.csv", index=False)

    print(f"\n----SENSIBILIDAD SIN LA ADYACENCIA CHI200-PSI200----\n")
    print(df_sens.to_string(index=False))
    rango_con = df[df.capa == "topologia"].f1
    print(f"    - rango F1 CON CHI-PSI: [{rango_con.min():.3f}, {rango_con.max():.3f}]  orden: {df[df.capa == 'topologia'].sort_values('f1', ascending=False).modelo.tolist()}")
    print(f"    - rango F1 SIN CHI-PSI: [{df_sens.f1.min():.3f}, {df_sens.f1.max():.3f}]  orden: {df_sens.modelo.tolist()}")

    # Análisis de la distribución de F1 utilizando las réplicas del bootstrap de cada modelo (comparación entre W_obs puntual y las replicas)
    print(f"\n----DISTRIBUCIÓN DEL F1 (CAPA DE TOPOLOGÍA)----\n")
    dist = {}
    for m in modelos:
        f = boot_dir / f"muestras_W_hist_{m}_B{B}.pkl"
        with open(f, "rb") as fh:
            reps = pickle.load(fh)["muestras"]
        vals = {c: [] for c, _ in capas}
        for rep in reps:
            r = analisis_enlaces(obs, normalizar_enlaces(rep))
            for c, _ in capas:
                vals[c].append(r[c]["f1"])
        dist[m] = {c: np.array(v) for c, v in vals.items()}
        t = dist[m]["topologia"]
        print(f"    - {m:<20}: mediana={np.median(t):.3f}  IQR=[{np.percentile(t, 25):.3f}, {np.percentile(t, 75):.3f}]")

    # Evaluación estadística de solapamiento entre las distribuciones de F1 de los modelos
    orden_m = sorted(dist, key=lambda m: -np.median(dist[m]["topologia"]))
    comp = []
    rng = np.random.default_rng(11)
    for i, a in enumerate(orden_m):
        for b in orden_m[i + 1:]:
            p = float((rng.choice(dist[a]["topologia"], 20000) > rng.choice(dist[b]["topologia"], 20000)).mean())
            comp.append({
                "mejor": a,
                "peor": b,
                "P_a_mayor_b": round(p, 3),
                "separados": bool(p > 0.95)
            })

    pd.DataFrame(comp).to_csv(res_dir / "resumen_solapamiento_distribucion_F1_mod_vs_obs.csv", index=False)
    plot_f1_capas(df, fig_dir / "f1_capas_hist_vs_obs.png", dist=dist)

    # Cálculo de los pesos del ensemble basados en tres criterios diferentes para la ponderación del ensamble (EM)
        # Usamos solo la capa de topología
    top = df[df.capa == "topologia"].set_index("modelo")["f1"]

    # F1 score
    pesos = pd.DataFrame({"f1_topologia": top})
    pesos["f1_mediana_boot"] = pd.Series({m: np.median(dist[m]["topologia"]) for m in dist})
    pesos["f1_p25"] = pd.Series({m: np.percentile(dist[m]["topologia"], 25) for m in dist})
    pesos["f1_p75"] = pd.Series({m: np.percentile(dist[m]["topologia"], 75) for m in dist})

    # Pesos bajo tres criterios:
        # DEMOCRACIA -> distribución uniforme de pesos
    pesos["democracia"] = 1 / len(top)
        # SOLO CALIDAD -> ponderación basada en el rendimiento puro, f1 de topología normalizado
    pesos["solo_calidad"] = top / top.sum()
        # CALIDAD E INDEPENDENCIA -> corrección por familia de modelos, agrupando los modelos para penalizar la redundancia.
    familias = {"ukesm1_0_ll": "HadAM", "access_cm2": "HadAM", "hadgem3_gc31_ll": "HadAM",
                "ec_earth3_cc": "ECMWF", "cnrm_esm2_1": "ECMWF",
                "noresm2_mm": "CESM/CAM", "miroc6": "MIROC"}
    fam = {m: familias.get(m, m) for m in top.index}
    tam = pd.Series(fam).value_counts()
    indep = pd.Series({m: 1 / tam[fam[m]] for m in top.index})
    pesos["calidad_independencia"] = (top * indep) / (top * indep).sum()
    pesos["familia"] = pd.Series(fam)

    pesos.round(4).to_csv(res_dir / "resumen_pesos_EM.csv")
    print(f"\n----PESOS DEL ENSEMBLE----\n")
    print(pesos.round(3).to_string())
