"""
diagnostico_aristas_ausentes_FP.py
Reproduce las cifras de robustez del grafo observacional (frecuencia en el bootstrap de cada arista del grafo puntual, aristas ausentes de él y fracción FDR de los enlaces con desfase), busca las aristas fantasma de cada modelo en el histórico (frecuencia de al menos 0.4 en el bootstrap completo pero ausentes del grafo puntual) y calcula la frecuencia en el bootstrap de los falsos positivos de la capa de topología frente a la observación. La frecuencia de un enlace con desfase es la de su τ concreto.
Entradas: data/indices/bootstrap/frecuencias_W_obs_B500.csv y frecuencias_W_hist_<modelo>_B500.csv (de bootstrap_grafos.py), data/indices/pcmci_plus/obs/ y historical/ (de pcmci_plus.py) y resultados/resumen_enlaces_mod_vs_obs.csv (de comparacion_obs_modelos_hist.py)
Salidas: solo consola
Memoria: secciones 5.7, 6.1 y 6.3.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import bootstrap_grafos as boot

# Definimos directorios de entrada y salida
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

boot_dir = root_data / "data" / "indices" / "bootstrap"
pcmci_base = root_data / "data" / "indices" / "pcmci_plus"
pcmci_dir = pcmci_base / "historical"
res_dir = root / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)

# Parámetros del análisis
B = boot.B
f_min = 0.4     # umbral de frecuencia mínima para detectar "aristas fantasma" (frecuentes en el bootstrap pero ausentes en el grafo puntual).

modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]

# Generamos clave común para los enlaces: las contemporáneas se ordenan alfabéticamente en los dos lados
        # Las aristas contemporáneas (tau = 0) no tienen dirección estricta, por lo que se ordenan alfabéticamente.
clave = lambda o, d, t: (min(o, d), max(o, d), int(t)) if t == 0 else (o, d, int(t))


# REFERENCIA OBSERVACIONAL: cifras de la sección 6.1 (cada tau por separado)
print(f"\n----REFERENCIA OBSERVACIONAL----\n")
f = pd.read_csv(boot_dir / f"frecuencias_W_obs_B{B}.csv")
p = pd.read_csv(pcmci_base / "obs" / "enlaces_W_obs.csv")
p = p[p.origen != p.destino]

# Misma clave para el grafo puntual y para el bootstrap
for d in (f, p):
    d["k1"] = [clave(r.origen, r.destino, r.tau)[0] for r in d.itertuples()]
    d["k2"] = [clave(r.origen, r.destino, r.tau)[1] for r in d.itertuples()]
claves_p = set(zip(p.k1, p.k2, p.tau))
f["en_puntual"] = [(r.k1, r.k2, r.tau) in claves_p for r in f.itertuples()]

mitad = f[f.frecuencia >= 0.5]
print(f"    - aristas distintas vistas en el bootstrap: {len(f)}")
print(f"    - con frecuencia >= 0.5: {len(mitad)} (todas en el grafo puntual: {bool(mitad.en_puntual.all())})")
print(f"    - frecuencia máxima de las ausentes del grafo puntual: {f[~f.en_puntual].frecuencia.max():.3f}")

j = p[["k1", "k2", "tau", "val", "q"]].merge(f[["k1", "k2", "tau", "frecuencia", "coherencia_signo", "frac_fdr"]],
                                             on=["k1", "k2", "tau"], how="left")
j["banda"] = np.select([j.frecuencia >= 0.9, j.frecuencia >= 0.5], ["robusta", "media"], "frágil")
j = j.sort_values("frecuencia", ascending=False)
print(f"\n    - {len(j)} aristas del grafo puntual: {j.banda.value_counts().to_dict()}")
print(f"    - coherencia de signo mínima: {j.coherencia_signo.min():.3f}\n")
df = j[["k1", "k2", "tau", "val", "q", "frecuencia", "banda", "coherencia_signo"]]
print("\n".join(" " * 4 + linea for linea in df.to_string(index=False).split("\n")))

rob = j[j.banda == "robusta"]
print(f"\n    - |val| mínimo de las robustas: {rob.val.abs().min():.3f} | máximo del resto: {j[j.banda != 'robusta'].val.abs().max():.3f}")
print(f"\n    - Enlaces con desfase (la corrección FDR solo se aplica a estos):\n")
df = j[j.tau >= 1][["k1", "k2", "tau", "val", "q", "frecuencia", "frac_fdr"]]
print("\n".join(" " * 4 + linea for linea in df.to_string(index=False).split("\n")))

# ARISTAS FANTASMA por modelo: frecuentes en el bootstrap pero ausentes del grafo puntual
print(f"\n----ARISTAS FANTASMA----\n")
for m in modelos:
    print(f'\n   {m}')
    rf = boot_dir / f"frecuencias_W_hist_{m}_B{B}.csv"
    rp = pcmci_dir / f"enlaces_W_hist_{m}.csv"
    if not (rf.exists() and rp.exists()):
        print(f"        [FALTA] {m}")
        continue

    f = pd.read_csv(rf)
    p = pd.read_csv(rp)
    p = p[p.origen != p.destino]

    # Conjunto de enlaces presentes en el grafo puntual original del modelo
    puntual = {clave(r.origen, r.destino, r.tau) for r in p.itertuples()}
    f["en_puntual"] = [clave(r.origen, r.destino, r.tau) in puntual for r in f.itertuples()]

    # Filtramos aristas fantasma: alta frecuencia en el bootstrap (>= f_min) pero ausentes en el grafo puntual
    fant = f[(~f.en_puntual) & (f.frecuencia >= f_min)].copy(); fant.insert(0, "modelo", m)

    print(f"\n        - {len(fant)} aristas frecuentes ausentes del grafo puntual:\n")
    if len(fant):
        print(fant[["origen", "destino", "tau", "frecuencia", "val_medio", "coherencia_signo", "frac_fdr"]].to_string(index=False))

# FALSOS POSITIVOS (FP) en la capa de topología
det = pd.read_csv(res_dir / "resumen_enlaces_mod_vs_obs.csv")
print(f"\n----FRECUENCIA DE LOS FALSOS POSITIVOS (CAPA DE TOPOLOGÍA)----\n")
for m in modelos:
    print(f'\n   {m}')

    # Filtramos los FP de este modelo en la capa de topología
    fp = det[(det.modelo == m) & (det.capa == "topologia") & (det.tipo == "FP")]
    if fp.empty:
        print()
        continue
    fp = fp.copy()
    f = pd.read_csv(boot_dir / f"frecuencias_W_hist_{m}_B{B}.csv")

    # Estandarización de claves (k1, k2) para asegurar un cruce correcto (merge)
            # entre el detalle de validación y las frecuencias del bootstrap, evitando desajustes alfabéticos.
    f["k1"] = [min(r.origen, r.destino) if r.tau == 0 else r.origen for r in f.itertuples()]
    f["k2"] = [max(r.origen, r.destino) if r.tau == 0 else r.destino for r in f.itertuples()]
    fp["k1"] = [min(r.nodo_a, r.nodo_b) if r.tau_modelo == 0 else r.nodo_a for r in fp.itertuples()]
    fp["k2"] = [max(r.nodo_a, r.nodo_b) if r.tau_modelo == 0 else r.nodo_b for r in fp.itertuples()]

    # Cruzamos los Falsos Positivos con sus respectivas métricas de bootstrap
    j = fp.merge(f, left_on=["k1", "k2", "tau_modelo"], right_on=["k1", "k2", "tau"], how="left")

    if j.frecuencia.isna().any():
        print(f"\n        - AVISO: {int(j.frecuencia.isna().sum())} aristas sin emparejar en {m}")

    print(f"\n            - (FP medio: {j.frecuencia.mean():.3f})")
    print(j[["nodo_a", "nodo_b", "tau_modelo", "frecuencia", "coherencia_signo", "frac_fdr"]].to_string(index=False))