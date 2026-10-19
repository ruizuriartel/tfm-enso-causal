"""
sensibilidad_orden_helmholtz.py
Test de sensibilidad al orden entre la interpolación a 1° y la descomposición de Helmholtz en CHI200 y PSI200. Orden A (el del pipeline): u y v se interpolan a la rejilla común y después se descomponen con windspharm. Orden B: se descomponen en su rejilla nativa y después se interpolan chi y psi. Compara las anomalías (30°S-30°N, octubre-marzo), las EOF observacionales (varianza explicada y congruencia de EOF1 y EOF2) y las series de los nodos, en ERA5 y en el histórico de MIROC6 y HadGEM3-GC31-LL, proyectados sobre la base observacional como en compute_matrices_cmip6.py. Importa compute_solvers_obs.py, por eso está en esta carpeta.
Entradas: data/raw/era5/ y data/raw/cmip6/ (viento en rejilla nativa) y data/processed/era5|cmip6/ (psi y chi de compute_wind_variables.py)
Salidas: data/sensibilidad_helmholtz/ (cálculos intermedios) y resultados/resumen_sensibilidad_orden_helmholtz.csv
Requiere WSL o Linux. Memoria: sección 4.5.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from tqdm import tqdm
from windspharm.xarray import VectorWind

import compute_solvers_obs as cso

warnings.filterwarnings("ignore")

# Definimos parámetros comunes
nivel = "200hPa"
dom = dict(lat=slice(-30, 30))                      # dominio de CHI200 y PSI200
p_base = slice("1981-01-01", "2010-12-31")          # climatología del histórico y las observaciones
ventana = slice("1950", "2014")
meses = [10, 11, 12, 1, 2, 3]
modelos_defecto = ["miroc6", "hadgem3_gc31_ll"]     # una rejilla gaussiana y un modelo de la familia HadAM

# Rejilla común (la misma que en preprocess_*.py)
target_grid = xr.Dataset({
    "lat": (["lat"], np.arange(-89.5, 90.5, 1.0)),
    "lon": (["lon"], np.arange(-179.5, 180.5, 1.0))})

def fechas_estandar(da, inicio="1950"):
    # Normalizamos el eje temporal como en preprocess_*.py (día 15 de cada mes)
    t = pd.date_range(start=f"{inicio}-01-01", periods=da.sizes["time"], freq="MS") + pd.Timedelta(days=14)
    return da.assign_coords(time=t.normalize())

def abrir_viento_era5(raw_dir, comp):       # comp: 'u' o 'v'
    # Cargamos el viento de ERA5 a 200 hPa tal y como se descargó (0,25°, con polos)
    carpeta = raw_dir / "era5" / f"{comp}_component_of_wind_{nivel}"
    ds = xr.open_dataset(next(carpeta.glob("*.nc")), engine="netcdf4")
    nv = list(ds.data_vars)[0]
    ds = ds.rename({nv: comp})
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    # Eliminamos dimensiones y coordenadas sobrantes
    for dim in [d for d in ds.dims if d not in ("time", "latitude", "longitude")]:
        ds = ds.squeeze(dim, drop=True)
    ds = ds.drop_vars([c for c in ds.coords if c not in ("time", "latitude", "longitude")])
    ds = ds.sel(time=slice("1950-01-01", "2014-12-31"))
    return fechas_estandar(ds[comp])

def abrir_viento_cmip6(raw_dir, modelo, carpeta_var):
    # Cargamos el viento histórico del modelo en su rejilla nativa
    d = raw_dir / "cmip6" / carpeta_var / modelo / "historical"
    ds = xr.open_dataset(list(d.glob("*.nc"))[0], engine="netcdf4")
    nv = ds.attrs["variable_id"]

    # Análisis de unidades
    if ds[nv].attrs.get("units") != "m s-1":
        raise ValueError(f"{modelo} {nv}: unidades inesperadas ({ds[nv].attrs.get('units')})")

    # Eliminamos la dimensión de presión y altura
    for dim in ("plev", "height"):
        if dim in ds.dims:
            ds = ds.squeeze(dim, drop=True)
        elif dim in ds.coords:
            ds = ds.drop_vars(dim)

    # Periodo temporal común y normalización de calendario (igual que en preprocess_cmip6.py)
    ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False, align_on="year")
    ds = ds.sel(time=slice("1950-01-01", "2014-12-31"))

    # Comprobamos que la rejilla es rectangular y renombramos coordenadas
    lat_name = [c for c in ds.coords if "lat" in c.lower()][0]
    lon_name = [c for c in ds.coords if "lon" in c.lower()][0]
    if ds[lat_name].ndim != 1:
        raise ValueError(f"{modelo}: rejilla no rectangular, no se puede descomponer con windspharm")
    da = ds[nv].rename({lat_name: "latitude", lon_name: "longitude"})
    da = da.drop_vars([c for c in da.coords if c not in ("time", "latitude", "longitude")])
    return fechas_estandar(da)

def chi_psi(u, v, legfunc="stored"):
    # Igual que calc_chi_psi_lib de compute_wind_variables.py
    # legfunc='computed' solo cambia cómo se guardan los polinomios de Legendre (menos memoria a 0,25°), no el resultado
    u = u.fillna(0.0)
    v = v.fillna(0.0)
    if u.latitude[0] < u.latitude[-1]:
        u = u.sortby("latitude", ascending=False)
        v = v.sortby("latitude", ascending=False)
    w = VectorWind(u, v, legfunc=legfunc)
    psi, chi = w.sfvp()
    psi = psi - psi.mean(dim="longitude")      # media zonal de psi, como en el pipeline
    return chi.astype(np.float32).rename("chi"), psi.astype(np.float32).rename("psi")

def descomponer_e_interpolar(u, v, weights_file, legfunc="stored", bloque=12, etiqueta=""):
    # Orden B: descomponemos en la rejilla nativa y después interpolamos chi y psi
    # Trabajamos por bloques de meses para no cargar todo el viento a 0,25° en memoria
    regridder = None
    chis, psis = [], []
    nt = u.sizes["time"]
    for i in tqdm(range(0, nt, bloque), desc=f"  {etiqueta}"):
        ub = u.isel(time=slice(i, i + bloque)).load()
        vb = v.isel(time=slice(i, i + bloque)).load()
        chi, psi = chi_psi(ub, vb, legfunc=legfunc)

        # Rejilla común (los pesos se calculan una sola vez, con el primer bloque)
        campos = xr.Dataset({"chi": chi, "psi": psi}).sortby("latitude")
        campos = campos.rename({"latitude": "lat", "longitude": "lon"})
        if regridder is None:
            regridder = xe.Regridder(campos, target_grid, "bilinear", periodic=True, ignore_degenerate=True,
                                     reuse_weights=weights_file.exists(), filename=str(weights_file))
        out = regridder(campos).rename({"lat": "latitude", "lon": "longitude"})
        chis.append(out.chi.astype(np.float32))
        psis.append(out.psi.astype(np.float32))
    return xr.concat(chis, dim="time"), xr.concat(psis, dim="time")

def pc_proyectada(anom, solver, npc=1):
    # Misma proyección que PC_sA (compute_nodos_obs.py) y PC_sB (compute_matrices_cmip6.py)
    pc = solver.projectField(anom.transpose("time", ...), neofs=npc, eofscaling=1).isel(mode=npc - 1)
    return (pc - pc.mean()) / pc.std()

def congruencia(a, b):
    # Coeficiente de congruencia (Ecuación 18 de la memoria) sobre los puntos comunes a ambos patrones
    a, b = xr.align(a, b)
    m = np.isfinite(a) & np.isfinite(b)
    a = a.where(m)
    b = b.where(m)
    return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))

def corr_abs(x, y):
    x, y = xr.align(x, y)
    return abs(float(xr.corr(x, y)))

def metricas_campo(anom_a, anom_b):
    # Diferencia relativa (norma de la diferencia entre norma del campo del pipeline) y correlación
    # sobre las anomalías ponderadas del dominio en los meses de octubre a marzo
    a, b = xr.align(anom_a.where(anom_a["time.month"].isin(meses), drop=True),
                    anom_b.where(anom_b["time.month"].isin(meses), drop=True))
    dif = float(np.sqrt(((b - a) ** 2).sum() / (a ** 2).sum()))
    return dif, float(xr.corr(a, b))

if __name__ == "__main__":

    # Uso: python sensibilidad_orden_helmholtz.py [modelo ...] [--solo-era5]
    # Sin argumentos usa MIROC6 y HadGEM3-GC31-LL; admite otros modelos o solo ERA5
    args = sys.argv[1:]
    solo_era5 = "--solo-era5" in args
    modelos = [] if solo_era5 else ([a for a in args if not a.startswith("--")] or modelos_defecto)

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    raw_dir = root_data / "data" / "raw"
    pro_dir = root_data / "data" / "processed"
    work_dir = root_data / "data" / "indices" / "sensibilidad_helmholtz"; work_dir.mkdir(parents=True, exist_ok=True)
    res_dir = root / "resultados"; res_dir.mkdir(parents=True, exist_ok=True)

    filas = []

    # ERA5
    print('\nERA5')
        # Orden B (se guarda para no repetir el cálculo, que es la parte lenta)
    f_chi_b = work_dir / "ERA5_chi_nativo.nc"
    f_psi_b = work_dir / "ERA5_psi_nativo.nc"
    if f_chi_b.exists() and f_psi_b.exists():
        chi_b = xr.open_dataarray(f_chi_b)
        psi_b = xr.open_dataarray(f_psi_b)
    else:
        u = abrir_viento_era5(raw_dir, "u")
        v = abrir_viento_era5(raw_dir, "v")
        chi_b, psi_b = descomponer_e_interpolar(u, v, work_dir / "weights_era5_nativo.nc",
                                                legfunc="computed", etiqueta="ERA5 nativo (0,25°)")
        chi_b.to_netcdf(f_chi_b)
        psi_b.to_netcdf(f_psi_b)

        # Orden A, tal y como salió del pipeline
    chi_a = xr.open_dataset(pro_dir / "era5" / f"velocity_potential_{nivel}" / "ERA5_chi.nc").chi
    psi_a = xr.open_dataset(pro_dir / "era5" / f"streamfunction_{nivel}" / "ERA5_psi.nc").psi

        # Comparación de EOF y nodos de cada orden (los solvers servirán luego de base a los modelos)
    solvers = {}
    for nodo, campo_a, campo_b in [("CHI200", chi_a, chi_b), ("PSI200", psi_a, psi_b)]:
        anom_a = cso.preproceso_pcs(campo_a, **dom)
        anom_b = cso.preproceso_pcs(campo_b, **dom)
        sol_a = cso.calc_solver(anom_a)
        sol_b = cso.calc_solver(anom_b)
        solvers[nodo] = (sol_a, sol_b)

        eof_a = sol_a.eofs(neofs=2)
        eof_b = sol_b.eofs(neofs=2)
        var_a = sol_a.varianceFraction(neigs=2).values * 100
        var_b = sol_b.varianceFraction(neigs=2).values * 100
        pc_a = pc_proyectada(anom_a, sol_a)
        pc_b = pc_proyectada(anom_b, sol_b)
        pc_b_base_a = pc_proyectada(anom_b, sol_a)
        dif, r_campo = metricas_campo(anom_a, anom_b)

        filas.append(dict(conjunto="ERA5", nodo=nodo,
                          dif_rel_campo=dif, corr_campo=r_campo,
                          var_exp_EOF1_A=var_a[0], var_exp_EOF1_B=var_b[0],
                          var_exp_EOF2_A=var_a[1], var_exp_EOF2_B=var_b[1],
                          cong_EOF1=abs(congruencia(eof_a.isel(mode=0), eof_b.isel(mode=0))),
                          cong_EOF2=abs(congruencia(eof_a.isel(mode=1), eof_b.isel(mode=1))),
                          corr_PC_nodo=corr_abs(pc_a, pc_b),
                          corr_PC_base_A=corr_abs(pc_a, pc_b_base_a)))

    # Modelos CMIP6 en el histórico
    for modelo in modelos:
        print(f"  {modelo}")

        # Cargamos el viento en rejilla nativa
        try:
            u = abrir_viento_cmip6(raw_dir, modelo, f"eastward_wind_{nivel}")
            v = abrir_viento_cmip6(raw_dir, modelo, f"northward_wind_{nivel}")
        except (ValueError, IndexError, StopIteration) as e:
            print(f"  Se omite {modelo}: {e}")
            continue

        # Si u y v no comparten rejilla (rejilla escalonada), no se pueden descomponer sin interpolar antes
        if not (np.allclose(u.latitude.values, v.latitude.values) and
                np.allclose(u.longitude.values, v.longitude.values)):
            print(f"  Se omite {modelo}: u y v vienen en rejillas nativas distintas (rejilla escalonada)")
            print(f"    lat u: {u.latitude.values[[0, -1]]} ({u.latitude.size})  lat v: {v.latitude.values[[0, -1]]} ({v.latitude.size})")
            filas.append(dict(conjunto=modelo, nodo="rejilla escalonada"))
            continue

        # Orden B: descomponemos en nativa e interpolamos
        try:
            chi_mb, psi_mb = descomponer_e_interpolar(u, v, work_dir / f"weights_{modelo}_nativo.nc",
                                                      etiqueta=f"{modelo} nativo")
        except ValueError as e:
            print(f"  Se omite {modelo}: windspharm no reconoce la rejilla ({e})")
            continue

        # Orden A, tal y como salió del pipeline
        chi_ma = xr.open_dataset(pro_dir / "cmip6" / f"velocity_potential_{nivel}" / "chi_historical_allmodels.nc").chi.sel(model=modelo)
        psi_ma = xr.open_dataset(pro_dir / "cmip6" / f"streamfunction_{nivel}" / "psi_historical_allmodels.nc").psi.sel(model=modelo)

        # Comparación de nodos proyectados sobre la base observacional
        for nodo, campo_a, campo_b in [("CHI200", chi_ma, chi_mb), ("PSI200", psi_ma, psi_mb)]:
            sol_a, sol_b = solvers[nodo]
            anom_a = cso.preproceso_pcs(campo_a, p_base=p_base, **dom).sel(time=ventana)
            anom_b = cso.preproceso_pcs(campo_b, p_base=p_base, **dom).sel(time=ventana)
            nodo_a = pc_proyectada(anom_a, sol_a)          # nodo actual: orden A en el modelo y en la base
            nodo_b = pc_proyectada(anom_b, sol_b)          # orden B en el modelo y en la base
            nodo_b_base_a = pc_proyectada(anom_b, sol_a)   # orden B en el modelo sobre la base actual
            dif, r_campo = metricas_campo(anom_a, anom_b)
            filas.append(dict(conjunto=modelo, nodo=nodo,
                              dif_rel_campo=dif, corr_campo=r_campo,
                              corr_PC_nodo=corr_abs(nodo_a, nodo_b),
                              corr_PC_base_A=corr_abs(nodo_a, nodo_b_base_a)))

    # Resumen resultados
    tabla = pd.DataFrame(filas)
    tabla.to_csv(res_dir / "resumen_sensibilidad_orden_helmholtz.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.4f}".format):
        print('\n \n---- RESUMEN ----\n')
        print("      A = pipeline, B = descomponer en nativa y después interpolar.\n\n")
        print(tabla.to_string(index=False))
