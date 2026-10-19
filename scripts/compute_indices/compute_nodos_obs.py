"""
compute_nodos_obs.py
Define los nodos (variable, base de datos y dominio espacial), calcula el índice Niño 3.4 y, para cada variable observacional, las series de PC1 y PC2 (proyección sobre el solver, estandarizadas) y sus patrones de regresión sobre el campo global. Contiene también las funciones comunes de figuras y los nombres oficiales de los modelos (nombres_modelos), que usan otros scripts de la carpeta. Pregunta por pantalla antes de mostrar las figuras.
Entradas: data/processed/ y data/indices/solvers_obs/ (de compute_solvers_obs.py)
Salidas: data/indices/nodos_obs/series_pc1_pc2.pkl y patrones_pc1_pc2.pkl; figures/verificacion_modos/comparacion_patrones_pc1_pc2.png y comparacion_series_pc1_pc2.png
Memoria: secciones 5.1, 5.2 y Apéndice D.
"""

import xarray as xr
import numpy as np
import warnings
import os
from pathlib import Path
import pickle
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import compute_solvers_obs as cso

warnings.filterwarnings("ignore")

# Nombres oficiales de los modelos (para títulos y etiquetas de las figuras)
nombres_modelos = {"hadisst": "HadISST",
                   "ec_earth3_cc": "EC-Earth3-CC", "access_cm2": "ACCESS-CM2", "cnrm_esm2_1": "CNRM-ESM2-1",
                   "hadgem3_gc31_ll": "HadGEM3-GC31-LL", "miroc6": "MIROC6", "noresm2_mm": "NorESM2-MM",
                   "ukesm1_0_ll": "UKESM1-0-LL"}

def get_nodos(mask_land):
    # Diccionario con los nodos observacionales, bases de datos, directorios y dominios espaciales
    return {
        'N34': dict(db='HadISST', folder='sea_surface_temperature'),
        'SST': dict(db='HadISST', folder='sea_surface_temperature', u="ºC", dom=dict(lat=slice(-20, 20), lon=slice((120 + 360) % 360, (-80 + 360) % 360), moc=True)),
        'CHI200': dict(db='ERA5', folder='velocity_potential_200hPa', u="m$^2$ s$^{-1}$", dom=dict(lat=slice(-30, 30))),
        'CHI850': dict(db='ERA5', folder='velocity_potential_850hPa', u="m$^2$ s$^{-1}$", dom=dict(lat=slice(-30, 30))),
        'PSI200': dict(db='ERA5', folder='streamfunction_200hPa', u="m$^2$ s$^{-1}$", dom=dict(lat=slice(-30, 30))),
        'PSI850': dict(db='ERA5', folder='streamfunction_850hPa', u="m$^2$ s$^{-1}$", dom=dict(lat=slice(-30, 30))),
        'OLR': dict(db='ERA5', folder='toa_outgoing_longwave_radiation', u="W m$^{-2}$", dom=dict(lat=slice(-30, 30))),
        'PSL': dict(db='ERA5', folder='sea_level_pressure', u="Pa", dom=dict(lat=slice(-20, 20), lon=slice((80 + 360) % 360, (-80 + 360) % 360))),
        'Z200_PNA': dict(db='ERA5', folder='geopotential_height_200hPa', u="m", dom=dict(lat=slice(20, 70), lon=slice((120 + 360) % 360, (-60 + 360) % 360))),
        'Z200_PSA': dict(db='ERA5', folder='geopotential_height_200hPa', u="m", dom=dict(lat=slice(-70, -20), lon=slice((120 + 360) % 360, (-60 + 360) % 360))),
        'PR_land': dict(db='GPCC', folder='precipitation', u="kg m$^{-2}$ s$^{-1}$", dom=dict(lat=slice(-30, 30), mla=mask_land)),
        'TAS_land': dict(db='CRU', folder='near_surface_air_temperature', u="°C", dom=dict(lat=slice(20, 70), mla=mask_land))}

def indice_n34(var, rolling=False, p_base=slice("1981-01-01", "2010-12-31")):

    # Recorte del dominio espacial para el índice Niño 3.4 (5°S-5°N, 190°E-240°E)
    var = var.assign_coords(longitude=((var.longitude + 360) % 360)).sortby("longitude")
    var = var.sel(latitude=slice(-5, 5), longitude=slice(190, 240))

    # Cálculo de anomalías mensuales respecto al periodo base de referencia
    clim = var.sel(time=p_base).groupby("time.month").mean("time")
    anom = var.groupby("time.month") - clim

    # Suavizado temporal opcional mediante media móvil trimestral centrada
    if rolling:
            anom = anom.rolling(time=3, center=True).mean().dropna(dim="time", how="all")

    # Aplicación de máscara oceánica estricta
    anom = anom.where(anom.notnull().all(dim="time"))

    # Ponderación espacial por el coseno de la latitud
    weights = np.cos(np.deg2rad(anom.latitude)).clip(0)
    n34 = anom.weighted(weights).mean(dim=["latitude", "longitude"])
    return n34

def PC_sA(anom_dom, anom_glob, solver, npc=1):

    # Componentes principales
    mode_idx = npc - 1
    pc = solver.pcs(npcs=npc, pcscaling=1).isel(mode=mode_idx)
    pc_n = (pc - pc.mean()) / pc.std()

    # Fracción de varianza explicada
    var_exp = solver.varianceFraction(neigs=npc).values[mode_idx] * 100

    # Filtrado estacional para retener únicamente los meses de octubre a marzo
    meses_validos = [10, 11, 12, 1, 2, 3]
    anom_ge = anom_glob.where(anom_glob["time.month"].isin(meses_validos), drop=True)
    common_time = np.intersect1d(anom_ge.time.values, pc_n.time.values)
    anom_ge = anom_ge.sel(time=common_time); pc_n = pc_n.sel(time=common_time)

    # Patrones de regresión global
    eof = xr.cov(anom_ge, pc_n, dim='time')

    # Proyección del campo de anomalías locales sobre el solver para obtener la PC proyectada y normalizada
    pc_reg = solver.projectField(anom_dom, neofs=npc, eofscaling=1).isel(mode=mode_idx)
    pc_reg_n = (pc_reg - pc_reg.mean()) / pc_reg.std()
    return eof, var_exp, pc_reg_n

def plot_pattern_grid(datos, fig_path, columnas=4):
    # Genera una cuadrícula de mapas espaciales para visualizar los patrones de regresión de los EOF observados

    n_plots = len(datos)
    cols = columnas; rows = int(np.ceil(n_plots / cols))

    bb = [d[3:5] for d in datos.values() if d[0] is not None]
    lat_i = max(-90, min(b[0] for b in bb) - 5)
    lat_f = min(90, max(b[1] for b in bb) + 5)

    fig = plt.figure(figsize=(3.5 * cols, 2.8 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.1, wspace=0.05)
    cmap = plt.cm.RdBu_r

    for i, (name, data) in enumerate(datos.items()):
        if data[0] is None:
            continue
        eof_map, var_exp, unit, lat_min, lat_max, lon_min, lon_max = data

        mask_lon = (eof_map.longitude >= lon_min) & (eof_map.longitude <= lon_max)
        mask_lat = (eof_map.latitude >= lat_min) & (eof_map.latitude <= lat_max)
        dominio = eof_map.where(mask_lon & mask_lat)

        vmin = float(dominio.min()); vmax = float(dominio.max())
        limite = max(abs(vmin), abs(vmax))
        vmin, vmax = -limite, limite
        levels = np.linspace(vmin, vmax, 21)

        ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_aspect('equal', adjustable='box')
        ax.set_extent([0, 360, lat_i, lat_f], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)

        cf = ax.contourf(eof_map.longitude, eof_map.latitude, eof_map,
                         levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())

        ax.set_title(name, fontsize=8)
        if var_exp is not None:
            ax.text(0.98, 0.05, f"{var_exp:.1f}%", transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlabel_style = {'size': 7}; gl.ylabel_style = {'size': 7}
        gl.ylocator = ticker.FixedLocator(np.arange(-60, lat_f + 1, 30))
        gl.top_labels = False
        gl.right_labels = False
        if i % cols != 0:
            gl.left_labels = False
        if i < (rows - 1) * cols:
            gl.bottom_labels = False

        width = lon_max - lon_min
        height = lat_max - lat_min
        rect = patches.Rectangle((lon_min, lat_min), width, height, linewidth=1, edgecolor='black',
                                  facecolor='none', linestyle='--', transform=ccrs.PlateCarree())
        ax.add_patch(rect)

        cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.12, shrink=0.85)
        cbar.set_label(f'Regresión ({unit} σ$^{{-1}}$)', fontsize=7)
        cbar.set_ticks(np.linspace(vmin, vmax, 5))
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.xaxis.get_offset_text().set_fontsize(6)

    plt.subplots_adjust(bottom=0.12, top=0.88 if rows != 1 else 0.8, hspace=0.3, wspace=0.05)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    aux = input(f'Mostrar {fig_path.name} [y/n]  ')
    if aux == 'y':
        plt.show()
    plt.close(fig)

def plot_series_grid(series, fig_path, columnas=4):
    # Genera una cuadrícula de series temporales

    nombres = list(series.keys()); n_plots = len(nombres)
    cols = columnas; rows = int(np.ceil(n_plots / cols))

    fig = plt.figure(figsize=(5 * cols, 3.2 * rows))

    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.35, wspace=0.15)

    for i, n in enumerate(nombres):
        s = series[n]; t = s.time.values; v = s.values

        ax = fig.add_subplot(gs[i])
        ax.plot(t, v, color='tab:blue', linewidth=0.6)
        ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
        ax.set_title(n, fontsize=13); ax.tick_params(labelsize=11)
        ax.set_xlim(t[0], t[-1]); ax.set_ylabel("PC (std)", fontsize=13)
        if i % cols != 0:
            ax.set_ylabel("")

    plt.subplots_adjust(bottom=0.12, top=0.88 if rows != 1 else 0.8, hspace=0.35, wspace=0.15)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight"); plt.close(fig)

if __name__ == '__main__':

    # Definimos directorios de entrada y salida
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    pro_dir = root_data / "data" / "processed"
    solv_dir = root_data / "data" / "indices" / "solvers_obs"
    out_dir = root_data / "data" / "indices" / "nodos_obs"; out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = root / "figures" / "verificacion_modos"; fig_dir.mkdir(parents=True, exist_ok=True)

    # Cargamos solvers ya calculados
    solvers = {}
    for pkl in sorted(solv_dir.glob("solver_*.pkl")):
        nombre = pkl.stem.replace("solver_", "")
        with open(pkl, "rb") as f:
            solvers[nombre] = pickle.load(f)

    # Cargamos máscara de tierra y nodos
    mask_land = xr.open_dataarray(solv_dir / "mask_land.nc")
    nodos = get_nodos(mask_land)

    series = {}; patrones = {}; print('\n')

    # Iteración sobre cada nodo para calcular series temporales y patrones espaciales
    for nodo, datos in nodos.items():
        print(f"Procesando nodo: {nodo}")
        dir_var = pro_dir / datos['db'].lower() / datos['folder']
        path_var = list(dir_var.glob(f"{datos['db']}*.nc"))[0]
        nv = path_var.stem.split('_')[1]
        campo = xr.open_dataset(path_var, engine="netcdf4")[nv]

        if nodo == 'N34':
            n34_obs = indice_n34(campo)
            series[nodo] = n34_obs
            continue

        dominio = datos['dom']
        mascara = {k: v for k, v in datos['dom'].items() if k in ('moc', 'mla')}
        anom_dom = cso.preproceso_pcs(campo, **dominio)
        anom_glob = cso.preproceso_pcs(campo, **mascara)

        lat_min, lat_max = (dominio['lat'].start, dominio['lat'].stop) if 'lat' in dominio else (-90, 90)
        lon_min, lon_max = (dominio['lon'].start, dominio['lon'].stop) if 'lon' in dominio else (0, 360)
        solv = solvers[nodo]

        for npc in [1, 2]:
            eof, var, pc = PC_sA(anom_dom, anom_glob, solv, npc=npc)
            key = f"{nodo}_PC{npc}"
            patrones[key] = (eof, var, datos['u'], lat_min, lat_max, lon_min, lon_max)
            series[key] = pc

    # Graficamos patrones y series
    plot_pattern_grid(patrones, fig_dir / "comparacion_patrones_pc1_pc2.png", columnas=4)
    plot_series_grid(series, fig_dir / "comparacion_series_pc1_pc2.png", columnas=4)

    # Guardamos patrones y series
    with open(out_dir / "series_pc1_pc2.pkl", "wb") as f:
        pickle.dump(series, f)
    with open(out_dir / "patrones_pc1_pc2.pkl", "wb") as f:
        pickle.dump(patrones, f)
