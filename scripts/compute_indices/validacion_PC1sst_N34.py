"""
validacion_PC1sst_N34.py
Valida que el PC1 de la SST tropical corresponde al índice Niño 3.4, para HadISST y para cada modelo con su propia EOF (estrategia A): correlación con Niño 3.4, alineación del signo y mapas de regresión con los puntos significativos (p < 0.05) marcados.
Entradas: data/processed/hadisst/sea_surface_temperature/HadISST_sst.nc y data/processed/cmip6/sea_surface_temperature/sst_historical_allmodels.nc
Salidas: figures/validacion_PC1sst_N34/sst_eof1_regression_patterns.png y pc1_vs_nino34.png
Memoria: sección 5.1 y Apéndice C.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
import os
import compute_nodos_obs as cno
import compute_solvers_obs as cso

warnings.filterwarnings("ignore")

def pval_func(x, y):
    # Calcula el p-valor de la correlación de Pearson evitando valores nulos o varianza cero
    if np.isnan(x).any() or np.std(x) == 0:
        return np.nan
    _, p = pearsonr(x, y)
    return p

def significancia_pearsonr(anom, pc, alpha=0.05):
    # Aplica la prueba de significancia estadística basada en la correlación de Pearson para cada punto de rejilla
    pvals = xr.apply_ufunc(pval_func, anom, pc, input_core_dims=[['time'], ['time']], vectorize=True)
    return pvals < alpha   # True = significativo

def plot_patrones_sst(eof):
    # Genera una malla de mapas de regresión de los EOF con enmascaramiento de puntos significativos

    n_plots = len(eof)
    cols = 4; rows = int(np.ceil(n_plots / cols))

    fig = plt.figure(figsize=(15, 1.6 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.1, wspace=0.05)

    vmin, vmax = -1.0, 1.0
    levels = np.linspace(vmin, vmax, 21)
    cmap = plt.cm.RdBu_r

    for i, (name, data) in enumerate(eof.items()):
        eof_map, var_exp, sig = data
        lats, lons = eof_map.latitude.values, eof_map.longitude.values

        ax = fig.add_subplot(gs[i], projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_aspect('equal', adjustable='box')
        ax.set_extent([0, 360, -60, 60], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)

        cf = ax.contourf(eof_map.longitude, eof_map.latitude, eof_map,
                         levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())

        lons2d, lats2d = np.meshgrid(lons, lats)
        skip = (slice(None, None, 3), slice(None, None, 3))  #submuestreo de la malla para el punteado (saltamos de 3 en 3 pixels)
        lons_sub = lons2d[skip]
        lats_sub = lats2d[skip]
        sig_sub = sig.values[skip]

        ax.scatter(lons_sub[sig_sub], lats_sub[sig_sub], s=1.0, color='black', alpha=0.6,
                   transform=ccrs.PlateCarree(), marker='.', edgecolor='none')

        ax.set_title(cno.nombres_modelos[name], fontsize=8)
        ax.text(0.98, 0.05, f"{var_exp:.1f}%", transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlabel_style = {'size': 7}; gl.ylabel_style = {'size': 7}
        gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
        gl.top_labels = False; gl.right_labels = False
        if i % cols != 0:
            gl.left_labels = False
        if i < (rows - 1) * cols:
            gl.bottom_labels = False

        # Rectángulo de referencia geográfica para el dominio de análisis
        rect = mpatches.Rectangle((120, -20), 160, 40, linewidth=1, edgecolor='black',
                                  facecolor='none', linestyle='--', transform=ccrs.PlateCarree())
        ax.add_patch(rect)

    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Regresión de la anomalía de SST sobre PC1 (°C σ$^{-1}$)', fontsize=10)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    plt.subplots_adjust(bottom=0.12, hspace=0.3, wspace=0.05)
    fig.savefig(fig_dir / "sst_eof1_regression_patterns.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_pc_vs_nino34(pc, n34):
    # Genera gráficos de comparación temporal entre las Componentes Principales (PC1) y el índice Niño 3.4

    n_plots = len(pc)
    cols = 4; rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.5 * rows), sharex=True)
    axes = axes.flatten()

    for i, name in enumerate(pc.keys()):
        ax1 = axes[i]; ax2 = ax1.twinx()
        pc_data, corr = pc[name]; n34_data = n34[name]

        ax1.plot(pc_data.time, pc_data, color='tab:blue', linewidth=1.2, alpha=0.8, label='PC1 (eje izquierdo)')
        ax2.plot(n34_data.time, n34_data, color='tab:red', linewidth=1.2, alpha=0.8, label='Niño 3.4 (eje derecho)')

        ax1.set_ylim(-5, 5); ax2.set_ylim(-5, 5)
        ax1.set_yticks([-4, -2, 0, 2, 4]); ax2.set_yticks([-4, -2, 0, 2, 4])
        ax1.tick_params(axis='both', labelsize=7); ax2.tick_params(axis='both', labelsize=7)
        ax1.tick_params(axis='y', colors='tab:blue'); ax2.tick_params(axis='y', colors='tab:red')
        ax1.set_title(f"{cno.nombres_modelos[name]} | r = {corr:+.3f}", fontsize=8)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.75)
        if i % cols == 0:
            ax1.set_ylabel('PC1 (std)', color='tab:blue', fontsize=10)
        else:
            ax1.tick_params(axis='y', labelleft=False)
        if i % cols == cols - 1 or i == n_plots - 1:
            ax2.set_ylabel('Niño 3.4 (°C)', color='tab:red', fontsize=10)
        else:
            ax2.tick_params(axis='y', labelright=False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    handles = [
        plt.Line2D([0], [0], color='tab:blue', lw=2, label='PC1 (eje izquierdo)'),
        plt.Line2D([0], [0], color='tab:red', lw=2, label='Niño 3.4 (eje derecho)')]

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(fig_dir / "pc1_vs_nino34.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == '__main__':

    # Cargamos datos
    root = Path(__file__).resolve().parent.parent.parent
    root_data = Path(os.getenv("DATA_PATH", root))
    if not root_data.exists(): root_data = root

    campos_h = xr.open_dataset(root_data / "data" / "processed" / "hadisst" / "sea_surface_temperature" / "HadISST_sst.nc", engine="netcdf4")
    campos_c = xr.open_dataset(root_data / "data" / "processed" / "cmip6" / "sea_surface_temperature" /  "sst_historical_allmodels.nc", engine="netcdf4")

    modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]

    # Definimos directorio de salida
    fig_dir = root / "figures" / "validacion_PC1sst_N34"
    fig_dir.mkdir(parents=True, exist_ok=True)
    PC1 = {}; EOF1 = {}; N34 = {}

        # Datos OBS
    m = "hadisst"
    print(f"\n{m}")
    n34_h = cno.indice_n34(campos_h.sst)
    N34[m] = n34_h

    ssta_pt_h = cso.preproceso_pcs(campos_h.sst, lat=slice(-20, 20), lon=slice(120, 280), moc = True)
    ssta_g_h = cso.preproceso_pcs(campos_h.sst, moc = True)
    solver = cso.calc_solver(ssta_pt_h)

    eof1_obs, var_obs, pc1_obs = cno.PC_sA(ssta_pt_h, ssta_g_h, solver, npc=1)

    # Sincronización temporal y alineación del signo con el índice Niño 3.4
    common_time_h = np.intersect1d(pc1_obs.time.values, n34_h.time.values)
    corr_val_h = float(xr.corr(pc1_obs.sel(time=common_time_h), n34_h.sel(time=common_time_h)))
    sign = np.sign(corr_val_h)
    pc1_obs, eof1_obs, corr_val_h = sign*pc1_obs, sign*eof1_obs, sign*corr_val_h

    sig_h = significancia_pearsonr(ssta_g_h, pc1_obs)
    PC1[m] = (pc1_obs, corr_val_h)
    EOF1[m] = [eof1_obs, var_obs, sig_h]

        # Modelos
    for m in modelos:
        print(f"{m}")
        campos_m = campos_c.sel(model=m)
        n34_m = cno.indice_n34(campos_m.sst)
        N34[m] = n34_m

        ssta_pt_m = cso.preproceso_pcs(campos_m.sst, lat=slice(-20, 20), lon=slice(120, 280), moc=True)
        ssta_g_m = cso.preproceso_pcs(campos_m.sst, moc=True)
        solver = cso.calc_solver(ssta_pt_m)

        eof1_m, var_m, pc1_m = cno.PC_sA(ssta_pt_m, ssta_g_m, solver, npc=1)

        # Sincronización temporal y corrección de signo orientada a la correlación con Niño 3.4
        common_time_m = np.intersect1d(pc1_m.time.values, n34_m.time.values)
        corr_val_m = float(xr.corr(pc1_m.sel(time=common_time_m), n34_m.sel(time=common_time_m)))
        sign = np.sign(corr_val_m)
        pc1_m, eof1_m, corr_val_m = sign * pc1_m, sign * eof1_m, sign*corr_val_m

        sig_m = significancia_pearsonr(ssta_g_m, pc1_m)
        PC1[m] = (pc1_m, corr_val_m)
        EOF1[m] = [eof1_m, var_m, sig_m]

    # Generación y almacenamiento de figuras de validación
    plot_patrones_sst(EOF1)
    plot_pc_vs_nino34(PC1, N34)

