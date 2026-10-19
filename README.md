# Estructura causal de la respuesta atmosférica a ENSO bajo calentamiento global en observaciones y modelos CMIP6

Código del Trabajo de Fin de Máster del Máster en Matemáticas y Computación, Universidad de Cantabria.

**Autora:** Laura Ruiz Uriarte  
**Director:** Roberto Suárez-Moreno

---

## Resumen

Este repositorio contiene el código con el que se obtienen todos los resultados, tablas y figuras de la memoria del TFM. El trabajo estudia si la estructura causal de la respuesta atmosférica a ENSO cambia bajo calentamiento global. Para ello se aplica PCMCI+ (Runge, 2020), con la librería [tigramite](https://github.com/jakobrunge/tigramite) (versión 5.2.10.1) y el test de independencia condicional ParCorr, a un conjunto de ocho nodos climáticos (Niño 3.4, potencial de velocidad y función corriente en 200 hPa, OLR, dos modos de geopotencial en 200 hPa, y un modo de precipitación y otro de temperatura sobre tierra) construidos a partir de observaciones y reanálisis, y de siete modelos CMIP6 (ACCESS-CM2, EC-Earth3-CC, CNRM-ESM2-1, HadGEM3-GC31-LL, MIROC6, NorESM2-MM y UKESM1-0-LL). Se compara el periodo histórico con los escenarios SSP2-4.5 y SSP5-8.5.

El código cubre todo el proceso: descarga y preprocesado de los datos, selección de modelos por su Niño 3.4, construcción de los nodos y reducción por correlación y VIF, inferencia causal con PCMCI+ y bootstrap por bloques, validación histórica frente a observaciones (F1 en tres capas), y contraste del cambio estructural entre el histórico y los escenarios.

## Estructura del repositorio

```
tfm-enso-causal/
|-- data/                    # datos y resultados pesados, no se suben a GitHub (ver Datos)
|-- figures/                 # figuras generadas por los scripts, por bloque temático
|-- resultados/              # tablas .csv con los resultados numéricos de cada análisis
|-- scripts/
|   |-- download/            # descarga de observaciones, reanálisis y CMIP6
|   |-- preprocess/          # homogeneización de calendario, unidades y rejilla común de 1°
|   |-- seleccion_modelo/    # índice Niño 3.4 y métricas de Planton et al. (2021)
|   |-- compute_indices/     # Helmholtz, EOF, nodos y matrices de series por ventana
|   `-- compute_grafos/      # PCMCI+, bootstrap, comparaciones y diagnósticos
|-- .gitignore               # archivos que no se suben: datos, PDFs, cachés y archivos del editor
|-- CITATION.cff             # metadatos para citar el código
|-- environment_windows.yml  # entorno conda de Windows
|-- environment_linux.yml    # entorno conda mínimo de Linux o WSL2
|-- LICENSE                  # GNU GPL v3 o posterior
`-- README.md
```

Cada carpeta de `scripts/` tiene su propio README con la descripción de cada script y su orden de ejecución. La cabecera de cada script indica sus entradas, sus salidas y la sección, figura o apéndice de la memoria a los que corresponde.

La carpeta `data/` no se incluye en el repositorio por su tamaño (el archivo `.gitignore` la excluye). Contiene los datos descargados (`data/raw/`), los preprocesados (`data/processed/`) y los resultados intermedios pesados (`data/indices/`: EOF, nodos, matrices de series por ventana, muestras de bootstrap y grafos de PCMCI+). Tampoco se incluyen los PDFs de los artículos citados, por copyright.

## Datos

Los datos se regeneran ejecutando los scripts de `scripts/download/` y después los de `scripts/preprocess/`. Fuentes utilizadas:

| Fuente | Variable(s) | Acceso |
|---|---|---|
| ERA5 (Copernicus CDS) | viento y geopotencial en 200 y 850 hPa, PSL, OLR | API de CDS (`~/.cdsapirc`) |
| CMIP6 (Copernicus CDS, `projections-cmip6`) | SST, OLR, PSL, precipitación, temperatura a 2 m, viento y geopotencial | API de CDS (`~/.cdsapirc`) |
| CMIP6 (catálogo Pangeo) | cinco variables que CDS no ofrece (ver `scripts/download/README_d.md`) | anónimo, sin credenciales |
| HadISST 1.1 (Met Office) | temperatura superficial del mar | descarga directa |
| GPCC v2025 (DWD) | precipitación | descarga directa |
| CRU TS4.09 (UEA) | temperatura y precipitación | descarga directa |

Para usar la API de CDS hay que tener una cuenta, configurar `~/.cdsapirc` y aceptar los términos de uso de cada conjunto de datos en la web del CDS. Las versiones de los conjuntos de Pangeo están fijadas en `download_pangeo.py` para que la descarga sea reproducible.

Todos los scripts leen la ruta base de los datos de la variable de entorno `DATA_PATH`. Si no está definida o no existe, se usa la raíz del repositorio. Se define así:

En Linux o WSL2:

```bash
export DATA_PATH=/ruta/a/tus/datos                        # solo en la terminal abierta
echo 'export DATA_PATH=/ruta/a/tus/datos' >> ~/.bashrc    # fija, para las terminales nuevas
```

En Windows (símbolo del sistema):

```bash
set DATA_PATH=C:\ruta\a\tus\datos                        # solo en la terminal abierta
setx DATA_PATH "C:\ruta\a\tus\datos"                     # fija, para las terminales nuevas
```

Las figuras y los resultados se guardan siempre dentro del repositorio (`figures/` y `resultados/`).

## Entorno

```bash
git clone https://github.com/ruizuriartel/tfm-enso-causal
cd tfm-enso-causal
```

Después se crea el entorno que corresponda al sistema (en los dos casos se llama `tfm-enso-causal`):

```bash
conda env create -f environment_windows.yml    # Windows
conda env create -f environment_linux.yml      # Linux o WSL2
conda activate tfm-enso-causal
```

Hay dos archivos de entorno, con las versiones exactas de cada uno:

- `environment_windows.yml` es el entorno de Windows (Python 3.13.13) con el que se generaron los resultados de la memoria, salvo tres scripts que requieren Linux. Es una exportación con versiones exactas y solo se puede crear en Windows.
- `environment_linux.yml` es un entorno mínimo para Linux o WSL2 (Python 3.14.6), con las versiones con las que se ejecutaron los tres scripts que requieren Linux. No incluye tigramite ni el resto de librerías del pipeline, así que no permite ejecutar el resto de scripts.

Ambos entornos se comprobaron el 17/10/2026: se crearon desde cero con `conda env create` sin conflictos y todas sus librerías principales se importan correctamente.

**Scripts que requieren WSL2 o Linux.** Tres scripts necesitan `windspharm`, que no es compatible con Windows nativo y que no aparece en `environment_windows.yml`: `compute_wind_variables.py` y `sensibilidad_orden_helmholtz.py` (en `compute_indices/`), y `download_pangeo.py` (en `download/`). Se ejecutaron con `windspharm` 2.0.0 y `pyspharm` 1.0.9 (conda-forge).

## Pipeline de ejecución

Los scripts se ejecutan en el orden indicado, porque cada bloque depende de las salidas del anterior. El detalle de cada script está en el README de su carpeta.

Los scripts de `preprocess/` y `seleccion_modelo/`, y algunos de `compute_indices/`, hacen preguntas por pantalla (`[y/n]`) para mostrar información de comprobación o figuras. Responder `n` en cada una permite ejecutarlos sin supervisión. `seleccion_modelo_nino34.py` hay que lanzarlo dos veces, cambiando la variable `datos` (`'hadisst'` y `'cmip6'`).

### 1. Descarga (`scripts/download/`)

`download_hadisst.py`, `download_era5.py`, `download_gpcc.py`, `download_cru.py` y `download_cmip6.py` (en cualquier orden entre sí), después `download_pangeo.py` y, al final, `comparacion_miembro_ensamble.py`, que comprueba que todas las variables de cada par modelo-experimento comparten el mismo miembro del ensamble.

### 2. Preprocesado (`scripts/preprocess/`)

`preprocess_hadisst.py`, `preprocess_era5.py`, `preprocess_gpcc.py`, `preprocess_cru.py` y `preprocess_cmip6.py`. Son independientes entre sí.

### 3. Selección de modelo (`scripts/seleccion_modelo/`)

`seleccion_modelo_nino34.py` (dos veces) y después `plot_nino34_comparison.py`, que calcula las cuatro métricas de Planton et al. (2021) y escribe `resultados/resumen_metricas_planton.csv`.

### 4. Nodos observacionales (`scripts/compute_indices/`, bloque 1)

`compute_wind_variables.py`, `sensibilidad_orden_helmholtz.py`, `compute_solvers_obs.py`, `validacion_PC1sst_N34.py`, `compute_nodos_obs.py`, `north_test_obs.py`, `compute_matriz_obs.py` y `reduccion_nodos.py`, que deja los 8 nodos finales.

### 5. Extensión a CMIP6 (`scripts/compute_indices/`, bloque 2)

`compute_matrices_cmip6.py` proyecta cada modelo sobre la base de EOF observacional y construye las matrices de cada ventana. `diagnostico_climatologia.py` y `diagnostico_nodos_cmip6.py` son los diagnósticos asociados.

### 6. Inferencia causal y figuras (`scripts/compute_grafos/`)

`pcmci_plus.py` y `bootstrap_grafos.py` (el bootstrap por bloques, con B=500 sobre 22 ventanas, es la parte más costosa del pipeline), seguidos de `plot_grafos_causales.py` y `plot_frecuencias_bootstrap.py`.

### 7. Validación histórica frente a observaciones (`scripts/compute_grafos/`)

`comparacion_obs_modelos_hist.py` (F1 en tres capas, distribución bootstrap y pesos del ensamble) y `diagnostico_aristas_ausentes_FP.py`.

### 8. Cambio estructural bajo calentamiento (`scripts/compute_grafos/`)

`comparacion_escenarios.py`, `plot_cambios_ensemble.py` y `diagnostico_varianza_autocorr.py`.

## Salidas

- `figures/`: las figuras de la memoria, organizadas por bloque temático.
- `resultados/`: las tablas `.csv` con los resultados numéricos (métricas de selección de modelo, diagnósticos de nodos, enlaces de PCMCI+, frecuencias de bootstrap, comparaciones con observaciones, cambio estructural, etc.).
- `data/` (no versionada): los datos descargados y preprocesados y los resultados intermedios pesados. Los scripts la crean al ejecutarse, y su ubicación se puede cambiar con `DATA_PATH`.

## Referencias principales

Los artículos no se incluyen por copyright. La cita completa de cada uno está en la bibliografía de la memoria.

- Runge, J. et al. (2019): PCMCI.
- Runge, J. (2020): PCMCI+ y reglas de orientación.
- Runge, J. et al. (2023): revisión de inferencia causal para series temporales, recomendada por tigramite como punto de partida.
- Nowack, P. et al. (2020): ponderación del ensamble por similitud causal.
- Karmouche, S. et al. (2023): métrica F1 con desfase y signo.
- Kuma, R. et al. (2023): familias de modelos CMIP6.
- Planton, Y. et al. (2021): métricas de ENSO en CMIP6.
- Bretherton, C. S. et al. (1999): grados de libertad efectivos.
- Langford, S. et al. (2001): coeficiente de congruencia.

Datos y software que conviene citar al usar este código:

- Hersbach, H. et al. (2020): reanálisis ERA5. Quarterly Journal of the Royal Meteorological Society.
- Eyring, V. et al. (2016): diseño experimental de CMIP6. Geoscientific Model Development.
- Abernathey, R. P. et al. (2021): catálogo CMIP6 de Pangeo. Computing in Science & Engineering.
- Rayner, N. A. et al. (2003): HadISST. Journal of Geophysical Research.
- Harris, I. et al. (2020): CRU TS. Scientific Data.
- Rustemeier, E. et al. (2025): GPCC Precipitation Analysis Monthly v2025 a 1,0° (Deutscher Wetterdienst).
- Dawson, A. (2016): `windspharm`. Journal of Open Research Software.
- Dawson, A. (2016): `eofs`. Journal of Open Research Software.
- Zhuang, J. et al.: `xESMF` v0.9.2 (Zenodo).

Los datos no se redistribuyen en este repositorio: cada fuente tiene sus propias condiciones de uso y se descarga desde su origen. Los datos de ERA5 y de CMIP6 obtenidos del CDS requieren citar el servicio de Copernicus y a cada grupo de modelización.

## Cómo citar

Si usas este código, cita la versión archivada en Zenodo. Si además usas PCMCI+, cita también a los autores de tigramite (Runge, 2020; Runge et al., 2019), como pide su licencia de uso.

Ruiz Uriarte, L. (2026). *Código del TFM: Estructura causal de la respuesta atmosférica a ENSO bajo calentamiento global en observaciones y modelos CMIP6*. Zenodo. [DOI pendiente]

## Licencia

Copyright (C) 2026 Laura Ruiz Uriarte.

Este código es software libre: puedes redistribuirlo y modificarlo bajo los términos de la licencia GNU General Public License publicada por la Free Software Foundation, en su versión 3 o, a tu elección, cualquier versión posterior. Es la misma licencia que utiliza la librería tigramite en la que se apoya. Se distribuye con la esperanza de que sea útil, pero sin ninguna garantía, ni siquiera la garantía implícita de comercialización o de idoneidad para un fin concreto. El texto completo está en el archivo [LICENSE](LICENSE).
