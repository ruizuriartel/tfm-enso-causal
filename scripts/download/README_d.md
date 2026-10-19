# scripts/download/

Descarga de los datos crudos: observaciones, reanálisis y ensemble CMIP6.

**Orden de ejecución:** los cinco scripts de descarga principal pueden lanzarse en cualquier orden entre sí. `download_pangeo.py` se ejecuta después de `download_cmip6.py`, ya que depende de los atributos (`variant_label`) de los ficheros ya descargados. `comparacion_miembro_ensamble.py` se ejecuta al final, como chequeo.

| Script | Descripción |
|---|---|
| `download_cmip6.py` | Descarga las variables mensuales de CMIP6 (SST, OLR, PSL, precipitación, temperatura a 2m, viento en 850/200 hPa, geopotencial en 200 hPa) para los 7 modelos y los 3 experimentos (histórico, ssp245, ssp585) vía la API de CDS (`projections-cmip6`). Requiere API key de CDS configurada en `~/.cdsapirc`. |
| `download_pangeo.py` | Completa los huecos de variables no disponibles en CDS para pares modelo-experimento concretos: geopotencial en ssp245 y viento (ambos niveles) en ssp585 para EC-Earth3-CC; viento (ambos niveles) en ssp245 para HadGEM3-GC31-LL. Descarga del catálogo público de Pangeo/Google Cloud Storage con acceso anónimo (`token="anon"`, sin credenciales). **Requiere WSL/Linux** (usa `asyncio` con política de event loop específica). |
| `download_era5.py` | Descarga ERA5 (viento en niveles de presión, geopotencial, presión a nivel del mar, radiación de onda larga saliente) vía la API de CDS. Requiere API key de CDS. |
| `download_hadisst.py` | Descarga y descomprime la SST de HadISST desde el servidor del Met Office. Acceso directo, sin credenciales. |
| `download_gpcc.py` | Descarga y concatena la precipitación mensual de GPCC (1946-2015) desde el repositorio abierto de DWD. Acceso directo, sin credenciales. |
| `download_cru.py` | Descarga precipitación y temperatura de CRU TS4.09 desde el servidor de la UEA. Acceso directo, sin credenciales. |
| `comparacion_miembro_ensamble.py` | Chequeo de control de calidad: recorre todos los ficheros CMIP6 descargados y verifica que, para cada par modelo-experimento, todas las variables comparten el mismo `variant_label`/`member_id`. |

## Credenciales

- **CDS (Copernicus Climate Data Store):** necesaria para `download_cmip6.py` y `download_era5.py`. Configurar en `~/.cdsapirc` (no versionado).
- **Pangeo/GCS, HadISST, GPCC, CRU:** acceso directo, sin registro ni credenciales.
