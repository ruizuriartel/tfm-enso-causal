"""
download_cmip6.py
Descarga CMIP6 mensual desde el CDS (projections-cmip6): SST, OLR, PSL, precipitación, TAS, viento (850/200 hPa) y geopotencial (200 hPa), para los 7 modelos y los experimentos historical, ssp2_4_5 y ssp5_8_5.
Entradas: API de CDS (~/.cdsapirc).
Salidas: data/raw/cmip6/<variable>[_<nivel>hPa]/<modelo>/<experimento>/*.nc
Memoria: sección 4.3.
"""

import cdsapi
from pathlib import Path
import zipfile
import os

modelos = ["ec_earth3_cc", "access_cm2", "cnrm_esm2_1", "hadgem3_gc31_ll", "miroc6", "noresm2_mm", "ukesm1_0_ll"]
variables = ["sea_surface_temperature", "toa_outgoing_longwave_radiation", "sea_level_pressure", "precipitation", "near_surface_air_temperature"]
v_wind = ["eastward_wind", "northward_wind"]
experimentos = ["historical",  "ssp2_4_5", "ssp5_8_5"]
descargas_fallidas = []

# Definimos directorio de salida
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root


def descargar_cmip6(v, m, e, level=None):

    # Definimos dataset a descargar
    dataset = "projections-cmip6"
    request = {
        "temporal_resolution": "monthly",
        "experiment": e,
        "variable": v,
        "model": m,
        "year": [str(year) for year in range(1950, 2015)] if e == "historical" else [str(year) for year in range(2015, 2101)]
    }

    # Añadimos nivel de presión cuando lo necesitemos
    if level:
        request["level"] = level
        suffix = f"_{level}hPa"
    else:
        suffix = ""

    data_dir = root_data / "data" / "raw" / "cmip6" / (f"{request['variable']}"+suffix) / f"{request['model']}" / f"{request['experiment']}"
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = data_dir / f"{request['model']}_{request['experiment']}_{request['variable']}.zip"

    # Descarga
    try:
        print(f"Descargando {e} - {m} - {v}")
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(filename)

        # Descomprimir
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Borramos archivos innecesarios
        os.remove(filename)
        for extra in ["provenance.json", "provenance.png"]:
            extra_path = data_dir / extra
            if os.path.exists(extra_path):
                os.remove(extra_path)

    except Exception as err:
        print(f"Error al procesar {m} - {v}: {err}")
        descargas_fallidas.append(f"{e},{m},{v},{level}")

if __name__ == '__main__':

    for e in experimentos:
        for v in variables:
            for m in modelos:
                descargar_cmip6(v, m, e)

        for v in v_wind:
            for m in modelos:
                for l in ["850","200"]:
                    descargar_cmip6(v, m, e, level=l)

        v = "geopotential_height"
        l = "200"
        for m in modelos:
            descargar_cmip6(v, m, e, level=l)


    if descargas_fallidas:
        print('\nHa habido fallos en la descarga de las siguientes variables:')
        for fallo in descargas_fallidas:
            print(f"  {fallo}")
    else:
        print('\nTodas las descargas se han realizado con éxito')


'''
Eastward wind (m s-1) --> Magnitude of the eastward component of the two-dimensional horizontal air velocity.

Northward wind (m s-1) --> Magnitude of the northward component of the two-dimensional horizontal air velocity.

Near-Surface air temperature (K) --> Temperature of air at 2m above the surface of land, sea or inland waters. 2m temperature 
                                        is calculated by interpolating between the lowest model level and the Earth's surface, 
                                        taking account of the atmospheric conditions.
                                        
Precipitation (kg m-2 s-1) --> The sum of liquid and frozen water, comprising rain and snow, that falls to the Earth's surface. 
                                It is the sum of large-scale precipitation and convective precipitation. This parameter does 
                                not include fog, dew or the precipitation that evaporates in the atmosphere before it lands 
                                at the surface of the Earth. This variable represents amount of water per unit area and time.

Sea level pressure (Pa) --> The pressure (force per unit area) of the atmosphere at the surface of the Earth, adjusted to 
                            the height of sea level. It is a measure of the weight that all the air in a column vertically 
                            above a point on the Earth's surface would have, if the point were located at sea level. It is 
                            calculated over all surfaces - land, sea and inland water.

Sea surface temperature (degC) --> Temperature of sea water near the surface. 

TOA outgoing longwave radiation (W m-2) --> Longwave radiation from the top of the atmosphere to space per unit area.
'''


