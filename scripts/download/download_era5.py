"""
download_era5.py
Descarga ERA5 (medias mensuales, 1950-2014) desde el CDS: viento (850/200 hPa), geopotencial (200 hPa), PSL y flujo neto de onda larga en el techo (OLR con el signo cambiado).
Entradas: API de CDS (~/.cdsapirc).
Salidas: data/raw/era5/<variable>[_<nivel>hPa]/ERA5_<variable>[_<nivel>hPa].nc
Memoria: sección 4.2.
"""

import cdsapi
from pathlib import Path
import os

variables_pl = ["u_component_of_wind", "v_component_of_wind"]
variables_sl = ["mean_sea_level_pressure", "mean_top_net_long_wave_radiation_flux"]

# Definimos directorio de salida
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root
raw = root_data / "data" / "raw" / "era5"

def descargar_era5_pl(v,l):

    # Definimos dataset a descargar
    dataset = "reanalysis-era5-pressure-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [v],
        "pressure_level": [l],
        "year": [str(year) for year in range(1950, 2015)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"}

    # Definimos lugar y nombre de descarga
    data_dir = raw / f"{v}_{l}hPa"
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = data_dir / f"ERA5_{v}_{l}hPa.nc"
    if filename.exists(): return

    # Descarga
    try:
        print(f"Descargando {v}")
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(filename)

    except Exception as e:
        print(f"Error al procesar {v}: {e}")

def descargar_era5_sl(v):

    # Definimos dataset a descargar
    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [v],
        "year": [str(year) for year in range(1950, 2015)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    # Definimos lugar y nombre de descarga
    data_dir = raw / f"{v}"
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = data_dir / f"ERA5_{v}.nc"
    if filename.exists(): return

    # Descarga
    try:
        print(f"Descargando {v}")
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(filename)

    except Exception as e:
        print(f"Error al procesar {v}: {e}")

if __name__ == '__main__':

    for v in variables_sl:
        descargar_era5_sl(v)

    for v in variables_pl:
        for l in ["850","200"]:
            descargar_era5_pl(v, l)

    descargar_era5_pl("geopotential", "200")

'''
Geopotential (m2 s-2) --> This parameter is the gravitational potential energy of a unit mass, at a particular location at 
                            the surface of the Earth, relative to mean sea level. It is also the amount of work that would 
                            have to be done, against the force of gravity, to lift a unit mass to that location from mean 
                            sea level. The geopotential height can be calculated by dividing the geopotential by the Earth's 
                            gravitational acceleration, g (=9.80665 m s-2). The geopotential height plays an important role 
                            in synoptic meteorology (analysis of weather patterns). Charts of geopotential height plotted 
                            at constant pressure levels (e.g., 300, 500 or 850 hPa) can be used to identify weather systems 
                            such as cyclones, anticyclones, troughs and ridges. At the surface of the Earth, this parameter 
                            shows the variations in geopotential (height) of the surface, and is often referred to as the orography.
                            
U-component of wind (m s-1) --> This parameter is the eastward component of the wind. It is the horizontal speed of air 
                                    moving towards the east. A negative sign indicates air moving towards the west.

V-component of wind (m s-1) --> This parameter is the northward component of the wind. It is the horizontal speed of air 
                                    moving towards the north. A negative sign indicates air moving towards the south.
                 
2m temperature (K) --> This parameter is the temperature of air at 2m above the surface of land, sea or inland waters. 2m 
                        temperature is calculated by interpolating between the lowest model level and the Earth's surface, 
                        taking account of the atmospheric conditions. This parameter has units of kelvin (K). Temperature 
                        measured in kelvin can be converted to degrees Celsius (°C) by subtracting 273.15.
           
Mean sea level pressure (Pa) --> This parameter is the pressure (force per unit area) of the atmosphere at the surface of 
                                    the Earth, adjusted to the height of mean sea level. It is a measure of the weight that 
                                    all the air in a column vertically above a point on the Earth's surface would have, if 
                                    the point were located at mean sea level. It is calculated over all surfaces - land, 
                                    sea and inland water. Maps of mean sea level pressure are used to identify the locations 
                                    of low and high pressure weather systems, often referred to as cyclones and anticyclones. 
                                    Contours of mean sea level pressure also indicate the strength of the wind. Tightly packed 
                                    contours show stronger winds. The units of this parameter are pascals (Pa). Mean sea 
                                    level pressure is often measured in hPa and sometimes is presented in the old units of 
                                    millibars, mb (1 hPa = 1 mb = 100 Pa).

Top net thermal radiation (J m-2) --> The thermal (also known as terrestrial or longwave) radiation emitted to space at 
                                        the top of the atmosphere is commonly known as the Outgoing Longwave Radiation (OLR).
                                        The top net thermal radiation (this parameter) is equal to the negative of OLR. 
                                        This parameter is accumulated over a particular time period which depends on the 
                                        data extracted. For the monthly averaged reanalysis and the monthly averaged ensemble
                                        members, the accumulation period is 1 day. For the monthly averaged reanalysis by 
                                        hour of day, the accumulation period is 1 hour and for the monthly averaged ensemble 
                                        members by hour of day, the accumulation period is 3 hours. The units are joules 
                                        per square metre (J m^-2 ). To convert to watts per square metre (W m^-2 ), the 
                                        accumulated values should be divided by the accumulation period expressed in seconds. 
                                        The ECMWF convention for vertical fluxes is positive downwards.

Mean top net long-wave radiation flux (W m-2) --> The thermal (also known as terrestrial or longwave) radiation emitted to 
                                                    space at the top of the atmosphere is commonly known as the Outgoing 
                                                    Longwave Radiation (OLR). The top net thermal radiation (this parameter) 
                                                    is equal to the negative of OLR. This parameter is a mean over a particular 
                                                    time period (the processing period) which depends on the data extracted. 
                                                    For the monthly averaged reanalysis and the monthly averaged ensemble 
                                                    members, the processing period is the complete, whole month. For the 
                                                    monthly averaged reanalysis by hour of day, the processing period is 1 
                                                    hour for every day of the month and for the monthly averaged ensemble 
                                                    members by hour of day, the processing period is 3 hours for every day 
                                                    of the month. The ECMWF convention for vertical fluxes is positive downwards.
'''

