"""
download_cru.py
Descarga y descomprime precipitación y temperatura de CRU TS4.09 (UEA). https://crudata.uea.ac.uk/cru/data/hrg/
Entradas: ninguna (acceso directo).
Salidas: data/raw/cru/<variable>/cru_ts4.09.1901.2024.<pre|tmp>.dat.nc
Memoria: sección 4.2.
"""

from pathlib import Path
import gzip
import shutil
import requests
import os

variables = {"precipitation": "pre", "near_surface_air_temperature": "tmp"}
root_download = ("https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/cruts.2503051245.v4.09")

# Definimos directorio de salida
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

def descarga_cru(v):

    vn = variables[v]
    print(f"\nDescargando - {v}")

    data_dir = root_data / "data" / "raw" / "cru" / v
    data_dir.mkdir(parents=True, exist_ok=True)
    url_uea = f"{root_download}/{vn}/cru_ts4.09.1901.2024.{vn}.dat.nc.gz"
    gz_filename = data_dir / f"cru_ts4.09.1901.2024.{vn}.dat.nc.gz"
    nc_filename = data_dir / f"cru_ts4.09.1901.2024.{vn}.dat.nc"

    # Descarga UEA
    try:
        response = requests.get(url_uea, stream=True, timeout=60)
        response.raise_for_status()

        with open(gz_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        # Descomprimir
        with gzip.open(gz_filename, "rb") as f_in:
            with open(nc_filename, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Borramos archivo comprimido
        os.remove(gz_filename)

    except requests.exceptions.RequestException as e_uea:
        print(f"Error crítico: No se pudo conectar con la UEA: {e_uea}")
    except Exception as e_decomp:
        print(f"Error durante la descompresión del archivo de la UEA: {e_decomp}")


if __name__ == '__main__':

    for var in variables:
        descarga_cru(var)