"""
download_hadisst.py
Descarga la SST de HadISST 1.1 (Met Office) y la descomprime. https://climatedataguide.ucar.edu/climate-data/sst-data-hadisst-v11
Entradas: ninguna (acceso directo).
Salidas: data/raw/hadisst/sea_surface_temperature/HadISST_sst.nc
Memoria: sección 4.2.
"""

from pathlib import Path
import requests
import gzip
import shutil
import os

# Definimos directorio de salida
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root

data_dir = root_data / "data" / "raw" / "hadisst" / "sea_surface_temperature"
data_dir.mkdir(parents=True, exist_ok=True)
gz_filename = data_dir / "HadISST_sst.nc.gz"
nc_filename = data_dir / "HadISST_sst.nc"

# Descarga
url = "https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz"
response = requests.get(url, stream=True)
response.raise_for_status()
with open(gz_filename, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Descomprimir
with gzip.open(gz_filename, 'rb') as f_in:
    with open(nc_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Borramos archivo comprimido
os.remove(gz_filename)

