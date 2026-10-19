"""
comparacion_miembro_ensamble.py
Control de calidad tras la descarga: comprueba que, para cada par modelo-experimento, todas las variables comparten el mismo variant_label/member_id.
Entradas: ficheros .nc de data/raw/cmip6/.
Salidas: informe por pantalla (no escribe ficheros).
Memoria: secciones 4.3 y 5.10.
"""

import os
import xarray as xr
from pathlib import Path

# Definimos directorio de entrada
root = Path(__file__).resolve().parent.parent.parent
root_data = Path(os.getenv("DATA_PATH", root))
if not root_data.exists(): root_data = root
raw = root_data / "data" / "raw" / "cmip6"

# Recogemos metadatos de todas las variables
filas = {}
for f in sorted(raw.rglob("*.nc")):
    ds = xr.open_dataset(f, decode_times=False)
    a = ds.attrs
    m = a.get("source_id", "?")
    e = a.get("experiment_id", "?")
    miembro = a.get("variant_label", a.get("member_id", "?"))
    var = a.get("variable_id", "?")
    origen = a.get("source", "CDS")
    filas.setdefault((m, e), []).append((var, f.parent.parent.name, miembro, origen))

# Análisis de miembros de cada par modelo-experimento (deben ser los mismos para las distintas variables dentro del par)
for (m, e), lst in sorted(filas.items()):
    miembros = sorted({x[2] for x in lst})
    marca = "" if len(miembros) == 1 else "   <-- MIEMBROS DISTINTOS"
    print(f"{m} - {e}: {', '.join(miembros)}{marca}")
    if len(miembros) > 1:
        for var, campo, mi, org in sorted(lst, key=lambda x: x[2]):
            print(f"    {campo:<28} {var:<4} {mi:<10} {org}")