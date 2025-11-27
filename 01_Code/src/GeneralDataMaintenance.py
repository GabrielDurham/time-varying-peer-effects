#################################
### GeneralDataMaintenance.py ###
#################################

### PURPOSE: This code handles data pipeline shenanigans
### PROGRAMMER: Gabriel Durham (GJD)
### CREATED ON: 17 NOV 2025 
### EDITS: 


from pathlib import Path
from datetime import datetime
import string

# Make Output Folder for Raw Results
def make_output_folder():
    base = Path("02_Data")
    base.mkdir(exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    folder = base / f"sim_output_{date_str}"

    # If it doesn't exist, use it
    if not folder.exists():
        folder.mkdir()
        return str(folder) + "/"

    # Otherwise try sim_output_YYYYMMDDa, b, c, ...
    for suffix in string.ascii_lowercase:
        candidate = base / f"sim_output_{date_str}{suffix}"
        if not candidate.exists():
            candidate.mkdir()
            return str(candidate) + "/"

    raise RuntimeError("Ran out of suffixes")