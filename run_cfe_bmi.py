# %%
import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import bmi_cfe
import os
from fao_pet import FAO_PET

# %%
data_dir = "G:\Shared drives\SI_NextGen_Aridity\dCFE\data"
basin_id = "01137500"
partitioning_scheme = "Schaake"


# %%
output_dir = os.path.join(data_dir, "synthetic_case_from_original_code", partitioning_scheme)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
filename = f"{basin_id}_hourly_nldas.csv"
forcing_df = pd.read_csv(os.path.join(data_dir, filename))
forcing_df.set_index(pd.to_datetime(forcing_df["date"]), inplace=True)
forcing_df.head()

# # Convert pandas dataframe to PyTorch tensors
# Convert units
# (precip/1000)   # kg/m2/h = mm/h -> m/h
# (pet/1000/3600) # kg/m2/h = mm/h -> m/s
conversions_m_to_mm = 1000
precip =  forcing_df["total_precipitation"].values / conversions_m_to_mm

pet = FAO_PET(nldas_forcing=forcing_df, basin_id=basin_id).calc_PET()
# %%
