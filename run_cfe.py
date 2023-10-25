# %%
import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import bmi_cfe
import os
from fao_pet import FAO_PET

# %% [markdown]
# ## Read CAMELS data

# %%

def run_model(data_dir, basin_id, partitioning_scheme, soil_scheme):
    # %% Create output directory
    output_dir = os.path.join(data_dir, "synthetic_case_from_original_code", partitioning_scheme)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # %% Read Forcing 
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

    pet = FAO_PET(nldas_forcing=forcing_df, basin_id=basin_id).calc_PET().values

    # %% Read observation data
    filename = f"{basin_id}-usgs-hourly.csv"
    obs_q_ = pd.read_csv(os.path.join(data_dir, filename))
    obs_q_.set_index(pd.to_datetime(obs_q_["date"]), inplace=True)
    q = obs_q_["QObs(mm/h)"].values / conversions_m_to_mm

    # # %%
    # plt.plot(precip)
    # plt.plot(pet*3600)
    # plt.plot(q)

    # %% [markdown]
    # ## Run Normal CFE Simulations
    filename = f"cat_{basin_id}_bmi_config_cfe.json"
    cfe_instance = bmi_cfe.BMI_CFE(cfg_file=os.path.join(data_dir, filename), soil_scheme=soil_scheme, partitioning_scheme=partitioning_scheme)
    cfe_instance.stand_alone = 0
    cfe_instance.initialize()

    # %%
    outputs = cfe_instance.get_output_var_names()
    output_lists = {output:[] for output in outputs}

    for precip_t, pet_t in zip(precip, pet):
        
        cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip_t)
        cfe_instance.set_value("water_potential_evaporation_flux", pet_t)
        
        cfe_instance.update()
        
        for output in outputs:
        
            output_lists[output].append(cfe_instance.get_value(output))

    # %%
    cfe_instance.finalize(print_mass_balance=True)

    # %%
    # istart_plot = 490
    # iend_plot = 550
    # x = list(range(istart_plot, iend_plot))
    # for output in outputs:
    #     plt.plot(output_lists[output], label=output)
    #     plt.title(output)
    #     plt.legend()
    #     plt.show()
    #     plt.close()

    # %%
    # plt.plot(obs_q_["QObs(mm/h)"].values/conversions_m_to_mm, alpha=.5)
    # plt.plot(output_lists["land_surface_water__runoff_depth"], alpha=.5)

    # %% [markdown]
    # # Save results

    # %% Get the flow output
    sim_q = np.array(output_lists["land_surface_water__runoff_depth"]) * conversions_m_to_mm
    # Create a pandas DataFrame from the indexed list
    output_df = pd.DataFrame(sim_q, index=forcing_df.index, columns=["simQ[m/hr]"])
    # output_df.head()

    # %% Output 
    filename = f"{basin_id}_synthetic_{soil_scheme}.csv"
    output_df.to_csv(os.path.join(output_dir, filename))

# %%

def main():
    data_dir = "G:\Shared drives\SI_NextGen_Aridity\dCFE\data"
    basin_ids = ["01022500", "01031500", "01137500"]
    partitioning_schemes = ["Schaake"]
    soil_schemes = ["classic", "ode"]
    for basin_id in basin_ids:
        for partitioning_scheme in partitioning_schemes:
            for soil_scheme in soil_schemes:
                print(f"Processing {basin_id} - {partitioning_scheme} - {soil_scheme}")
                run_model(data_dir, basin_id, partitioning_scheme, soil_scheme)

if __name__ == "__main__":
    main()



