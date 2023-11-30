import os
import pandas as pd

pd.options.mode.chained_assignment = None
import json
import warnings

import sys

parent_dir = r"G:\Shared drives\Ryoko and Hilary\cfe_py_camels"
sys.path.append(os.path.join(parent_dir,"cfe_py"))
from bmi_cfe import BMI_CFE
from fao_pet import FAO_PET

import shutil
import random
import numpy as np


def duplicate_file(cfg):

    if cfg['DDS']['base_estimate'] == "NWM":
        filename = f"cat_{cfg['DATA']['basin_id']}_bmi_config_cfe.json"
    elif cfg['DDS']['base_estimate'] == "myDDS":
        filename = f"cat_{cfg['DATA']['basin_id']}_dds_calibrated.json"
    # Determine the directory and make a path for the new file
    source_path = os.path.join(
        cfg["PATHS"]["cfe_config"],
        filename,
    )

    temp_path = os.path.join(cfg["PATHS"]["homedir"],"temp")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    destination_path = os.path.join(temp_path, filename)

    # Copy the source file to the new location
    shutil.copy2(source_path, destination_path)

    return destination_path  # Optionally return the path to the new file


class CFEmodel:
    def __init__(self, config=None, vector=None):
        # Configs
        self.config = config
        self.basin_id = config["DATA"]["basin_id"]
        self.like_measure = config["spotpy"]["like_measure"]
        self.eval_variable = config["spotpy"]["eval_variable"]
        self.warmup_offset = int(config["spotpy"]["warmup_offset"])
        self.data_dir = config["PATHS"]["DATA"]

        self.start_time = config["DATA"]["start_time"]
        self.end_time = config["DATA"]["end_time"]

        # Copy the CFE config file for sensitivty analysis
        destination_path = duplicate_file(cfg=config)

        # write the randomly-generated parameters to the config json file
        with open(destination_path) as data_file:
            cfe_cfg = json.load(data_file)

        soil_params_keys = ["bb", "satdk", "slop", "smcmax"]

        try: 
            for key in vector.keys():
                value = vector[key]  # Adjust this line based on how you access values in 'vector'

                if key in soil_params_keys:
                    cfe_cfg["soil_params"][key] = value
                else:
                    cfe_cfg[key] = value

                if key in ["satdk", "Cgw"]:  # Assign to self if needed
                    setattr(self, key, value)
        except:
            for key in vector.name:
                value = getattr(vector,key)  # Adjust this line based on how you access values in 'vector'

                if key in soil_params_keys:
                    cfe_cfg["soil_params"][key] = value
                else:
                    cfe_cfg[key] = value

                if key in ["satdk", "Cgw"]:  # Assign to self if needed
                    setattr(self, key, value)

        cfe_cfg["partition_scheme"] = "Schaake"
        cfe_cfg["soil_scheme"] = "classic"

        with open(destination_path, "w") as out_file:
            json.dump(cfe_cfg, out_file, indent=4)

        # Here the model is actualy started with a unique parameter combination that it gets from spotpy for each time the model is called
        self.cfe_instance = BMI_CFE(cfg_file=destination_path, partitioning_scheme=cfe_cfg["partition_scheme"], soil_scheme=cfe_cfg["soil_scheme"])
        self.cfe_instance.initialize(Cgw=self.Cgw, satdk=self.satdk)

        self.read_data()

    def read_data(self):
        filename = f"{self.basin_id}_hourly_nldas.csv"
        _forcing_df = pd.read_csv(os.path.join(self.data_dir, "nldas_hourly", filename))
        _forcing_df.set_index(pd.to_datetime(_forcing_df["date"]), inplace=True)
        forcing_df = _forcing_df[self.start_time:self.end_time].copy()
        forcing_df.head()
        conversions_m_to_mm = 1000
        self.precip =  forcing_df["total_precipitation"].values / conversions_m_to_mm
        self.pet = FAO_PET(nldas_forcing=forcing_df, basin_id=self.basin_id).calc_PET().values

        filename = f"{self.basin_id}-usgs-hourly.csv"
        obs_q_ = pd.read_csv(os.path.join(self.data_dir, "usgs_streamflow", filename))
        obs_q_.set_index(pd.to_datetime(obs_q_["date"]), inplace=True)
        # obs_q_ = obs_q_["QObs(mm/h)"].values / conversions_m_to_mm
        q = obs_q_[self.start_time:self.end_time].copy()
        self.obs_q = q["QObs(mm/h)"] / conversions_m_to_mm

    def run(self):
        output_name = "land_surface_water__runoff_depth"
        self.sim_q = np.empty(len(self.precip))

        for t, (precip_t, pet_t) in enumerate(zip(self.precip, self.pet)):
            
            self.cfe_instance.set_value('atmosphere_water__time_integral_of_precipitation_mass_flux', precip_t)
            self.cfe_instance.set_value("water_potential_evaporation_flux", pet_t)
            
            self.cfe_instance.update()
            self.sim_q[t] = self.cfe_instance.get_value(output_name)
        self.cfe_instance.finalize(print_mass_balance=False)

    def return_runoff(self):
        sim = self.sim_q
        obs = self.obs_q

        if obs.index[0] != sim.index[0]:
            warnings.warn(
                "The start of observation and simulation time is different by %s"
                % obs.index[0]
                - sim.index[0]
            )

        if obs.index[-1] != sim.index[-1]:
            warnings.warn(
                "The end of observation and simulation time is different by %s"
                % obs.index[-1]
                - sim.index[-1]
            )

        df = pd.merge_asof(sim, obs, on="Time")

        sim_synced = df[self.eval_variable + "_x"]
        obs_synced = df[self.eval_variable + "_y"]

        return sim_synced, obs_synced

    def to_datetime(self, df, time_column, format):
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column], format=format)
        return df.set_index(time_column)
