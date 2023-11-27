import os
import pandas as pd

pd.options.mode.chained_assignment = None
import json
import warnings

import sys

sys.path.append(os.path.join(os.getcwd(), "libs", "cfe_py"))
from bmi_cfe import BMI_CFE

import shutil
import random


def duplicate_file(source_path):
    i = random.randint(1, 9999)
    # Determine the directory and make a path for the new file
    directory = os.path.join(
        os.path.dirname(source_path),
        "temporary_parameter_files_for_calibration",
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    destination_path = os.path.join(directory, f"cfe_config_{i}.json")

    # Copy the source file to the new location
    shutil.copy2(source_path, destination_path)

    return destination_path  # Optionally return the path to the new file


class CFEmodel:
    def __init__(self, config=None, vector=None):
        # Configs
        self.config = config
        self.like_measure = config["spotpy"]["like_measure"]
        self.eval_variable = config["spotpy"]["eval_variable"]
        self.warmup_offset = int(config["spotpy"]["warmup_offset"])
        self.warmup_iteration = int(config["spotpy"]["warmup_iteration"])

        # Copy the CFE config file for sensitivty analysis
        destination_path = duplicate_file(
            source_path=self.config["PATHS"]["cfe_config"]
        )

        self.cfe_instance = BMI_CFE(cfg_file=destination_path)

        # write the randomly-generated parameters to the config json file
        with open(self.cfe_instance.cfg_file) as data_file:
            cfe_cfg = json.load(data_file)

        cfe_cfg["soil_params"]["bb"] = vector["bb"]
        cfe_cfg["soil_params"]["satdk"] = vector["satdk"]
        cfe_cfg["soil_params"]["slop"] = vector["slop"]
        cfe_cfg["soil_params"]["smcmax"] = vector["smcmax"]
        cfe_cfg["soil_params"]["wltsmc"] = vector["wltsmc"]
        cfe_cfg["max_gw_storage"] = vector["max_gw_storage"]
        cfe_cfg["soil_params"]["satpsi"] = vector["satpsi"]
        cfe_cfg["Cgw"] = vector["Cgw"]
        cfe_cfg["expon"] = vector["expon"]
        cfe_cfg["K_nash"] = vector["K_nash"]
        cfe_cfg["refkdt"] = vector["refkdt"]
        cfe_cfg["trigger_z_fact"] = vector["trigger_z_fact"]
        cfe_cfg["alpha_fc"] = vector["alpha_fc"]
        cfe_cfg["K_lf"] = vector["K_lf"]
        cfe_cfg["num_nash_storage"] = int(vector["num_nash_storage"])

        with open(self.cfe_instance.cfg_file, "w") as out_file:
            json.dump(cfe_cfg, out_file, indent=4)

        # Here the model is actualy started with a unique parameter combination that it gets from spotpy for each time the model is called
        self.cfe_instance.initialize()

    def return_obs_data(self):
        self.cfe_instance.load_unit_test_data()
        obs = self.cfe_instance.unit_test_data[self.eval_variable]
        return obs

    def run(self):
        self.cfe_instance.run_unit_test(
            plot=False,
            print_fluxes=False,
            warm_up=True,
            warmup_offset=self.warmup_offset,
            warmup_iteration=self.warmup_iteration,
        )

    def return_sim_data(self):
        if self.eval_variable == "Flow":
            sim = self.cfe_instance.cfe_output_data["Flow"]
        elif self.eval_variable == "Soil Moisture Content":
            sim = (
                self.cfe_instance.cfe_output_data["SM storage"]
                / self.cfe_instance.soil_params["D"]
            )
        return sim

    def return_runoff(self):
        sim = self.return_sim_runoff()
        obs = self.return_obs_runoff()

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
