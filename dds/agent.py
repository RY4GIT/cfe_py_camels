import os
import pandas as pd
import numpy as np
from model import CFEmodel
import datetime
import shutil
import spotpy
import json
import csv


def read_spotpy_config(config_path):
    """Reads configuration setting for spotpy calibration analysis from text file."""

    # Read the data
    df = pd.read_csv(config_path)
    config_spotpy = df[df["calibrate"] == 1]

    return config_spotpy


def create_output_folder(home_dir, study_site):
    # Define output folder
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    out_dir = os.path.join(
        home_dir,
        "results",
        f"{study_site}-{current_date}",
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


class Spotpy_Agent:
    def __init__(self, config=None):
        """Initialize the Spotpy agent

        Args:
            config (_type_, optional): _description_. Defaults to None.
        """

        # Setup config files and output directories
        self.config = config
        self.nrun = int(self.config["DDS"]["N"])
        self.out_dir = create_output_folder(
            config["PATHS"]["homedir"], self.config["DATA"]["site"]
        )

        # Setting manual seed for reproducibility
        self.seed = 0
        np.random.seed(self.seed)

        # Setup spotpy
        self.spotpy_setup = Spotpy_setup(config=self.config)
        spotpy_runtype = self.config["spotpy"]["method"]
        if spotpy_runtype == "DDS":
            self.sampler = spotpy.algorithms.dds(
                self.spotpy_setup,
                dbname="raw_result_file",
            )
            # self.sampler = spotpy.algorithms.dds(
            #     self.spotpy_setup, parallel="mpi", dbname="raw_result_file"
            # )
            # mpiexec -n 2 C:/Users/flipl/miniconda3/envs/CFE/python.exe .\1_pre_calibration\__main__.py This stuck forever
        else:
            print(f"Invalid runtype: {spotpy_runtype}")

    def run(self):
        """Implement spotpy analysis"""
        # x_inital = self.spotpy_setup.param_bounds.set_index("name")[
        #     "optguess"
        # ].to_dict()
        # self.sampler.sample(self.nrun, x_initial=x_inital)
        self.sampler.sample(self.nrun)
        self.results = self.sampler.getdata()
        df = pd.DataFrame(self.results)
        df[
            [
                "like1",
                "parbb",
                "parsatdk",
                "parslop",
                "parsmcmax",
                "parwltsmc",
                "parmax_gw_storage",
                "parsatpsi",
                "parCgw",
                "parexpon",
                "parK_nash",
                "parrefkdt",
                "partrigger_z_fact",
                "paralpha_fc",
                "parK_lf",
                "parnum_nash_storage",
            ]
        ].to_csv(os.path.join(self.out_dir, "DDS_allresults.csv"))

    def finalize(self):
        self.get_the_best_run(self.results)
        self.remove_temp_files()

    def get_the_best_run(self, results):
        # Save parameter bounds used for calibration
        self.spotpy_setup.param_bounds.to_csv(
            os.path.join(self.out_dir, "parameter_bounds_used.csv")
        )

        # Get the best parameter in a dictionary format
        best_params_ = spotpy.analyser.get_best_parameterset(results, maximize=True)[0]
        best_params = dict()
        for i, row in self.spotpy_setup.param_bounds.iterrows():
            best_params[row["name"]] = best_params_[i]

        # Get the best simulation run and objetive function
        bestindex, bestobjf = spotpy.analyser.get_maxlikeindex(results)
        best_model_run_ = results[bestindex]
        obj_values = results["like1"]
        # fields = [word for word in best_model_run.dtype.names if word.startswith("sim")]
        best_model_run = np.array(best_model_run_)

        # Save everything in a json
        best_run = {
            "best parameters": best_params,
            "best objective values": [bestobjf],
        }

        with open(os.path.join(self.out_dir, "DDS_bestrun_params.json"), "w") as f:
            json.dump(best_run, f, indent=4)

        # Open the CSV file in write mode
        with open(
            os.path.join(self.out_dir, "DDS_bestrun_Q.csv"), mode="w", newline=""
        ) as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            # Write the tuple to the CSV file
            csv_writer.writerow(best_model_run[0])

    def remove_temp_files(self):
        directory = os.path.join(
            os.path.dirname(self.config["PATHS"]["cfe_config"]),
            "temporary_parameter_files_for_calibration",
        )
        shutil.rmtree(directory)


class Spotpy_setup:
    def __init__(self, config=None):
        self.config = config
        self.setup_params()

    def setup_params(self):
        self.param_bounds = read_spotpy_config(
            config_path=self.config["PATHS"]["spotpy_config"]
        )

        # setup calibration parameters
        self.params = [
            spotpy.parameter.Uniform(
                row["name"],
                low=row["lower_bound"],
                high=row["upper_bound"],
                optguess=row["optguess"],
            )
            for i, row in self.param_bounds.iterrows()
        ]

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, x):
        self.model = CFEmodel(config=self.config, vector=x)

        try:
            self.model.run()
            self.sim_results = self.model.return_sim_data()
            return self.sim_results.values

        except Exception as e:
            print(f"Error in simulation: {e}")
            return np.nan  # TODO: fix the length?

    def evaluation(self, evaldates=False):
        vector_ = self.parameters()
        vector = dict()
        for item in vector_:
            vector[item[1]] = item[0]
        self.model = CFEmodel(config=self.config, vector=vector)
        self.obs_data = self.model.return_obs_data()
        return self.obs_data.values

    def objectivefunction(self, simulation, evaluation):
        if np.isnan(simulation.all()):
            self.obj_function = np.nan
        else:
            if self.config["spotpy"]["like_measure"] == "NashSutcliffe":
                self.obj_function = spotpy.objectivefunctions.nashsutcliffe(
                    evaluation[~np.isnan(evaluation)], simulation[~np.isnan(evaluation)]
                )
            elif self.config["spotpy"]["like_measure"] == "KGE":
                self.obj_function = spotpy.objectivefunctions.kge(
                    evaluation[~np.isnan(evaluation)], simulation[~np.isnan(evaluation)]
                )
        return self.obj_function
