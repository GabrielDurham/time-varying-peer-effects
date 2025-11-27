####################
### simulator.py ###
####################

### PURPOSE: This code runs the simulations given driver input
###   Intended to be run from command line with single argument (corresponding to name of 
###   main driver file). Working directory should be project folder.
### PROGRAMMER: Gabriel Durham (GJD)
### CREATED ON: 17 NOV 2025 

import argparse
import os
import numpy as np
import pandas as pd
import sys
import yaml

sys.path.append("01_Code/src")  # add parent directory to Python path
from GroupAssignment import GroupAssignmentSimulator
from OutcomeGeneration import OutcomeSimulator
from AttributeGeneration import AttributeSimulator
from ExperimentSimulation import ExperimentSimulator
from SimulationEngine import AnalysisOverhead, ExperimentAnalyzer, BaseSimulator, _worker_run_block
#from PostProcessing import ResultsProcessor, ResultsVisualizer
from GeneralDataMaintenance import make_output_folder



def main():
    # 1. Read argument
    parser = argparse.ArgumentParser()
    parser.add_argument("driver_filename")
    args = parser.parse_args()

    driver_filename = args.driver_filename
    run_simulations(driver_filename)






def run_simulations(driver_filename):
    # Read in config
    yaml_parms = {}
    # If it's via a .yaml file, read it in directly, otherwise, parse a csv version (easier to process visually)
    # and make it look like a .yaml driver
    if driver_filename[(len(driver_filename)-5):]==".yaml":
        with open("01_Code/configs/main_driver/" + driver_filename, "r") as f:
            yaml_parms["driver"] = yaml.safe_load(f)
    elif driver_filename[(len(driver_filename)-4):]==".csv":
        driver_df = pd.read_csv("01_Code/configs/main_driver/" + driver_filename)
        yaml_parms["driver"] = {}
        for row in driver_df.index:
            driver_row = driver_df.loc[row,:]
            sim_id = driver_row["sim_label"]
            yaml_parms["driver"][sim_id] = {}
            for var in ["run_simulation", "n", "T", "n_iter", "seed"]:
                # Storing explicitly as int to avoid yaml storage headaches down the line
                yaml_parms["driver"][sim_id][var] = int(driver_row[var])
            yaml_parms["driver"][sim_id]["configs"] = {}
            for config_type in ["attribute", "randomization", "analysis"]:
                yaml_parms["driver"][sim_id]["configs"][config_type] = driver_row["configs_"+config_type]
            yaml_parms["driver"][sim_id]["configs"]["outcome"] = {
                "y0":driver_row["configs_outcome_y0"],
                "tau":driver_row["configs_outcome_tau"],
            }
    # Populate the settings for the yaml file
    for setting in yaml_parms["driver"].keys():
        if setting=="driver":
            ValueError("Cannot use \'driver\' as simulation id label.")
        yaml_parms[setting] = {}
        for var in ["run_simulation", "n", "T", "n_iter", "seed"]:
            yaml_parms[setting][var] = yaml_parms["driver"][setting][var]
        for config_type in ["attribute", "randomization", "analysis"]:
            with open("01_Code/configs/" + config_type + "/" + yaml_parms["driver"][setting]["configs"][config_type], "r") as f:
                yaml_parms[setting][config_type] = yaml.safe_load(f)
        yaml_parms[setting]["outcome"] = {}
        with open("01_Code/configs/outcome/" + yaml_parms["driver"][setting]["configs"]["outcome"]["y0"], "r") as f:
            yaml_parms[setting]["outcome"]["y0"] = yaml.safe_load(f)
        with open("01_Code/configs/effect/" + yaml_parms["driver"][setting]["configs"]["outcome"]["tau"], "r") as f:
            yaml_parms[setting]["outcome"]["tau"] = yaml.safe_load(f)

    # Identify simulation IDs that we want to run
    sim_ids = [sim_id for sim_id in yaml_parms["driver"].keys() if yaml_parms["driver"][sim_id]["run_simulation"]==1]
    # Make an output folder
    if sim_ids:
        output_path = make_output_folder()
        # Save yaml_parms for transparency
        with open(output_path + "sim_parms.yaml", "w") as f:
            yaml.safe_dump(yaml_parms, f)
    # Run through simulations
    for sim_id in sim_ids:
        outcome_simulator = OutcomeSimulator(yaml_parms=yaml_parms[sim_id]["outcome"])
        attribute_simulator = AttributeSimulator(yaml_parms=yaml_parms[sim_id]["attribute"])
        group_simulator = GroupAssignmentSimulator(yaml_parms=yaml_parms[sim_id]["randomization"], 
                                                   H=attribute_simulator.H)
        analysis_overhead = AnalysisOverhead(yaml_parms=yaml_parms[sim_id]["analysis"], 
                                             n=yaml_parms[sim_id]["n"], 
                                             T=yaml_parms[sim_id]["T"], 
                                             m=group_simulator.m,
                                             H=group_simulator.H)
        simulator = BaseSimulator(analysis_overhead=analysis_overhead, 
                                  outcome_simulator=outcome_simulator, 
                                  attribute_simulator=attribute_simulator, 
                                  group_simulator=group_simulator)
        # The number of threads is set to five here, can alter it I guess
        sim_results = simulator.run_sims(n_iter=yaml_parms[sim_id]["n_iter"], 
                                         n_thread = 5, 
                                         seed=yaml_parms[sim_id]["seed"])
        # Overwrite some empty strings with np.nan for easy storage to parquet
        sim_results["a"] = sim_results["a"].replace("", np.nan)
        sim_results["t"] = sim_results["t"].replace("", np.nan)

        file_name = "results_" + sim_id + ".parquet"

        sim_results.to_parquet(output_path + file_name)

        
        
if __name__ == "__main__":
    main()