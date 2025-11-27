###############################
### PostProcessing.py ###
###############################

### PURPOSE: This code defines the following classes:
####  ResultsProcessor: Processes simulation results and calculates performance metrics
####  ResultsVisualizer: Visualizes the results of the Monte Carlo experiment
### PROGRAMMER: Gabriel Durham (GJD)
### CREATED ON: 2 NOV 2025 
### EDITS: 18 NOV 2025 (GJD): Edited to enable specifying multiple averages (over time) to estimate
###             E.g., wanting tau hat varied averaging tau_t hats from 1,...,t_max for various t_max values
###             Will help with time asymptotics sims (avoid having to run different simulations for each T)


import pandas as pd
import numpy as np
from scipy.stats import norm
import re


class ResultsProcessor:
    def __init__(self, simulation_results):
        # Can make this config-controlled later
        self.alpha = 0.05
        self.inf_data = simulation_results.loc[simulation_results["inference"]==True,:].reset_index(drop=True)
        self.add_performance_metrics()
        self.process_all_estimands()
        
        
    
    def add_performance_metrics(self):
        self.inf_data.loc[:, "bias"] = self.inf_data.loc[:, "estimated_val"] - self.inf_data.loc[:, "true_val"]
        self.inf_data.loc[:, "bias2"] = self.inf_data.loc[:, "bias"]**2
        critical_val = norm.ppf(1 - self.alpha/2)
        
        #se = np.where(
        #    self.inf_data["type"].isin(["tau_at", "tau_t"]),
        #    ((self.inf_data["estimated_var"]/self.inf_data["n"])**0.5),
        #    ((self.inf_data["estimated_var"]/self.inf_data["T"])**0.5)
        #)
        se = np.where(
            self.inf_data["type"].isin(["tau_at", "tau_t"]),
            ((self.inf_data["estimated_var"])**0.5),
            #((self.inf_data["estimated_var"]/self.inf_data["t"])**0.5)
            # If the estimator is tau, then it was an average of 1,....,t
            ((self.inf_data["estimated_var"]/self.inf_data["t"])**0.5)
        )
        #se = self.inf_data["estimated_var"]**0.5
        self.inf_data.loc[:, "ci_lower"] = self.inf_data.loc[:, "estimated_val"] - critical_val*se
        self.inf_data.loc[:, "ci_upper"] = self.inf_data.loc[:, "estimated_val"] + critical_val*se
        self.inf_data.loc[:, "coverage"] = ( (self.inf_data.loc[:, "true_val"]>=self.inf_data.loc[:, "ci_lower"]) & 
                                            (self.inf_data.loc[:, "true_val"]<=self.inf_data.loc[:, "ci_upper"]) )
        
        
    def process_all_estimands(self):
        self.inf_performance = {}
        self.estimands_for_inf = np.unique(self.inf_data["estimand"].values)
        for estimand in self.estimands_for_inf:
            self.inf_performance[estimand] = self.process_single_estimand(estimand)
            
    def process_single_estimand(self, estimand):
        results = self.inf_data.loc[self.inf_data["estimand"]==estimand, :]
        Output = {}
        for metric in ["bias", "coverage"]:
            Output[metric] = {
                "MC_est":results[metric].mean(),
                "MC_se":results[metric].std()/(len(results[metric])**0.5),
            }
        Output["RMSE"] = {"MC_est":results["bias2"].mean()}
        return Output
    
class ResultsVisualizer:
    def __init__(self, processed_results):
        self.processed_results = processed_results
    
    def create_basic_summary_table(self, selected_estimands=None, n_round=2, latex_formatting=False):
        output_table = pd.DataFrame()
        if selected_estimands is None:
            selected_estimands = self.processed_results.estimands_for_inf
        for estimand in selected_estimands:
            # Create the label for the estimand
            if latex_formatting:
                new_row = pd.DataFrame({"Estimand":self.estimand_to_latex(estimand)}, index=[0])
            else:
                new_row = pd.DataFrame({"Estimand":estimand}, index=[0])
            # Grab the metrics
            for metric in self.processed_results.inf_performance[estimand].keys():
                performance = self.processed_results.inf_performance[estimand][metric]
                MC_est = performance["MC_est"]
                metric_lab = f"{round(MC_est, n_round):.{n_round}f}"
                if "MC_se" in performance.keys():
                    MC_se = performance["MC_se"]
                    std_lab = f"{round(MC_se, n_round):.{n_round}f}"
                    metric_lab = metric_lab + " ("+ std_lab + ")"
                new_row[self.smart_capitalize(metric)] = metric_lab
            output_table = pd.concat([output_table, new_row], ignore_index=True)
        return output_table
    
    def create_basic_summary_table_verbose(self, selected_estimands=None, latex_formatting=False):
        output_table = pd.DataFrame()
        if selected_estimands is None:
            selected_estimands = self.processed_results.estimands_for_inf
        for estimand in selected_estimands:
            # Create the label for the estimand
            if latex_formatting:
                new_row = pd.DataFrame({"Estimand":self.estimand_to_latex(estimand)}, index=[0])
            else:
                new_row = pd.DataFrame({"Estimand":estimand}, index=[0])
            # Grab the metrics
            for metric in self.processed_results.inf_performance[estimand].keys():
                performance = self.processed_results.inf_performance[estimand][metric]
                MC_est = performance["MC_est"]
                new_row[self.smart_capitalize(metric)] = MC_est
                if "MC_se" in performance.keys():
                    MC_se = performance["MC_se"]
                    new_row[self.smart_capitalize(metric)+"_MC_se"] = MC_se
                
            output_table = pd.concat([output_table, new_row], ignore_index=True)
        return output_table
    

    def estimand_to_latex(self, expr: str) -> str:
        # Map the leading symbol to its LaTeX form
        symbol_map = {
            "tau": r"\hat{\tau}",
            "theta": r"\hat{\theta}",
            "beta": r"\beta",
            "gamma": r"\gamma",
            # add more as needed
        }

        # Regex: symbol + optional _subscript + (parenthetical)
        # Examples matched: "tau_t(r1, r2)" and "tau(r1, r2)"
        m = re.match(r"""^\s*
                         ([A-Za-z]+)           # symbol
                         (?:_([^(]+))?         # optional subscript (no parens)
                         \(\s*([^)]+)\s*\)     # inside parentheses
                         \s*$""", expr, re.X)
        if not m:
            raise ValueError(f"Unrecognized format: {expr}")

        sym, sub, paren = m.groups()
        sym_ltx = symbol_map.get(sym, sym)      # default to raw symbol if not mapped

        # Split arguments by top-level commas (assumed simple r1, r2)
        args = [s.strip() for s in paren.split(",")]

        def fmt_arg(s: str) -> str:
            # If it looks like a dash-separated vector of integers, wrap as \set{…}
            if re.fullmatch(r"\d+(?:-\d+)*", s):
                return r"\set{" + s.replace("-", ",") + "}"
            # Otherwise pass through verbatim (you can add more rules here)
            return s

        args_ltx = ", ".join(fmt_arg(a) for a in args)

        subpart = f"_{{{sub}}}" if sub is not None else ""
        return rf"$ {sym_ltx}{subpart}\paren{{{args_ltx}}} $"
    def smart_capitalize(self, word: str) -> str:
        return word if word.isupper() else word.capitalize()