###############################
### SimulationEngine.py ###
###############################

### PURPOSE: This code defines the following classes:
####  AnalysisOverhead: Handles overhead for simulating/analyzing experiments
####  ExperimentAnalyzer: Calculates estimated and true values for all estimands in a given experiment
####  BaseSimulator: Iterates and simulates the necessary number of experiments
### PROGRAMMER: Gabriel Durham (GJD)
### CREATED ON: 2 NOV 2025 


import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import norm
from ExperimentSimulation import ExperimentSimulator


class AnalysisOverhead:
    def __init__(self, yaml_parms, n, T, m, H):
        self.n = n
        self.T = T
        self.m = m
        self.H = H
        
        self.estimand_yaml_parms = yaml_parms["estimands"]
        
        estimand_lists = self.identify_all_estimands()
        self.estimand_df = estimand_lists["all"]
        self.estimand_df_inference = estimand_lists["inference"]
        self.estimand_df.loc[:, "inference"] = self.estimand_df["estimand"].apply(lambda x:x in list(self.estimand_df_inference["estimand"]))
        
        
        estimator_shells = self.create_estimator_dict_shell()
        self.y_hat_shell = estimator_shells["y_hat_shell"]
        self.nu2_hat_shell = estimator_shells["nu2_hat_shell"]
    
    # Identify all estimands you'll need to target
    def identify_all_estimands(self):
        estimand_df = pd.DataFrame()
        # Also track just the ones you want to run inference on (to avoid running inference on all quantities you need to calculate)
        estimand_df_inference = pd.DataFrame()
        for estimand in self.estimand_yaml_parms.keys():
            estimand_parms = self.estimand_yaml_parms[estimand]
            r = estimand_parms["r1"]
            rr = estimand_parms["r2"]
            if estimand_parms["type"]=="tau_at":
                if estimand_parms["a"]=="all":
                    a_list = list(range(self.H))
                else:
                    a_list = estimand_parms["a"]
                if estimand_parms["t"]=="all":
                    t_list = list(range(1,self.T+1))
                else:
                    t_list = estimand_parms["t"]
                new_rows = self.enumerate_single_estimand("tau_at", r, rr, a_list, t_list)
                new_rows_inference = self.enumerate_single_estimand("tau_at", r, rr, a_list, t_list)
            elif estimand_parms["type"]=="tau_t":
                if estimand_parms["t"]=="all":
                    t_list = list(range(1,self.T+1))
                else:
                    t_list = estimand_parms["t"]
                # Need to calculate tau_t for all t and tau_at for all a and all t
                new_rows = pd.concat([self.enumerate_single_estimand("tau_t", r, rr, a_list=None, t_list=t_list),
                                      self.enumerate_single_estimand("tau_at", r, rr, a_list=list(range(self.H)), t_list=t_list)], 
                                     ignore_index=True)
                new_rows_inference = self.enumerate_single_estimand("tau_t", r, rr, a_list=None, t_list=t_list)
            elif estimand_parms["type"]=="tau":
                new_rows = pd.concat([self.enumerate_single_estimand("tau", r, rr),
                                      self.enumerate_single_estimand("tau_t", r, rr, a_list=None, t_list=list(range(1,self.T+1))),
                                      self.enumerate_single_estimand("tau_at", r, rr, list(range(self.H)), list(range(1,self.T+1)))], 
                                     ignore_index=True)
                new_rows_inference = self.enumerate_single_estimand("tau", r, rr)
            estimand_df = pd.concat([estimand_df, new_rows], ignore_index=True)
            estimand_df_inference = pd.concat([estimand_df_inference, new_rows_inference], ignore_index=True)
            
        return {"all":estimand_df.drop_duplicates().reset_index(drop=True),
                "inference":estimand_df_inference.drop_duplicates().reset_index(drop=True)}
        #return(estimand_df)
            
    # Create a dataframe row that has the estimand name, type, r, r', a, t values
    def enumerate_single_estimand(self, estimand_type, r, rr, a_list=None, t_list=None):
        if estimand_type=="tau_at":
            rows = list(product(["tau_at"],[r],[rr],a_list, t_list))
            estimand_data = pd.DataFrame(rows, columns=["type", "r", "rr", "a", "t"])
            estimand_data.loc[:,"estimand"] = ("tau_[" + estimand_data["a"].astype(str) + "]" 
                                            + estimand_data["t"].astype(str) + "(" 
                                            + estimand_data["r"].astype(str) + ","
                                            + estimand_data["rr"].astype(str) + ")")
        elif estimand_type=="tau_t":
            rows = list(product(["tau_t"],[r],[rr],[""], t_list))
            estimand_data = pd.DataFrame(rows, columns=["type", "r", "rr", "a", "t"])
            estimand_data.loc[:,"estimand"] = ("tau_" + estimand_data["t"].astype(str) + "(" 
                                               + estimand_data["r"].astype(str) + ","
                                               + estimand_data["rr"].astype(str) + ")")
        elif estimand_type=="tau":
            estimand_label = "tau("+str(r)+","+str(rr)+")"
            estimand_data = pd.DataFrame([{"type":"tau", "r":r, "rr":rr, "a":"", "t":"", "estimand":estimand_label}])
        return estimand_data
    
    # Define the primal quantities we'll need to estimate
    def identify_all_primal_quantities(self, estimand_df):
        # Identify Y_hats
        primal_estimand_df = pd.DataFrame(columns=["quantity", "type", "r", "a", "t"])
        for row in estimand_df.index:
            if estimand_df[row, "type"]=="tau_at":
                for r in ["r", "rr"]:
                    new_y_quantity = ("Yhat_["+str(estimand_df[row, "a"])+"]"+str(estimand_df[row, "t"])+
                                      "("+str(estimand_df[row, r])+")")
                    new_nu_quantity = ("nuhat_["+str(estimand_df[row, "a"])+"]"+str(estimand_df[row, "t"])+
                                       "("+str(estimand_df[row, r])+")")
                    new_row_y = pd.DataFrame([{"quantity":new_y_quantity, 
                                               "type":"Y_hat", 
                                               "r":r, 
                                               "a":estimand_df[row, "a"], 
                                               "t":estimand_df[row, "t"]}])
                    new_row_nu = pd.DataFrame([{"quantity":new_nu_quantity, 
                                                "type":"nu_hat", 
                                                "r":r, 
                                                "a":estimand_df[row, "a"], 
                                                "t":estimand_df[row, "t"]}])
                    primal_estimand_df = pd.concat([primal_estimand_df, new_row_y, new_row_nu], ignore_index=1)
    
    def create_estimator_dict_shell(self):
        y_hat_shell = {}
        nu2_hat_shell = {}
        for row in self.estimand_df.index:
            if self.estimand_df.loc[row, "type"]=="tau_at":
                temp_a = self.estimand_df.loc[row, "a"]
                temp_t = self.estimand_df.loc[row, "t"]
                temp_r = self.estimand_df.loc[row, "r"]
                temp_rr = self.estimand_df.loc[row, "rr"]
                # Store the a, t, and r for easy "filling in" (we will overwrite this later)
                y_hat_shell[self.atr_estimator_label("Yhat", temp_a, temp_t, temp_r)]={"a":temp_a, "t":temp_t, "r":temp_r}
                y_hat_shell[self.atr_estimator_label("Yhat", temp_a, temp_t, temp_rr)]={"a":temp_a, "t":temp_t, "r":temp_rr}
                nu2_hat_shell[self.atr_estimator_label("nu2hat", temp_a, temp_t, temp_r)]={"a":temp_a, "t":temp_t, "r":temp_r}
                nu2_hat_shell[self.atr_estimator_label("nu2hat", temp_a, temp_t, temp_rr)]={"a":temp_a, "t":temp_t, "r":temp_rr}
        return {"y_hat_shell":y_hat_shell, "nu2_hat_shell":nu2_hat_shell}
    def atr_estimator_label(self, qtype, a, t, r):
        return qtype+"_["+str(a)+"]"+str(t)+"("+str(r)+")"

    
    
# This class analyzes the experiment
class ExperimentAnalyzer:
    def __init__(self, analysis_overhead, experiment):
        self.n = analysis_overhead.n
        self.T = analysis_overhead.T
        self.m = analysis_overhead.m
        self.H = analysis_overhead.H

        
        self.atr_estimator_label = analysis_overhead.atr_estimator_label
        # Calculate estimated primal quantities
        self.y_hat = self.calculate_yhat(analysis_overhead.y_hat_shell, experiment)
        self.nu2_hat = self.calculate_nu2hat(analysis_overhead.nu2_hat_shell, experiment)
        # Calculate true primal quantities
        self.y_bar = self.calculate_ybar(analysis_overhead.y_hat_shell, experiment)
        self.nu2 = self.calculate_nu2(analysis_overhead.nu2_hat_shell, experiment)
        
        self.estimator_data = self.calculate_tausigma2hat(analysis_overhead.estimand_df, experiment, inference_estimands=None)
        self.estimator_data.loc[:,"T"] = self.T
        
        # Add true values
        self.add_tau(self.estimator_data, experiment)
        self.nu2_rr = self.calculate_nu2rr(estimand_df_w_tau=self.estimator_data, experiment=experiment)
        self.add_sigma2(self.estimator_data, experiment)
        
    def calculate_yhat(self, y_hat_shell, experiment):
        y_hat = {}
        for key in y_hat_shell.keys():
            y_hat[key] = self.calculate_yhat_single(y_hat_shell[key], experiment)
        return y_hat
    def calculate_yhat_single(self, y_hat_shell_value, experiment):
        a = y_hat_shell_value["a"]
        t = y_hat_shell_value["t"]
        r = y_hat_shell_value["r"]
        
        n_at = experiment.ra_coefs[t]["n_at"][a]
        pi_at_r = experiment.ra_coefs[t]["pi_at_r"][a][r]
        
        Y_t = experiment.observed_data.loc[:,"Y_"+str(t)]
        R_t = experiment.observed_data["R_"+str(t)]
        A_t = experiment.observed_data["A_"+str(t)]
        
        ys_to_sum = experiment.observed_data.loc[((A_t==a) & (R_t==r)), "Y_"+str(t)]
        return (1/(n_at*pi_at_r))*(ys_to_sum.sum())
    
    # Calculate nu2hat values
    # (Need to have instantiated self.y_hat first)
    def calculate_nu2hat(self, nu2_hat_shell, experiment):
        nu2_hat = {}
        for key in nu2_hat_shell.keys():
            nu2_hat[key] = self.calculate_nu2hat_single(nu2_hat_shell[key], experiment)
        return nu2_hat
    def calculate_nu2hat_single(self, nu2_hat_shell_value, experiment):
        a = nu2_hat_shell_value["a"]
        t = nu2_hat_shell_value["t"]
        r = nu2_hat_shell_value["r"]
            
        n_at = experiment.ra_coefs[t]["n_at"][a]
        pi_at_r = experiment.ra_coefs[t]["pi_at_r"][a][r]
        pi_aat_rr = experiment.ra_coefs[t]["pi_aat_rr"][a][a][r][r]
        
        Y_t = experiment.observed_data.loc[:,"Y_"+str(t)]
        Yhat_at = self.y_hat[self.atr_estimator_label("Yhat", a, t, r)]
        R_t = experiment.observed_data["R_"+str(t)]
        A_t = experiment.observed_data["A_"+str(t)]
        
        inflation_factor_1 = (n_at*(pi_at_r**2))/((n_at-1)*pi_aat_rr)
        if experiment.group_simulator.ra_type=="complete_ra":
            inflation_factor_2 = 1/(n_at*pi_at_r)
        
        ys_to_sum = experiment.observed_data.loc[((A_t==a) & (R_t==r)), "Y_"+str(t)]
        
        return inflation_factor_1*(inflation_factor_2*((ys_to_sum**2).sum()) - Yhat_at**2)
    
    # Calculate tauhat values and sigmahat values
    # (Need to have instantiated self.y_hat and self.nu2_hat first)
    
    def calculate_tausigma2hat(self, estimand_df_all, experiment, inference_estimands=None):
        estimator_df = pd.DataFrame()
        for i in estimand_df_all.index:
            row = estimand_df_all.loc[i,:]
            if row["type"]=="tau_at":
                new_row = self.calculate_tausigma2hat_at_single(row, experiment)
                #estimator_df = pd.concat([estimator_df, new_row.to_frame().T], ignore_index=True)
                estimator_df = pd.concat([estimator_df, new_row], ignore_index=True)
        for i in estimand_df_all.index:
            row = estimand_df_all.loc[i,:]
            if row["type"]=="tau_t":
                t = row["t"]
                estimators_at = estimator_df.loc[((estimator_df["t"]==t) & (estimator_df["type"]=="tau_at")), :] 
                if list(np.sort(estimators_at["a"].values))!=list(range(self.H)):
                    ValueError("Did not estimate all necessary components for tau_t at time: "+str(t))
                    
                new_row = row.copy()
                tau_t_hat_r_rr = (estimators_at["estimated_val"] * 
                                  estimators_at["a"].map(experiment.ra_coefs[t]["w_at"])).sum()
                sigma2_a_hat_r_rr = (estimators_at["estimated_var"] * 
                                      (estimators_at["a"].map(experiment.ra_coefs[t]["w_at"])**2)).sum()
                new_cols = {
                    "n":self.n,
                    "estimated_val":tau_t_hat_r_rr,
                    "estimated_var":sigma2_a_hat_r_rr
                }
                new_row = pd.DataFrame([new_row]).assign(**new_cols)
                
                #estimator_df = pd.concat([estimator_df, new_row.to_frame().T], ignore_index=True)
                estimator_df = pd.concat([estimator_df, new_row], ignore_index=True)
                
        for i in estimand_df_all.index:
            row = estimand_df_all.loc[i,:]
            if row["type"]=="tau":
                estimators_t = estimator_df.loc[(estimator_df["type"]=="tau_t"), :] 
                if list(np.sort(estimators_at["t"].values))!=list(range(1,self.T+t)):
                    ValueError("Did not estimate all necessary components for tau.")
                    
                new_row = row.copy()
                new_cols = {
                    "n":self.n,
                    "estimated_val":estimators_t["estimated_val"].mean(),
                    "estimated_var":estimators_t["estimated_var"].mean()
                }
                new_row = pd.DataFrame([new_row]).assign(**new_cols)
                #estimator_df = pd.concat([estimator_df, new_row.to_frame().T], ignore_index=True)
                estimator_df = pd.concat([estimator_df, new_row], ignore_index=True)
        return estimator_df
            
                
    
    def calculate_tausigma2hat_at_single(self, estimand_df_row, experiment):
        new_row = estimand_df_row.copy()
        a = new_row["a"]
        t = new_row["t"]
        r = new_row["r"]
        rr = new_row["rr"]
        y_hat_at_r = self.y_hat[self.atr_estimator_label("Yhat", a, t, r)]
        y_hat_at_rr = self.y_hat[self.atr_estimator_label("Yhat", a, t, rr)]
        nu2_hat_at_r = self.nu2_hat[self.atr_estimator_label("nu2hat", a, t, r)]
        nu2_hat_at_rr = self.nu2_hat[self.atr_estimator_label("nu2hat", a, t, rr)]
        
        tau_at_hat_r_rr = y_hat_at_r - y_hat_at_rr
        if experiment.group_simulator.ra_type=="complete_ra":
            n_at_r = experiment.ra_coefs[t]["n_at_r"][a][r]
            n_at_rr = experiment.ra_coefs[t]["n_at_r"][a][rr]
            sigma2_at_hat_r_rr = (nu2_hat_at_r/n_at_r) + (nu2_hat_at_rr/n_at_rr)
        
        new_cols = {
            "n":experiment.ra_coefs[t]["n_at"][a],
            "estimated_val":tau_at_hat_r_rr,
            "estimated_var":sigma2_at_hat_r_rr
        }
        return pd.DataFrame([new_row]).assign(**new_cols)

    
    
    
    # Add true values to an estimator/estimand dataframe
    def add_tau(self, estimand_df, experiment):
        estimand_df.loc[:, "true_val"] = np.nan
        for i in estimand_df.index:
            if estimand_df.loc[i, "type"]=="tau_at":
                estimand_df.loc[i, "true_val"] = self.calculate_tau_at_single(estimand_df.loc[i,:], experiment)
        for i in estimand_df.index:
            if estimand_df.loc[i, "type"]=="tau_t":
                t = estimand_df.loc[i, "t"]
                estimands_at = estimand_df.loc[((estimand_df["t"]==t) & (estimand_df["type"]=="tau_at")), :] 
                if list(np.sort(estimands_at["a"].values))!=list(range(self.H)):
                    ValueError("Did not estimate all necessary components for tau_t at time: "+str(t))
                estimand_df.loc[i, "true_val"] = (estimands_at["true_val"] * 
                                                  estimands_at["a"].map(experiment.ra_coefs[t]["w_at"])).sum()
        for i in estimand_df.index:
            if estimand_df.loc[i,"type"]=="tau":
                estimands_t = estimand_df.loc[(estimand_df["type"]=="tau_t"), :] 
                if list(np.sort(estimands_t["t"].values))!=list(range(1,self.T+t)):
                    ValueError("Did not estimate all necessary components for tau.")
                estimand_df.loc[i, "true_val"] = estimands_t["true_val"].mean()
                
    def calculate_tau_at_single(self, estimand_df_row, experiment):
        a = estimand_df_row["a"]
        t = estimand_df_row["t"]
        r = estimand_df_row["r"]
        rr = estimand_df_row["rr"]
        tau_at_df = experiment.tau_dfs[t].loc[experiment.tau_dfs[t]["A_"+str(t)]==a]
        
        n_at = experiment.ra_coefs[t]["n_at"][a]
        # Y = Y(r0) + Tau_df[r]
        # Thus, tau_it(r,r') = Y_it(r) - Y_it(r') = Tau_df[r] - Tau_df[r']
        tau_at_r_rr = (1/n_at)*((tau_at_df[r] - tau_at_df[rr]).sum())
        return tau_at_r_rr
    
    
    
    def add_sigma2(self, estimand_df, experiment):
        estimand_df.loc[:, "true_var"] = np.nan
        for i in estimand_df.index:
            if estimand_df.loc[i, "type"]=="tau_at":
                estimand_df.loc[i, "true_var"] = self.calculate_sigma2_at_single(estimand_df.loc[i,:], experiment)
        for i in estimand_df.index:
            if estimand_df.loc[i, "type"]=="tau_t":
                t = estimand_df.loc[i, "t"]
                estimands_at = estimand_df.loc[((estimand_df["t"]==t) & (estimand_df["type"]=="tau_at")), :] 
                if list(np.sort(estimands_at["a"].values))!=list(range(self.H)):
                    ValueError("Did not estimate all necessary components for tau_t at time: "+str(t))
                estimand_df.loc[i, "true_var"] = (estimands_at["true_var"] * 
                                                  (estimands_at["a"].map(experiment.ra_coefs[t]["w_at"])**2)).sum()
        for i in estimand_df.index:
            if estimand_df.loc[i,"type"]=="tau":
                estimands_t = estimand_df.loc[(estimand_df["type"]=="tau_t"), :] 
                if list(np.sort(estimands_t["t"].values))!=list(range(1,self.T+t)):
                    ValueError("Did not estimate all necessary components for tau.")
                estimand_df.loc[i, "true_var"] = estimands_t["true_var"].mean()
    
    def calculate_sigma2_at_single(self, estimand_df_row, experiment):
        a = estimand_df_row["a"]
        t = estimand_df_row["t"]
        r = estimand_df_row["r"]
        rr = estimand_df_row["rr"]
        
        if experiment.group_simulator.ra_type=="complete_ra":
            n_at_r = experiment.ra_coefs[t]["n_at_r"][a][r]
            n_at_rr = experiment.ra_coefs[t]["n_at_r"][a][rr]
            n_at = experiment.ra_coefs[t]["n_at"][a]

            nu2_at_r = self.nu2[self.atr_estimator_label("nu2", a, t, r)]
            nu2_at_rr = self.nu2[self.atr_estimator_label("nu2", a, t, rr)]
            nu2_at_r_rr = self.nu2_rr[self.atr_estimator_label("nu2", a, t, "("+r+")-("+rr+")")]
            sigma2_at_r_rr = (nu2_at_r/n_at_r) + (nu2_at_rr/n_at_rr) - (nu2_at_r_rr/n_at)

        return sigma2_at_r_rr
    
    
    def calculate_ybar(self, y_hat_shell, experiment):
        y_bar = {}
        for key in y_hat_shell.keys():
            new_key = self.atr_estimator_label("Ybar", y_hat_shell[key]["a"], y_hat_shell[key]["t"], y_hat_shell[key]["r"])
            y_bar[new_key] = self.calculate_ybar_single(y_hat_shell[key], experiment)
        return y_bar
    def calculate_ybar_single(self, y_hat_shell_value, experiment):
        a = y_hat_shell_value["a"]
        t = y_hat_shell_value["t"]
        r = y_hat_shell_value["r"]
            
        n_at = experiment.ra_coefs[t]["n_at"][a]
        
        y0_t = experiment.hist.loc[:,"y0_"+str(t)]
        y_t_r = y0_t + experiment.tau_dfs[t][r]
        
        ys_to_sum = y_t_r[experiment.tau_dfs[t]["A_"+str(t)]==a]
        
        return (1/n_at)*(ys_to_sum.sum())
    
    # Calculate nu2 values
    # (Need to have instantiated self.y_bar first)
    def calculate_nu2(self, nu2_hat_shell, experiment):
        nu2 = {}
        for key in nu2_hat_shell.keys():
            new_key = self.atr_estimator_label("nu2", nu2_hat_shell[key]["a"], nu2_hat_shell[key]["t"], nu2_hat_shell[key]["r"])
            nu2[new_key] = self.calculate_nu2_single(nu2_hat_shell[key], experiment)
        return nu2
    def calculate_nu2_single(self, nu2_hat_shell_value, experiment):
        a = nu2_hat_shell_value["a"]
        t = nu2_hat_shell_value["t"]
        r = nu2_hat_shell_value["r"]
            
        n_at = experiment.ra_coefs[t]["n_at"][a]
        
        y0_t = experiment.hist.loc[:,"y0_"+str(t)]
        y_t_r = y0_t + experiment.tau_dfs[t][r]
        
        A_t = experiment.observed_data["A_"+str(t)]
        
        vals_to_sum = (y_t_r - self.y_bar[self.atr_estimator_label("Ybar", a, t, r)])[A_t==a]
        
        return (1/(n_at-1))*((vals_to_sum**2).sum())
    
    def calculate_nu2rr(self, estimand_df_w_tau, experiment):
        nu2_rr = {}
        for i in estimand_df_w_tau.index:
            row = estimand_df_w_tau.loc[i, :]
            if row["type"]=="tau_at":
                a = row["a"]
                t = row["t"]
                r = row["r"]
                rr = row["rr"]
                new_key = self.atr_estimator_label("nu2", a, t, "("+r+")-("+rr+")")
                
                tau_at_rr = row["true_val"]
                A_t = experiment.observed_data["A_"+str(t)]
                n_at = experiment.ra_coefs[t]["n_at"][a]
                
                tau_r_rr = experiment.tau_dfs[t][r] - experiment.tau_dfs[t][rr]
                vals_to_sum = (tau_r_rr - tau_at_rr)[A_t==a]
                
                nu2_rr[new_key] = (1/(n_at-1))*((vals_to_sum**2).sum())
        return nu2_rr
    
    
    
# This class iterates and produces a list of simulation results
class BaseSimulator:
    def __init__(self, analysis_overhead, outcome_simulator, attribute_simulator, group_simulator):
        self.analysis_overhead = analysis_overhead
        self.outcome_simulator = outcome_simulator
        self.attribute_simulator = attribute_simulator
        self.group_simulator = group_simulator
        self.n = self.analysis_overhead.n
        self.T = self.analysis_overhead.T
    def run_sims(self, n_iter):
        results_list = []
        for iteration in range(n_iter):
            new_experiment = ExperimentSimulator(n=self.n, T=self.T, 
                                                 outcome_simulator=self.outcome_simulator, 
                                                 attribute_simulator=self.attribute_simulator, 
                                                 group_simulator=self.group_simulator)
            new_experiment.simulate_experiment()
            iteration_results = ExperimentAnalyzer(self.analysis_overhead, new_experiment).estimator_data
            iteration_results.loc[:,"iteration"] = iteration
            results_list.append(iteration_results)
        estimator_results = pd.concat(results_list, ignore_index = True)
        return estimator_results