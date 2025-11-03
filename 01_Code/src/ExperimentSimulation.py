###############################
### ExperimentSimulation.py ###
###############################

### PURPOSE: This code defines the ExperimentSimulator class, which handles attribute simulation/labeling
### PROGRAMMER: Gabriel Durham (GJD)
### CREATED ON: 29 OCT 2025 
### EDITS: 3 NOV 2025 (GJD) - Set self.hist = self.hist.copy() every 10th time step to defrag history

import pandas as pd
import numpy as np



class ExperimentSimulator:
    def __init__(self, n, T, outcome_simulator, attribute_simulator, group_simulator):
        self.n = n
        self.T = T
        
        self.outcome_simulator = outcome_simulator
        self.attribute_simulator = attribute_simulator
        self.group_simulator = group_simulator
        
        self.hist = pd.DataFrame(index=range(self.n))
        self.tau_dfs = {}
        self.ra_coefs = {}
    
    def simulate_time_step(self, t):
        # Simulate attributes
        self.hist.loc[:, "A_"+str(t)] = self.attribute_simulator.simulate_attributes(self.hist, t)
        # Simulate baseline outcome
        self.hist.loc[:,"y0_"+str(t)] = self.outcome_simulator.simulate_y0(self.hist, t)
        
        #new_cols = {
        #    "A_"+str(t):self.attribute_simulator.simulate_attributes(self.hist, t),
        #    "y0_"+str(t):self.outcome_simulator.simulate_y0(self.hist, t)
        #}
        #self.hist = pd.concat([self.hist, pd.DataFrame(new_cols)], axis=1)
        
        # Simulate causal effects
        # Attach attributes as well for later analyses
        self.tau_dfs[t] = pd.concat([self.hist.loc[:, "A_"+str(t)], 
                                     self.outcome_simulator.simulate_tau(self.hist, t)], 
                                    axis=1)
        # Construct groups
        t_group_assignment_information = self.group_simulator.assign_groups(self.hist, t)
        # Store randomization coefficients
        self.ra_coefs[t] = t_group_assignment_information["ra_coefs"]
        # Store peer compositions
        self.hist.loc[:, "R_"+str(t)] = t_group_assignment_information["peer_compositions"]
        # Calculate outcomes
        self.hist.loc[:, "Y_"+str(t)] = self.outcome_simulator.simulate_y(self.hist, self.tau_dfs[t], t)
        
        #new_cols = {
        #    "R_"+str(t):t_group_assignment_information["peer_compositions"],
        #    "Y_"+str(t):self.outcome_simulator.simulate_y(self.hist, self.tau_dfs[t], t)
        #}
        #self.hist = pd.concat([self.hist, pd.DataFrame(new_cols)], axis=1)
        
        observed_data = pd.concat([self.hist.loc[:, "A_"+str(t)], 
                                   self.hist.loc[:, "R_"+str(t)], 
                                   self.hist.loc[:, "Y_"+str(t)]],
                                 axis=1)
        return observed_data
    
    def simulate_experiment(self, verbose=False, return_observed_data=False):
        # Raise flag if it looks like you're simulating an experiment on top of another experiment
        if verbose:
            if self.hist.shape[1]!=0:
                print("Warning: Starting an experiment with previously observed history. This may lead to joining output from multiple experiments.")
        # Simulate y0_0 - Used for some attribute initialization schemes and autoregressive outcome construction (for A_1, Y0_1)
        self.hist.loc[:, "y0_0"] = self.outcome_simulator.simulate_y0(self.hist, t=0)
        observed_data = pd.DataFrame(index=range(self.n))
        for t in range(1,self.T+1):
            observed_data = pd.concat([observed_data, self.simulate_time_step(t)], axis=1)
            if t%10==0:
                self.hist = self.hist.copy()
        self.observed_data = observed_data
        if return_observed_data:
            return observed_data