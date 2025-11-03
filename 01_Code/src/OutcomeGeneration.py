############################
### OutcomeGeneration.py ###
############################

### PURPOSE: This code defines the OutcomeSimulator class, which handles outcome and causal effect simulation
### PROGRAMMER: Gabriel Durham (GJD)
### CREATED ON: 29 OCT 2025 


import pandas as pd
import numpy as np


class OutcomeSimulator:
    def __init__(self, yaml_parms):
        self.y0_parms=yaml_parms["y0"]
        self.tau_parms=yaml_parms["tau"]
        self.y0_transition_type=self.y0_parms["outcome_transition_type"]
        self.effect_spec_type=self.tau_parms["effect_spec_type"]
        # Store values for y0 (baseline outcome) transition
        if self.y0_transition_type=="ar1":
            self.y0_mean=self.y0_parms["mean"]
            self.y0_rho=self.y0_parms["rho"]
            self.error_structure=self.y0_parms["error_dist"]
        
        # Create map for tau (causal effect) construction
        if self.effect_spec_type=="enumerate":
            self.simulate_tau=self.simulate_tau_enumerate()
    
    # Simulate Y0 (baseline outcomes)
    def simulate_y0_noise(self, n):
        if self.error_structure["type"]=="normal":
            noise=self.error_structure["sd"]*np.random.randn(n)
        return(noise)
    def simulate_y0_ar1(self, hist, t):
        if t==0:
            y0_t=self.y0_mean + self.simulate_y0_noise(n=len(hist))
        else:
            y0_tm1=hist.loc[:,"y0_"+str(t-1)]
            y0_t=(1-self.y0_rho)*self.y0_mean + self.y0_rho*y0_tm1 + self.simulate_y0_noise(n=len(hist))
        return(y0_t)
    def simulate_y0(self, hist, t):
        if self.y0_transition_type=="ar1":
            y0_t = self.simulate_y0_ar1(hist, t)
        return y0_t
    
    
    # Simulate tau (causal effects)
    def simulate_tau_enumerate(self):
        def simulate_tau(hist, t):
            tau_df=pd.DataFrame(index=hist.index)
            for r in self.tau_parms["effects"].keys():
                r_parms=self.tau_parms["effects"][r]
                # Baseline composition - No effect
                if r_parms["type"]=="baseline":
                    tau_df.loc[:,r]=0
                # Constant effect
                elif r_parms["type"]=="constant":
                    tau_df.loc[:,r]=r_parms["tau"]
                # Constant effect, moderated by attribute
                elif r_parms["type"]=="constant_mod_a":
                    A_t=hist.loc[:,"A_"+str(t)]
                    tau_map=r_parms["tau_map"]
                    # tau_map maps a's to taus
                    tau_df.loc[:,r]=np.array([tau_map[a] for a in A_t])
            return(tau_df)
        return(simulate_tau)
    
    # Simulate Y (observed outcome)
    def simulate_y(self, hist, tau_df, t):
        y_0=hist.loc[:, "y0_"+str(t)]
        
        
        # Get the IDs we want (columns correspond to the treatment unit i received)
        col_ids = tau_df.columns.get_indexer(hist.loc[:, "R_"+str(t)])
        #tau_arr = tau_df.to_numpy()
        #col_ids = tau_arr.columns.get_indexer(hist.loc[:, "R_"+str(t)])
        
        # Vectorized row selection
        #tau_it_observed = tau_arr[np.arange(len(hist)), col_ids]
        tau_arr = tau_df.to_numpy()
        tau_it_observed = tau_arr[np.arange(len(hist)), col_ids]
        
        y = y_0 + tau_it_observed
        return(y)