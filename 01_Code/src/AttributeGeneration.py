##############################
### AttributeGeneration.py ###
##############################

### PURPOSE: This code defines the AttributeSimulator class, which handles attribute simulation/labeling
### PROGRAMMER: Gabriel Durham (GJD)
### CREATED ON: 29 OCT 2025 
### EDITS: 12 NOV 2025 (GJD) - Added rng argument for reproducibility


import pandas as pd
import numpy as np
from numpy.random import default_rng


class AttributeSimulator:
    def __init__(self, yaml_parms, rng=None):
        if rng is None:
            self.rng = default_rng()
        else:
            self.rng = rng
        self.attribute_type=yaml_parms["attribute_type"]
        if self.attribute_type=="rank":
            self.rank_var=yaml_parms["rank_var"]
            self.quantiles=yaml_parms["quantiles"]
            self.H=len(self.quantiles)+1
        
            
    def simulate_attributes_rank(self, hist, t):
        y0_prev = hist.loc[:,"y0_"+str(t-1)]
        cuts = np.quantile(y0_prev, self.quantiles)
        new_atts = np.digitize(y0_prev, cuts, right=True)
        return new_atts
    def simulate_attributes(self, hist, t):
        if self.attribute_type=="rank":
            new_atts = self.simulate_attributes_rank(hist, t)
        return new_atts