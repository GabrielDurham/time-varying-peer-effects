##########################
### GroupAssignment.py ###
##########################

### PURPOSE: This code defines the GroupAssignmentSimulator class, which handles randomized group formation 
### PROGRAMMER: Gabriel Durham (GJD)
### CREATED ON: 29 OCT 2025 
### EDITS: 12 NOV 2025 (GJD) - Added rng argument for reproducibility

import pandas as pd
import numpy as np
import math
from itertools import combinations
#import random
from numpy.random import default_rng


class GroupAssignmentSimulator:
    def __init__(self, yaml_parms, H, rng=None):
        if rng is None:
            self.rng = default_rng()
        else:
            self.rng = rng
        self.m=yaml_parms["m"]
        self.H=H
        self.ra_type=yaml_parms["ra_type"]
        if self.ra_type=="complete_ra":
            self.balanced=yaml_parms["balanced"]
            if self.balanced:
                # If we're doing (balanced) complete RA, unless we have the "right" amount of
                # units of each attribute, we won't be able to make each group the exact same
                # number of times. So, we'll have some slight differences in group realizations.
                # We decide this by enumerating all the groups (g=1,...,G), then we randomize to
                # create a group of structure g=1, g=2, ..., g=G, g=1, .... Eventually, we get
                # to a group we can't form, we then skip it and keep forming groups until everyone
                # is grouped up. "preferred_groups" allows the simulation designer (via config)
                # to put some of the groups up at the top of the list, helping "nudge" them to get
                # more realizations
                if "preferred_groups" in yaml_parms.keys():
                    self.preferred_groups=yaml_parms["preferred_groups"]

    # Assign groups according to specified mechanism
    # Right now, have only coded up complete ra
    def assign_groups(self, hist, t):
        if self.ra_type=="complete_ra":
            return(self.complete_ra(hist, t))
    
    # Complete randomization for completely balanced groups
    # Requires making every type of group (and the same number of each type)
    def complete_ra(self, hist, t):
        # This was code I used when I wanted to limit to cases where we had exactly the right number for equal numbers of each type of group
        # Check to make sure there are the right number of attributes
        # Should have same number of folks with each attribute, should be divisible by minimum number for one type of each group
        #if np.var(A_counts)!=0:
        #    error_message="For balanced complete randomization, all attributes must have the same number of individuals. Error occurred at time " + str(t) + "."
        #    raise ValueError(error_message)
        #if A_counts[0]%self.compute_attribute_requirements_complete_balanced()!=0:
        #    error_message="Incorrect group composition for balanced complete randomization."
        #    error_message+="For specified attribute/group structure (m="+str(m)+", H="+str(H)+"),"
        #    error_message+="must have a multiple of "+str(self.compute_attribute_requirements_complete_balanced())+ "units at each time."
        #    error_message+= "Error occurred at time " + str(t) + "."
        #    raise ValueError(error_message)
        if self.balanced:
            group_types_to_form = self.determine_group_types_balanced_ra(hist, t)
        group_assignments = self.complete_ra_group_assignment(hist, t, group_structures=group_types_to_form)
        peer_compositions = self.peer_composition_calculation(hist, t, group_assignments=group_assignments)
        ra_coefs = self.calculate_ra_coefficient_complete(group_structures=group_types_to_form)
        
        Output = {
            #"group_structures":group_types_to_form,
            "group_assignments":group_assignments,
            "peer_compositions":peer_compositions,
            "ra_coefs":ra_coefs
        }
        return Output
        
        
    
    # Grab the number of people (of a single attribute) needed to form one of every type of group
    def compute_attribute_requirements_complete_balanced(self):
        # Stars and bars => (m+H-1 / H-1) total ways of making groups of size m with folks with attributes 1,...,H
        # m*(m+H-1 / H-1) total people in those groups
        # (m/H)*(m+H-1 / H-1) people with each attribute (by symmetry)
        # (m/H)*(m+H-1 / H-1) = (m+H-1 / H) by basic algebra
        return(math.comb(self.m+self.H-1, self.H))
    
    # Create group structures to fill
    def enumerate_all_groups(self, m_override=None):
        m = self.m if m_override is None else m_override
        group_types = []
        total_slots = m + self.H - 1
        # We want to iterate over all the ways to chose H-1 distinct numbers from 1, 2, ..., m + self.H - 1
        # Each "bars" will be a tuple of size self.H - 1. (No two different "bars" will have the same elements)
        for bars in combinations(range(total_slots), self.H - 1):
            # We turn "bars" into a group structure
            # "bars" represents the location of where we put the bar (lining up all folks in the group 1,..., m
            # Everyone to the left of the first bar has A=0, everyone in between the first and second bars have A=1,...
            counts = []
            prev = -1
            for b in bars:
                counts.append(b - prev - 1)     # stars before this bar
                prev = b
            counts.append(total_slots - prev - 1)  # stars after the last bar
            group_types.append(tuple(counts))
        return group_types
    
    # Determine the group structures to be assigned for balanced complete RA given the current attribute makeup
    def determine_group_types_balanced_ra(self, hist, t):
        A_t = hist.loc[:, "A_"+str(t)]
        A_counts = A_t.value_counts().sort_index()
        running_A_counts = A_counts.copy()
        all_possible_groups = self.enumerate_all_groups()
        groups_to_form = []
        
        # Before structuring the groups, check to make sure the attributes are consistent with the simulator's specs
        if len(A_counts)!=self.H or len(hist)%self.m!=0:
            error_message="History does not match GroupAssignmentSimulator() specs. Possible attribute or sample size mismatch. Error occurred at time " + str(t) + "."
            raise ValueError(error_message)
        
        # Loop through the groups and keep assigning them when there's enough folks to do so. Stop when we've allocated all folks
        while running_A_counts.sum()>0:
            for group in all_possible_groups:
                form_group=True
                # Don't form group if you need more people from an attribute than there are left to randomize
                for a in range(self.H):
                    if group[a]>running_A_counts[a]:
                        form_group=False
                        break
                if form_group:
                    groups_to_form.append(group)
                    # If you form the group, subtract the group composition from the eligible folks to randomize
                    for a in range(self.H):
                        running_A_counts[a]-=group[a]
                # If you can't form the group, just check to see that you haven't run out of people
                # If you have, break out of the for loop (this may save a minor amount of run time)
                elif running_A_counts.sum()==0:
                    break
        return groups_to_form
    
    
    # Implement complete randomization given desired group structures
    def complete_ra_group_assignment(self, hist, t, group_structures):
        reshuffled_indices_by_a = {}
        # Reshuffle the indices for each attribute randomly then just fill in the groups
        for a in range(self.H):
            indices_a = list(hist.index[hist["A_"+str(t)]==a])
            #reshuffled_indices_by_a[a] = random.sample(indices_a, k=len(indices_a))
            reshuffled_indices_by_a[a] = list(self.rng.permutation(indices_a))
        groups_formed = []
        for group_structure in group_structures:
            group_ids = []
            # Going through each attribute to see how many we need
            for a, n_needed in enumerate(group_structure):
                if n_needed > 0:
                    # Take the first n_needed elements and remove them from the bucket
                    ids_to_add = reshuffled_indices_by_a[a][:n_needed]
                    del reshuffled_indices_by_a[a][:n_needed]
                    group_ids.extend(np.sort(ids_to_add))
            groups_formed.append(group_ids)
        return groups_formed
    
    def peer_composition_calculation(self, hist, t, group_assignments):
        group_compositions=[]
        for group in group_assignments:
            levels_in_group=hist.loc[group, "A_"+str(t)].to_numpy()
            group_compositions.append((np.bincount(levels_in_group, minlength=self.H)))
        
        peer_attributes=pd.DataFrame(index=hist.index)
        peer_attributes.loc[:, "R_"+str(t)] = ""
        # Loop through groups and identify peers attributes of each member
        for group, comp in zip(group_assignments, group_compositions):
            # group is the list of IDs in this group
            # comp is the corresponding composition tuple (e.g., (1,2,0))
            for i in group:
                a_it = hist.loc[i, "A_"+str(t)]
                peer_comp = comp.copy()
                peer_comp[a_it] -= 1
                # Create it as a string for easier access
                peer_attributes_string = str(peer_comp[0])
                for peer_at in range(1,self.H):
                    peer_attributes_string+="-"+str(peer_comp[peer_at])
                peer_attributes.loc[i, "R_"+str(t)] = peer_attributes_string
        return(peer_attributes)
    
    def calculate_ra_coefficient_complete(self, group_structures):
        n_at = {}
        w_at = {}
        n = self.m*len(group_structures)
        for a in range(self.H):
            n_at[a] = sum(group[a] for group in group_structures)
            w_at[a] = n_at[a]/n
        
        peer_comp_tuples = self.enumerate_all_groups(m_override=self.m-1)
        # Create text labels for r
        r_labels = []
        for r in peer_comp_tuples:
            r_label = str(r[0])
            for a in range(1,self.H):
                r_label += "-"+str(r[a])
            r_labels.append(r_label)
        
        n_at_r = {}
        for a in range(self.H):
            n_at_r[a] = {}
            for r, r_label in zip(peer_comp_tuples, r_labels):
                # A unit with A=a having peer composition r will be in a group of structure r with one more a unit
                corr_group_structure = tuple(np.array(r) + [1 if i == a else 0 for i in range(self.H)])
                n_valid_groups = group_structures.count(tuple(corr_group_structure))
                # For each group of that type, there are group_structure[a] units with a receiving the given r
                n_at_r[a][r_label] = corr_group_structure[a]*n_valid_groups
        
        pi_at_r = {}
        pi_aat_rr = {}
        for a in range(self.H):
            pi_at_r[a] = {}
            for r in r_labels:
                pi_at_r[a][r] = n_at_r[a][r]/n_at[a]
            
            pi_aat_rr[a] = {}
            for aa in range(self.H):
                pi_aat_rr[a][aa] = {}
                for r in r_labels:
                    pi_aat_rr[a][aa][r] = {}
                    for rr in r_labels:
                        if a!=aa:
                            pi_aat_rr[a][aa][r][rr] = (n_at_r[a][r]*n_at_r[aa][rr])/(n_at[a]*n_at[aa])
                        elif r!=rr:
                            pi_aat_rr[a][aa][r][rr] = (n_at_r[a][r]*n_at_r[a][rr])/(n_at[a]*(n_at[a]-1))
                        else:
                            pi_aat_rr[a][aa][r][rr] = (n_at_r[a][r]*(n_at_r[a][r]-1))/(n_at[a]*(n_at[a]-1))
        Output = {
            "n_at":n_at,
            "w_at":w_at,
            "n_at_r":n_at_r,
            "pi_at_r":pi_at_r,
            "pi_aat_rr":pi_aat_rr,
        }
        return Output
