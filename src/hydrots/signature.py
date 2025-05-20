
import pandas as pd 
import numpy as np
import itertools

from abc import ABC, abstractmethod
from typing import Optional, Dict, List

def compute_rbi(group):
    """
    Compute Richards-Baker Index for a single group (e.g., one year of streamflow data).
    
    Parameters:
        group (pd.DataFrame): DataFrame containing 'Q' column of streamflow values.
    
    Returns:
        float: Richards-Baker Index (RBI) for this group.
    """
    group = group.sort_index()  # Ensure time order
    q_diff = group['Q'].diff().abs()
    total_q = group['Q'].sum()
    if total_q == 0 or np.isnan(total_q):
        return np.nan
    return q_diff.sum() / total_q

def compute_cv(group): 
    group = group.sort_index()
    q_std = group['Q'].std()
    q_mean = group['Q'].mean()
    if q_mean == 0:
        return np.nan 
    return q_std / q_mean

def compute_dvia(group): 
    pass 

def compute_dvic(group): 
    pass

# FIXME this currently makes assumption that timeseries has daily resolution
class TSSignature:

    _custom_summary_functions = {}

    def __init__(self, ts: "HydroTS", use_complete_years=True): 
        self.ts = ts
        self.use_complete_years = use_complete_years 

    def coefficient_of_variation(self, by_year=False): 
        df = self.ts.data.copy()
        df = df[df['water_year'].isin(self.ts.valid_years)]
        if by_year: 
            cv = df.groupby('water_year')[['Q']].apply(compute_cv)
        else:
            cv = compute_cv(df)
        return cv

    def richards_baker_index(self, by_year=False): 
        df = self.ts.data.copy()
        df = df[df['water_year'].isin(self.ts.valid_years)]
        if by_year: 
            rbi = df.groupby('water_year')[['Q']].apply(compute_rbi)
        else:
            rbi = compute_rbi(df)
        return rbi
    
    def discharge_variability_index(self): 
        pass

    def cumulative_discharge_variability_index(self):
        pass