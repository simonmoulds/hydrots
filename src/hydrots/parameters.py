
import numpy as np 
import pandas as pd

from typing import Optional, Dict, List

class DynamicModelParameters:

    def __init__(self, params: pd.DataFrame):
        self.params = params.copy()

    @classmethod
    def from_dict(cls, param_dict: Dict[str, Dict[str, float]]):
        return cls(pd.DataFrame.from_dict(param_dict, orient='index'))

    def _check_parameter_values(self, values): 
        # Check length
        if len(values) != len(self.param_names):
            raise ValueError(f"This model requires {len(self.param_names)} parameters but {len(values)} were supplied.")

        # Check bounds
        values = [
            max(l, min(p, u))
            for p, l, u in zip(values, self.param_lower, self.param_upper)
        ]
        return values

    def set_initial_values(self, initial_values: Optional[List[float]] = None):
        if initial_values is None:
            initial_values = [(l + u) / 2 for l, u in zip(self.param_lower, self.param_upper)]
        else:
            initial_values = self._check_parameter_values(initial_values)
        
        return initial_values

    def update(self, values): 
        values = self._check_parameter_values(values)
        self.params['value'] = values

    def update_lower_bound(self, values): 
        self.params["lower_bound"] = values 

    def update_upper_bound(self, values): 
        self.params["upper_bound"] = values 

    def get_initial_value(self):
        return self.params["initial_value"].values

    def get_bounds(self):
        return list(zip(self.params["lower_bound"], self.params["upper_bound"]))

    def __len__(self): 
        return len(self.params)

    @property 
    def values(self): 
        return self.params['value'].values 

    @property 
    def shape(self): 
        return self.params.shape 


class Ihacres7p1sParams(DynamicModelParameters): 
    
    def __init__(self, initial_values: Optional[List[float]] = None): 
        self.param_names = ["lp", "d", "p", "alpha", "tau_q", "tau_s", "tau_d"]
        self.param_lower = [1, 1, 0, 0, 1, 1, 0]
        self.param_upper = [2000, 2000, 10, 1, 700, 700, 119]
        initial_values = self.set_initial_values(initial_values)
        params = pd.DataFrame.from_dict({'parameter_name': self.param_names, 'lower_bound': self.param_lower, 'upper_bound': self.param_upper, 'initial_value': initial_values, 'value': initial_values})
        super().__init__(params)


class Hymod5p5sParams(DynamicModelParameters): 
    
    def __init__(self, initial_values: Optional[List[float]] = None): 
        self.param_names = ["smax", "b", "a", "kf", "ks"] 
        self.param_lower = [1, 0, 0, 0, 0]
        self.param_upper = [2000, 10, 1, 1, 1]
        initial_values = self.set_initial_values(initial_values)
        params = pd.DataFrame.from_dict({'parameter_name': self.param_names, 'lower_bound': self.param_lower, 'upper_bound': self.param_upper, 'initial_value': initial_values, 'value': initial_values})
        super().__init__(params)

