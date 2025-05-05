
import numpy as np
import pandas as pd 
import sceua 
import cma

from scipy.optimize import root, fmin, minimize, newton

from bmipy import Bmi
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from numpy.typing import NDArray
from dataclasses import dataclass, field

from hydrots.timeseries import HydroTS
from hydrots.parameters import Ihacres7p1sParams, Hymod5p5sParams
from hydrots.newtonraphson import newton_raphson_solver
from hydrots.objectivefuns import of_KGE
from hydrots import unithydrographs as uh
from hydrots import fluxes 

FloatArray = NDArray[np.floating[Any]]

@dataclass
class SolverData:
    resnorm: np.ndarray
    solver: pd.Categorical
    iter: np.ndarray

    @classmethod
    def create(cls, t_end: int) -> "SolverData":
        """Factory method to create empty SolverData."""
        return cls(
            resnorm=np.zeros(t_end),
            solver=pd.Categorical(np.zeros(t_end, dtype=int)),
            iter=np.zeros(t_end)
        )

@dataclass
class SolverOptions: 
    resnorm_maxiter: int
    resnorm_tolerance: float


class TSDynamicModel: #(Bmi):

    def __init__(self):
        pass 

    def set_simulation_time(self, 
                            start_time: Optional[pd.Timestamp] = None,
                            end_time: Optional[pd.Timestamp] = None):

        if not start_time:
            start_time = self.forcing.index.min()

        if not end_time: 
            end_time = self.forcing.index.max()

        self.simulation_time = self.forcing.loc[start_time:end_time].index.values
    
    def initialize(self):
        """Initializes stores, fluxes, and solver tracking arrays."""

        self.store_min: np.ndarray = np.zeros(self.num_stores)
        self.store_max: np.ndarray = np.full(self.num_stores, np.inf)

        n_time = self.n_time 
        self.stores = np.zeros((n_time, self.num_stores)) #, columns=self.store_names, index=self.simulation_time)
        self.fluxes = np.zeros((n_time, self.num_fluxes)) #, columns=self.flux_names, index=self.simulation_time)
        self.output = np.zeros((n_time, self.num_outputs)) #, columns=self.output_names, index=self.simulation_time)

        self.solver_data_cls = SolverData
        self.solver_data = self.solver_data_cls.create(n_time)

        self.timestep = 0
        self.time = self.simulation_time[self.timestep]

    def update(self): 
        if self.timestep < (self.n_time - 1):
            self.timestep += 1 
            self.time = self.simulation_time[self.timestep]
        # self.timestep += 1 
        # self.time = self.simulation_time[self.timestep]

    def reset(self): 
        pass

    # @staticmethod
    def ODE_approx_IE(self, S: np.ndarray, Sold: np.ndarray) -> np.ndarray: #, S0: np.ndarray, model_fun: function, delta_t: float) -> np.ndarray:
        """ODE approximation with Implicit Euler time-stepping scheme.

        Args:
            S (np.ndarray): Current store state (any shape, will be flattened internally).

        Returns:
            np.ndarray: Error/residual of the Implicit Euler step.
        """
        S = S.flatten()  # ensure S is 1D
        delta_S, _ = self.model_fun(S)  # assumes model_fun(S) -> np.ndarray
        # if self.timestep == 0:
        #     Sold = self.initial_conditions.flatten()
        # else:
        #     Sold = self.stores[(self.timestep-1)]
        err = (S - Sold) / self.delta_t - delta_S
        return err

    def solve_stores(self, Sold): 
        """Solve the stores ODEs for the current time step."""
 
        # Dynamic tolerance
        resnorm_tolerance = self.solver_opts.resnorm_tolerance * min(min(abs(Sold)) + 1e-5, 1)

        # Prepare arrays to store solutions
        Snew_v = np.zeros((3, self.num_stores))
        resnorm_v = np.full(3, np.inf)
        iter_v = np.ones(3)

        # This works too
        # tmp_Snew = newton(self.ODE_approx_IE, Sold, tol=0.1, maxiter=6)
        # tmp_resnorm = 0.001
        tmp_Snew, tmp_fval, _ = newton_raphson_solver(self.ODE_approx_IE, x=Sold, x0=Sold)
        # solver = NewtonRaphsonSolver(self.ODE_approx_IE, Sold)
        # tmp_Snew, tmp_fval, _ = solver.solve()
        tmp_resnorm = np.sum(tmp_fval**2)

        Snew_v[0, :] = tmp_Snew
        resnorm_v[0] = tmp_resnorm
        # iter_v[0] = tmp_iter

        # Matlab function:
        # [tmp_Snew, tmp_fval] = ...
        #             NewtonRaphson(@obj.ODE_approx_IE,...
        #                             Sold,...
        #                             solver_opts.NewtonRaphson);

        # TODO Implement these methods [look up rerunSolver(...) to see how to initialize]
        # # 2 - If needed, try fsolve equivalent (hybr again or 'lm')
        # if tmp_resnorm > resnorm_tolerance:
        #     tmp_Snew, tmp_fval, success, tmp_iter = self.solve_with_method(
        #         obj.ODE_approx_IE, tmp_Snew, method=self.methods[1], tol=solver_opts['fsolve']['tol']
        #     )
        #     tmp_resnorm = np.sum(tmp_fval**2)
        #     Snew_v[1, :] = tmp_Snew
        #     resnorm_v[1] = tmp_resnorm
        #     iter_v[1] = tmp_iter
        #     # 3 - If still needed, try lsqnonlin equivalent (lm again)
        #     if tmp_resnorm > resnorm_tolerance:
        #         tmp_Snew, tmp_fval, success, tmp_iter = self.solve_with_method(
        #             obj.ODE_approx_IE, tmp_Snew, method=self.methods[2], tol=solver_opts['lsqnonlin']['tol']
        #         )
        #         tmp_resnorm = np.sum(tmp_fval**2)
        #         Snew_v[2, :] = tmp_Snew
        #         resnorm_v[2] = tmp_resnorm
        #         iter_v[2] = tmp_iter

        # 4 - Pick the best solution
        solver_id = np.argmin(resnorm_v)
        Snew = Snew_v[solver_id, :]
        resnorm = resnorm_v[solver_id]
        # iter_ = iter_v[solver_id]

        # return Snew, resnorm, solver, iter_
        return Snew, resnorm #, None, #iter_

    def run(self, 
            start_time: Optional[pd.Timestamp] = None,
            end_time: Optional[pd.Timestamp] = None,
            params = None, 
            initial_conditions = None): 

        self.set_simulation_time(start_time, end_time)

        if params is not None: 
            self.params.update(params)

        if initial_conditions is not None:
            self.initial_conditions = np.array(initial_conditions).flatten()

        self.initialize()

        for t in range(self.n_time): 
            if t == 0: 
                Sold = self.initial_conditions
            else: 
                Sold = self.stores[(self.timestep-1)]

            Snew, resnorm = self.solve_stores(Sold)
            dS, f = self.model_fun(Snew)

            self.fluxes[t] = f * self.delta_t
            self.stores[t] = Sold + dS * self.delta_t
            
            # self.solver_data.resnorm[t] = resnorm
            # self.solver_data.solver[t] = solver
            # self.solver_data.iter[t] = iter

            self.step()
            self.update()

        self.status = 1

    # def loss_function(self, objective_function, **kwargs): 
    #     self.run()
    #     Qsim = self.output['Q'].values
    #     Qobs = self.forcing['Q'].values
    #     return objective_function(Qsim, Qobs, **kwargs)

    def calibrate(self,
                  warmup_start_time,
                  warmup_end_time,
                  cal_start_time, 
                  cal_end_time, 
                  method = 'SCE-UA',
                  inverse_flag = True,
                  params = None,
                  initial_conditions = None,
                  **kwargs):

        # FIXME
        objective_function = of_KGE

        # FIXME - no need to run the warmup period multiple times 
        # Just run it once and save the store values
        def fitness_fun(params: FloatArray, *args): 
            # Unpack arguments
            warmup_start_time, warmup_end_time, cal_start_time, cal_end_time = args

            # First run the warmup period, use this to generate initial storage values
            self.run(warmup_start_time, warmup_end_time, params)
            store_ini = self.stores[-1] # Select the store values on the last warmup time
            self.run(cal_start_time, cal_end_time, params, store_ini)

            Qsim = self.output[:,0]
            Qobs = self.forcing.loc[self.simulation_time]['Q'].values 
            of = objective_function(Qobs, Qsim) #, **kwargs)
            of = (-1) ** inverse_flag * of 
            return of 

        if params is not None: 
            initial = np.array(params).flatten()
        else:
            initial = self.params.get_initial_value() 

        if initial_conditions is not None: 
            self.initial_conditions = np.array(initial_conditions).flatten()

        bounds = self.params.get_bounds()

        # # Method = "SCE-UA"
        # result = sceua.minimize(
        #     fitness_fun, 
        #     bounds, 
        #     args=(warmup_start_time, warmup_end_time, cal_start_time, cal_end_time)
        # ) # FIXME - any way to supply initial guess?

        # Method = CMA-ES
        sigma = min(np.array([0.3 * (high - low) for (low, high) in bounds]))
        bounds = self.params.get_bounds() 
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        result = cma.fmin(
            fitness_fun,
            initial,
            sigma,
            options={'bounds': [lower_bounds, upper_bounds]},
            args=(warmup_start_time, warmup_end_time, cal_start_time, cal_end_time)
        )
        self.calibration_result = result 

        # bounds = self.params.get_bounds()
        # result = minimize(
        #     lambda x: fitness_fun(x, warmup_start_time, warmup_end_time, cal_start_time, cal_end_time), #, **kwargs),
        #     initial,
        #     bounds=bounds
        # )
        # if result.success:
        #     self.params.update(result.x)
        # else:
        #     raise RuntimeError(f"Calibration failed: {result.message}")
        # # TODO return objective_function value 

    @property 
    def n_time(self):
        return len(self.simulation_time)

    @abstractmethod
    def init(self): 
        pass 

    @abstractmethod
    def model_fun(self): 
        pass 

    @abstractmethod 
    def step(self): 
        pass 

class Hymod5p5s(TSDynamicModel): 

    def __init__(self, 
                 params: Ihacres7p1sParams, 
                 initial_conditions: List, 
                 forcing: pd.DataFrame, 
                 delta_t: float, 
                 solver_opts: SolverOptions): 

        self.params = params 
        self.initial_conditions = initial_conditions
        self.forcing = forcing
        self.delta_t = delta_t
        self.solver_opts = solver_opts

        self.store_names = ["S1", "S2", "S3", "S4", "S5"]
        self.flux_names = ["ea", "pe", "pf", "ps", "qf1", "qf2", "qf3", "qs"]
        self.output_names = ["Q", "Ea"]

        self.num_params = len(self.params)
        self.num_stores = len(self.store_names)
        self.num_fluxes = len(self.flux_names)
        self.num_outputs = len(self.output_names)
        self.jacob_pattern = np.array([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1]
        ])

        self.initial_conditions = np.zeros(self.num_stores)

    def initialize(self): 
        super().initialize()

    def model_fun(self, S): 

        smax = self.params.values[0]
        b = self.params.values[1]
        a = self.params.values[2]
        kf = self.params.values[3]
        ks = self.params.values[4]

        S1 = S[0] 
        S2 = S[1]
        S3 = S[2]
        S4 = S[3]
        S5 = S[4]

        forcing_in = self.forcing.loc[self.time]
        P = forcing_in['P']
        Ep = forcing_in['Ep']
        T = forcing_in['T'] 

        # Fluxes functions
        flux_ea = fluxes.evap_7(S1, smax, Ep, self.delta_t)
        flux_pe = fluxes.saturation_2(S1, smax, b, P)
        flux_pf = fluxes.split_1(a, flux_pe)
        flux_ps = fluxes.split_1(1-a, flux_pe)
        flux_qf1 = fluxes.baseflow_1(kf, S2)
        flux_qf2 = fluxes.baseflow_1(kf, S3)
        flux_qf3 = fluxes.baseflow_1(kf, S4)
        flux_qs = fluxes.baseflow_1(ks, S5)

        # Stores 
        dS1 = P - flux_ea - flux_pe 
        dS2 = flux_pf - flux_qf1 
        dS3 = flux_qf1 - flux_qf2 
        dS4 = flux_qf2 - flux_qf3 
        dS5 = flux_ps - flux_qs

        # Outputs 
        dS = np.array([dS1, dS2, dS3, dS4, dS5]).flatten()
        flux = np.array([flux_ea, flux_pe, flux_pf, flux_ps, flux_qf1, flux_qf2, flux_qf3, flux_qs]).flatten()
        return dS, flux

    def step(self):
        self.output[self.timestep, 0] = np.sum(self.fluxes[self.timestep][[6,7]])
        self.output[self.timestep, 1] = self.fluxes[self.timestep][0]


# class Ihacres7p1s(TSDynamicModel): 

#     def __init__(self, params: Ihacres7p1sParams, initial_conditions: List, forcing: HydroTS, delta_t: float):

#         self.params = params 
#         self.initial_conditions = initial_conditions
#         self.forcing = forcing
#         self.delta_t = delta_t

#         self.store_names = ["S1"]
#         self.flux_names = ["Ea", "u", "uq", "us", "xq", "xs", "Qt"] 

#         self.num_stores = len(self.store_names)
#         self.num_fluxes = len(self.flux_names)
#         self.num_params = len(self.params)
#         self.jacob_pattern = [1]

#         # FIXME - probably a better way of doing this
#         self.flux_groups = {
#             'Ea': 0, # Index/indices of fluxes to add to AET (adjusted for zero-indexing)
#             'Q': 6 # Index/indices of fluxes to add to streamflow (adjusted for zero-indexing)
#         }
#         self.store_signs = -1 # Signs to give to stores (-1 is a deficit store), only needed for water balance
#         super().__init__()

#     def init(self): 
#         tau_q = self.params.values[4]
#         tau_s = self.params.values[5]
#         tau_d = self.params.values[6]
        
#         # Initialise the unit hydrographs and still-to-flow vectors            
#         uh_q = uh.uh_5_half(tau_q, self.delta_t)
#         uh_s = uh.uh_5_half(tau_s, self.delta_t)
#         uh_t = uh.uh_8_delay(tau_d, self.delta_t)

#         # Could this be a dataframe?
#         self.uhs = {'uh_q': uh_q, 'uh_s': uh_s, 'uh_t': uh_t}

#     def model_fun(self, S): 

#         lp = self.params.values[0] # Wilting point [mm]
#         d = self.params.values[1] # Threshold for flow generation [mm]
#         p = self.params.values[2] # Flow response non-linearity [-]
#         alpha = self.params.values[3] # Fast/slow flow division [-]

#         # timestep 
#         # FIXME select correct time
#         # t = obj.t;                             % this time step
#         # climate_in = obj.input_climate(t,:);   % climate at this step
#         P = self.forcing['P']
#         Ep = self.forcing['Ep'] 

#         flux_ea = fluxes.evap_12(S, lp, Ep)
#         flux_u = fluxes.saturation_5(S, d, p, P)
#         flux_uq = fluxes.split_1(alpha, flux_u)
#         flux_us = fluxes.split_1(1 - alpha, flux_u)
#         flux_xq = uh.route(flux_uq, self.uhs['uh_q'])
#         flux_xs = uh.route(flux_us, self.uhs['uh_s'])
#         flux_xt = uh.route(flux_xq + flux_xs, self.uhs['uh_t'])

#         # Stores ODEs 
#         dS = -P + flux_ea + flux_u
#         fluxes = [flux_ea, flux_u, flux_uq, flux_us, flux_xq, flux_xs, flux_xt] 

#         return dS, fluxes

#     def step(self): 
#         # Update still-to-flow vectors using fluxes at current step and
#         # unit hydrographs
#         self.uhs['uh_q'] = uh.update_uh(self.uhs['uh_q'], self.fluxes['flux_uq'])
#         self.uhs['uh_s'] = uh.update_uh(self.uhs['uh_s'], self.fluxes['flux_us'])
#         self.uhs['uh_t'] = uh.update_uh(self.uhs['uh_t'], self.fluxes['flux_xq'] + self.fluxes['flux_xs'])

