
import numpy as np
import pandas as pd
import importlib
import hydrots.dynamicmodel as dm

from hydrots.objectivefuns import of_KGE

importlib.reload(dm)

# TESTING 
forcing = pd.read_csv("data/input_climatology.csv", parse_dates=['dates_as_datenum'])
forcing = forcing.rename(columns={'dates_as_datenum': 'time', 'precipitation': 'P', 'temperature': 'T', 'potential_evapotranspiration': 'Ep', 'streamflow': 'Q'})
forcing = forcing[['time', 'P', 'T', 'Ep', 'Q']].set_index('time')

solver_opts = dm.SolverOptions(resnorm_maxiter=6, resnorm_tolerance=0.1)

params = dm.Hymod5p5sParams()

par_ini = [35., 3.7, 0.4, 0.25, 0.01]
store_ini = [15., 7., 3., 8., 22.]

model = dm.Hymod5p5s(params=params, initial_conditions=store_ini, forcing=forcing, delta_t=1, solver_opts=solver_opts)

model.run(forcing.index[0], forcing.index[-1], params=par_ini, initial_conditions=store_ini)

# self = dm.Hymod5p5s(params=params, initial_conditions=store_ini, forcing=forcing, delta_t=1, solver_opts=solver_opts)

warmup_start_time = pd.Timestamp('1989-01-01')
warmup_end_time = pd.Timestamp('1989-12-31')
cal_start_time = pd.Timestamp('1990-01-01')
cal_end_time = pd.Timestamp('1991-12-31')

model.calibrate(warmup_start_time, warmup_end_time, cal_start_time, cal_end_time, inverse_flag=True)

# Qsim = model.output['Q']
# Qobs = model.forcing.loc[model.simulation_time]['Q']
# of = of_KGE(Qobs, Qsim)

# store_ini = model.stores.iloc[0].to_numpy()
# model.run(cal_start_time, cal_end_time, params=par_ini, initial_conditions=store_ini) # OK
# Qsim = model.output['Q']
# Qobs = model.forcing.loc[model.simulation_time]['Q']
# of = of_KGE(Qobs, Qsim)

# par_ini = model.calibration_result.x 
# store_ini = model.stores.iloc[0].to_numpy()
# model.run(cal_start_time, cal_end_time, params=par_ini, initial_conditions=store_ini) # OK
# Qsim = model.output['Q']
# Qobs = model.forcing.loc[model.simulation_time]['Q']
# of = of_KGE(Qobs, Qsim)