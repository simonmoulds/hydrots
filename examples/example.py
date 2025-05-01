
import pandas as pd
import importlib
import hydrots.dynamicmodel as dm

importlib.reload(dm)

# TESTING 
forcing = pd.read_csv("data/input_climatology.csv", parse_dates=['dates_as_datenum'])
forcing = forcing.rename(columns={'dates_as_datenum': 'time', 'precipitation': 'P', 'temperature': 'T', 'potential_evapotranspiration': 'Ep', 'streamflow': 'Q'})
forcing = forcing[['time', 'P', 'T', 'Ep', 'Q']].set_index('time')

par_ini = [35., 3.7, 0.4, 0.25, 0.01]
store_ini = [15., 7., 3., 8., 22.]
solver_opts = dm.SolverOptions(resnorm_maxiter=6, resnorm_tolerance=0.1)

params = dm.Hymod5p5sParams()
model = dm.Hymod5p5s(params=params, initial_conditions=store_ini, forcing=forcing, delta_t=1, solver_opts=solver_opts)

model.run(params=par_ini, initial_conditions=store_ini) # OK