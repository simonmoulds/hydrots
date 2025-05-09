
import numpy as np
import pandas as pd
import importlib
from pathlib import Path 
from functools import reduce

import hydrots.timeseries as hts

importlib.reload(hts)

DATADIR = Path("data/3b077711-f183-42f1-bac6-c892922c81f4")

meta = pd.read_csv(DATADIR / 'supporting-documents' / 'robin_station_metadata_public_v1-1.csv', encoding='latin1')

id = 'GB00055' # Lambourn at Shaw

x = pd.read_csv(DATADIR / 'data' / f'{id}.csv')

ts = hts.HydroTS(x, metadata=None, freq='1D')
ts.update_validity_criteria(start_year=1960, end_year=2020, min_tot_years=40, min_availability=0.95)
ts.update_water_year(wettest=True)

min7 = ts.summary.n_day_low_flow_extreme(n=7)
min30 = ts.summary.n_day_low_flow_extreme(n=30)
min7dur = ts.summary.max_low_flow_duration(0.99)
max_deficit = ts.summary.max_low_flow_deficit(0.05)
noflow_freq = ts.summary.no_flow_frequency()
noflow_dur = ts.summary.no_flow_event_duration()

ts.update_water_year(wettest=False)
amax = ts.summary.annual_maximum_flow()

dfs = [amax, min7, min30, min7dur, max_deficit, noflow_freq, noflow_dur]
merged = reduce(
    lambda left, right: pd.merge(left, right, on="water_year", how="outer"),
    dfs
)