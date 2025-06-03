
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
id = 'AU00001'

x = pd.read_csv(DATADIR / 'data' / f'{id}.csv')
x = pd.read_csv('data/extra/valid_data/OHDB_GBR_NRFA_00337.csv')

ts = hts.HydroTS(x, metadata=None, freq='1D')
ts.update_validity_criteria(start_year=1960, end_year=2020, min_tot_years=40, min_availability=0.95)
# ts.update_water_year(wettest=True)
ts.update_water_year(use_water_year=False)

# BFI 
ts.summary.baseflow_index()

# Richards-Baker index
ts.summary.richards_baker_index(by_year=False)
ts.summary.richards_baker_index(by_year=True)
ts.summary.richards_baker_index(by_year=False, rolling=5, center=False)

# Annual maximum flow
ts.summary.annual_maximum_flow()
ts.summary.maximum_flow()

# N-Day flow extreme 
ts.summary.n_day_low_flow_extreme()
ts.summary.n_day_low_flow_extreme(by_year=True)
ts.summary.n_day_low_flow_extreme(rolling=5)

import hydrots.summary.summary as hsm
importlib.reload(hsm)

res, summary = hsm.no_flow_events(ts, summarise=True)

res = hsm.flow_quantile(ts, quantile=0.95)
res = hsm.flow_quantile(ts, quantile=0.05)

res, summary = hsm.high_flow_events(ts, summarise=True)
res = hsm._NDayFlowExtreme(ts).compute(by_year=True)
res = hsm.discharge_variability_index(ts, by_year=True, safe=True)
res = hsm.cumulative_discharge_variability_index(ts)
res = hsm.richards_baker_index(ts)

res, summary = hsm.low_flow_events(ts, summarise=True)

res = hsm.no_flow_fraction(ts)
res = hsm.no_flow_fraction(ts, rolling=5)

res = hsm.baseflow_index(ts, rolling=5)

res = hsm.slope_flow_duration_curve(ts)

res = hsm._CV(ts).compute(by_year=True)
res = hsm._CV(ts).compute(rolling=5)

threshold = ts.data['Q'].quantile(0.25)
# threshold += 1000
res, summary = hsm._POT(ts).compute(threshold=threshold, summarise=True, by_year=True)
res = hsm._POT(ts).compute(threshold=threshold, summarise=False, by_year=True)

res, summary = hsm._DryDownPeriod(ts).compute(summarise=True, by_year=True)

res = hsm._RichardsBakerIndex(ts).compute(by_year=False)
res = hsm._MaximumFlow(ts).compute(by_year=True)
res = hsm._NDayFlowExtreme(ts).compute(by_year=True)
res = hsm._NDayFlowExtreme(ts).compute(rolling=5)

threshold = ts.data['Q'].quantile(0.95)
res, summary = hsm._POT(ts).compute(threshold=threshold, summarise=True)
res, summary = hsm._POT(ts).compute(threshold=threshold, summarise=True, by_year=True)
res, summary = hsm._POT(ts).compute(threshold=threshold, summarise=True, rolling=5)

res, summary = hsm._LowFlowEvents(ts).compute(summarise=True)
res, summary = hsm._LowFlowEvents(ts).compute(summarise=True, by_year=True)
res, summary = hsm._LowFlowEvents(ts).compute(summarise=True, rolling=5)

res = hsm.HighFlowEvents(ts).compute()
res, summary = hsm._HighFlowEvents(ts).compute(summarise=True)
res, summary = hsm._HighFlowEvents(ts).compute(summarise=True, by_year=True)
res, summary = hsm._HighFlowEvents(ts).compute(summarise=True, rolling=5)

res = hsm.NoFlowEvents(ts).compute() 
res, summary = hsm._NoFlowEvents(ts).compute(summarise=True)
res, summary = hsm._NoFlowEvents(ts).compute(summarise=True, by_year=True)
res, summary = hsm._NoFlowEvents(ts).compute(summarise=True, rolling=5)

# ts.signature.coefficient_of_variation(by_year=False, rolling=10, center=False)
# ts.signature.skewness(rolling=10, center=False)

# q50 = ts.data['Q'].quantile(0.5)
# ts.summary.peaks_over_threshold_events(threshold=q50)
# # ts.summary.peaks_over_threshold_events_original(threshold=q50)

# q95 = ts.data['Q'].quantile(0.95)
# ts.summary.peaks_over_threshold_events(threshold=q95)
# # ts.summary.peaks_over_threshold_events_original(threshold=q95)
# ts.summary.low_flow_events()

# ts.summary.no_flow_event_duration() 
# ts.summary.dry_down_period(quantile=0.25)

# # ts.summary.peaks_over_threshold_events(threshold)
# # ts.signature.coefficient_of_variation()
# # ts.signature.richards_baker_index()
# # ts.signature.discharge_variability_index()
# # ts.signature.cumulative_discharge_variability_index()

# min7 = ts.summary.n_day_low_flow_extreme(n=7)
# min30 = ts.summary.n_day_low_flow_extreme(n=30)
# min7dur = ts.summary.max_low_flow_duration(0.99)
# max_deficit = ts.summary.max_low_flow_deficit(0.05)
# noflow_freq = ts.summary.no_flow_frequency()
# noflow_dur = ts.summary.no_flow_event_duration()

# ts.update_water_year(wettest=False)
# amax = ts.summary.annual_maximum_flow()

# dfs = [amax, min7, min30, min7dur, max_deficit, noflow_freq, noflow_dur]
# merged = reduce(
#     lambda left, right: pd.merge(left, right, on="water_year", how="outer"),
#     dfs
# )