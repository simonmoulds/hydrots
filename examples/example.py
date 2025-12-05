#!/usr/bin/env python3

import calendar
import numpy as np
import pandas as pd
import importlib
from pathlib import Path 
from functools import reduce

import hydrots.timeseries as hts
importlib.reload(hts)

# ROBIN data
DATADIR = Path("data/3b077711-f183-42f1-bac6-c892922c81f4")

meta = pd.read_csv(DATADIR / 'supporting-documents' / 'robin_station_metadata_public_v1-1.csv', encoding='latin1')

id = 'GB00055' # Lambourn at Shaw
id = 'AU00001'

x = pd.read_csv(DATADIR / 'data' / f'{id}.csv')

ts = hts.HydroTS(x, metadata=None, use_water_year=False)
# ts.update_validity_criteria(start_year=1960, end_year=2020, min_tot_years=40, min_availability=0.9)
# ts.update_water_year(use_water_year=False)
# ts.update_water_year(use_water_year=True, water_year_start=(7, 1))
ts.update_water_year(use_water_year=False) #, water_year_start=(7, 1))
# ts.update_validity_criteria(start_year=1984, end_year=1994, min_tot_years=5, min_availability=0.4)
ts.update_validity_criteria(start_year=1950, end_year=2025, min_tot_years=5, min_availability=0.8)
# ts.update_intermittency_criteria(min_zero_flow_days=1, min_zero_flow_years=1)

# FIXME update with `valid` column

import hydrots.summary.summary as hsm
importlib.reload(hsm)

# If `by_season=True` then the water year will be recomputed to match the seasons (e.g. MAM, JJA, SON, DJF -> water_year_start updated to (3, 1)
# Is this reasonable?
res = hsm._MaximumFlow(ts).compute(by_year=True, rolling=5, by_season=True)
res = hsm._MaximumFlow(ts).compute(by_year=True, by_season=True)
res = hsm._GSIM(ts).compute(annual=True)
res = hsm._GSIM(ts).compute(annual=False, monthly=True)
res = hsm._GSIM(ts).compute(annual=False, seasonal=True)
res = hsm._StreamflowIndices(ts).compute(by_year=False) 
res = hsm._StreamflowIndices(ts).compute(by_year=True, rolling=10) # NOTE - this works 
res = hsm._StreamflowIndices(ts).compute(by_year=True)
res, summary = hsm._HighFlowEvents(ts).compute(summarise=True, threshold=50, by_year=True, rolling=5)
res = hsm._HighFlowEvents(ts).compute(summarise=False, threshold=50, by_year=True)

# Check summary accessor
ts.summary.streamflow_indices(by_year=True)

q50 = ts.summary.flow_quantile(quantile=0.5)['Q50'].iloc[0]
res = hsm._HighFlowFraction(ts).compute(threshold=q50, by_year=True) #summarise=False, threshold=50, by_year=True)

self  = hsm._HighFlowFraction(ts)#.compute(threshold=q50) #summarise=False, threshold=50, by_year=True)

highflow_events = ts.summary.high_flow_fraction(threshold={'Q50_mult_1pt5': q50 * 1.5}, by_year=True) # FIXME
high_q_dur_mean = highflow_events['event_duration'].mean()
high_q_dur_std = highflow_events['event_duration'].std()
zero_q = ts.summary.no_flow_fraction(threshold=0.1, by_year=False)
zero_q = ts.summary.no_flow_fraction(threshold=0.1, by_year=True)

# BFI 
res = hsm._BFI(ts).compute()
ts.summary.baseflow_index()

# Richards-Baker index
ts.summary.richards_baker_index(by_year=False)
ts.summary.richards_baker_index(by_year=True)
ts.summary.richards_baker_index(by_year=True, rolling=5, center=False)

# Annual maximum flow
ts.summary.annual_maximum_flow()
ts.summary.maximum_flow()

# N-Day flow extreme FIXME what does `index` column refer to?
ts.summary.n_day_low_flow_extreme()
ts.summary.n_day_low_flow_extreme(by_year=True)
ts.summary.n_day_low_flow_extreme(by_year=True, rolling=5)

res = hsm._NDayFlowExtreme(ts).compute(by_year=True)
res = hsm._NDayFlowExtreme(ts).compute(rolling=5)

# FIXME
# _, res = hsm.dry_down_period(ts, summarise=True) # FIXME

quantiles = [0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
res = hsm.flow_quantile(ts, quantile=quantiles, by_year=False)
res = hsm.flow_quantile(ts, by_year=True, quantile=0.05)

res = hsm.high_flow_fraction(ts, threshold={'Q50': 0.27, 'Q80': 0.714})
res = hsm.low_flow_fraction(ts, threshold={'Q50': 0.27, 'Q80': 0.714}) # FIXME

q50 = hsm.flow_quantile(ts, quantile=[0.5], by_year=True)
q50 = hsm.flow_quantile(ts, quantile=[0.5, 0.99], by_year=False)['Q50'].iloc[0]
res = hsm.high_flow_fraction(ts, threshold={'Q50_times_1pt5': q50 * 1.5}, by_year=True)

hsm.no_flow_fraction(ts, threshold=0.1)

hsm.no_flow_fraction(ts, threshold=0.1)

res, summary = hsm.no_flow_events(ts, summarise=True)

res, summary = hsm.high_flow_events(ts, summarise=True)
res = hsm._NDayFlowExtreme(ts).compute(by_year=True)
res = hsm.discharge_variability_index(ts, by_year=True)
res = hsm.cumulative_discharge_variability_index(ts)
res = hsm.richards_baker_index(ts)

res, summary = hsm.low_flow_events(ts, summarise=True)

res = hsm.no_flow_fraction(ts, threshold=0.01)
res = hsm.no_flow_fraction(ts, threshold=0.01, rolling=5)

res = hsm.baseflow_index(ts, rolling=5)

res = hsm.slope_flow_duration_curve(ts)

res = hsm._CV(ts).compute(by_year=True)
res = hsm._CV(ts).compute(rolling=5)

threshold = ts.data['Q'].quantile(0.9)
# threshold += 1000
res, summary = hsm._POT(ts).compute(threshold=threshold, summarise=True, by_year=True)
res = hsm._POT(ts).compute(threshold=threshold, summarise=False, by_year=True)

# # FIXME
# res, summary = hsm._DryDownPeriod(ts).compute(summarise=True, by_year=True)

# res = hsm._RichardsBakerIndex(ts).compute(by_year=False)
# res = hsm._MaximumFlow(ts).compute(by_year=True)

# threshold = ts.data['Q'].quantile(0.95)
# res, summary = hsm._POT(ts).compute(threshold=threshold, summarise=True)
# res, summary = hsm._POT(ts).compute(threshold=threshold, summarise=True, by_year=True)
# res, summary = hsm._POT(ts).compute(threshold=threshold, summarise=True, rolling=5)

# res, summary = hsm._LowFlowEvents(ts).compute(summarise=True)
# res, summary = hsm._LowFlowEvents(ts).compute(summarise=True, by_year=True)
# res, summary = hsm._LowFlowEvents(ts).compute(summarise=True, rolling=5)

# res = hsm.HighFlowEvents(ts).compute()
# res, summary = hsm._HighFlowEvents(ts).compute(summarise=True)
# res, summary = hsm._HighFlowEvents(ts).compute(summarise=True, by_year=True)
# res, summary = hsm._HighFlowEvents(ts).compute(summarise=True, rolling=5)

# res = hsm.NoFlowEvents(ts).compute() 
# res, summary = hsm._NoFlowEvents(ts).compute(summarise=True)
# res, summary = hsm._NoFlowEvents(ts).compute(summarise=True, by_year=True)
# res, summary = hsm._NoFlowEvents(ts).compute(summarise=True, rolling=5)
