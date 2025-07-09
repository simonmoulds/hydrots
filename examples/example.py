
import calendar
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
# x = pd.read_csv('data/extra/valid_data/OHDB_GBR_NRFA_00011.csv')
# x = pd.read_csv('/Users/smoulds/projects/streamflow-data/results/OHDB_BOM__AUS_03586.csv')
# x = pd.read_csv('/Users/smoulds/dev/hydrots/data/extra/timeseries/OHDB_ARSO_SVN_00010.csv')
# x = pd.read_csv('/Users/smoulds/projects/streamflow-data/results/OHDB_WRIS_IND_00414.csv')
x = pd.read_csv('/Users/smoulds/dev/hydrots/data/extra/timeseries/OHDB_NRFA_GBR_00001.csv')
x = pd.read_csv('/Users/smoulds/dev/hydrots/data/extra/timeseries/OHDB_NRFA_GBR_00009.csv')
x = pd.read_csv('/Users/smoulds/dev/hydrots/data/extra/timeseries/OHDB_NRFA_GBR_00010.csv')


ts = hts.HydroTS(x, metadata=None, use_water_year=False)
# ts.update_validity_criteria(start_year=1960, end_year=2020, min_tot_years=40, min_availability=0.9)
# ts.update_water_year(use_water_year=False)
ts.update_water_year(use_water_year=True, water_year_start=(7, 1))
# ts.update_validity_criteria(start_year=1984, end_year=1994, min_tot_years=5, min_availability=0.4)
ts.update_validity_criteria(start_year=1984, end_year=2025, min_tot_years=5, min_availability=0.4)

ts.valid_mean_annual_availability 
ts.valid_start
ts.valid_end
ts.max_consecutive_valid_years 
ts.n_valid_years 

mean_monthly_availability = ts.valid_mean_monthly_availability.to_frame().T
mean_monthly_availability.columns = ['ohdb_valid_data_mean_' + m.lower() + '_availability' for m in mean_monthly_availability.columns] 

# ts.update_water_year(use_water_year=False)
q50 = ts.summary.flow_quantile(quantile=0.5)['Q50']
highflow_events = ts.summary.high_flow_fraction(threshold={'Q50_mult_1pt5': q50.values * 1.5}, by_year=True)
high_q_dur_mean = highflow_events['event_duration'].mean()
high_q_dur_std = highflow_events['event_duration'].std()
zero_q = ts.summary.no_flow_fraction(threshold=0.1, by_year=False)

# BFI 
ts.summary.baseflow_index()

# Richards-Baker index
ts.summary.richards_baker_index(by_year=False)
ts.summary.richards_baker_index(by_year=True)
ts.summary.richards_baker_index(by_year=False, rolling=5, center=False)

# Annual maximum flow
ts.summary.annual_maximum_flow()
ts.summary.maximum_flow()

# N-Day flow extreme FIXME what does `index` column refer to?
ts.summary.n_day_low_flow_extreme()
ts.summary.n_day_low_flow_extreme(by_year=True)
ts.summary.n_day_low_flow_extreme(rolling=5)

import hydrots.summary.summary as hsm
importlib.reload(hsm)

res = hsm._NDayFlowExtreme(ts).compute(by_year=True)
res = hsm._NDayFlowExtreme(ts).compute(rolling=5)

_, res = hsm.dry_down_period(ts, summarise=True)

quantiles = [0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
res = hsm.flow_quantile(ts, quantile=quantiles, by_year=False)
res = hsm.flow_quantile(ts, quantile=0.05)

res = hsm.high_flow_fraction(ts, threshold={'Q50': 0.27, 'Q80': 0.714})
res = hsm.low_flow_fraction(ts, threshold={'Q50': 0.27, 'Q80': 0.714})

# FIXME - if list of length one is given then column names incorrect
q50 = hsm.flow_quantile(ts, quantile=[0.5, 0.99], by_year=False)['Q50'].values
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

threshold = ts.data['Q'].quantile(0.25)
# threshold += 1000
res, summary = hsm._POT(ts).compute(threshold=threshold, summarise=True, by_year=True)
res = hsm._POT(ts).compute(threshold=threshold, summarise=False, by_year=True)

res, summary = hsm._DryDownPeriod(ts).compute(summarise=True, by_year=True)

res = hsm._RichardsBakerIndex(ts).compute(by_year=False)
res = hsm._MaximumFlow(ts).compute(by_year=True)

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