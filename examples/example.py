
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
x = pd.read_csv('data/extra/valid_data/OHDB_GBR_NRFA_00011.csv')

ts = hts.HydroTS(x, metadata=None)
# ts.update_validity_criteria(start_year=1960, end_year=2020, min_tot_years=40, min_availability=0.9)
# ts.update_water_year(use_water_year=False)
ts.update_water_year(use_water_year=True, water_year_start=(7, 1))
ts.update_validity_criteria(start_year=1950, end_year=2021, min_tot_years=20, min_availability=0.9)

qmax = ts.summary.maximum_flow(by_year=False)['QMAX']
ts.summary.no_flow_fraction(threshold=0.1)
events, res = ts.summary.dry_down_period(summarise=True)

# TESTING DRY DOWN PERIOD 
import hydrots.summary.summary as hsm
importlib.reload(hsm)
threshold = ts.valid_data['Q'].quantile(0.25)
pot_events = hsm._POT(ts).compute(threshold=threshold)
noflow_events = hsm._NoFlowEvents(ts).compute(threshold=0.)
high_flow_end_times = pot_events['event_end_time'].values
noflow_start_times = noflow_events['event_start_time'].values
dry_down_events = []
j = 0  # pointer for noflow_start_dates

for n in noflow_start_times:
    # Advance j to find the last high flow before n
    while j < len(high_flow_end_times) and high_flow_end_times[j] < n:
        j += 1
    if j == 0:
        continue  # no high-flow end before this no-flow start
    else:
        most_recent_high_end = high_flow_end_times[j - 1]
        # Need to do this because the water year in the pot_events dataframe corresponds to the start of the event, not the end
        water_year = ts.valid_data.loc[most_recent_high_end, 'water_year']
        dry_down_period = (n - most_recent_high_end).astype("timedelta64[D]").item()
        dry_down_events.append(pd.DataFrame({'water_year': [water_year], 'event_start_time': [most_recent_high_end], 'event_end_time': [n], 'event_duration': [dry_down_period]}))

dry_down_events = pd.concat(dry_down_events, axis=0).reset_index(drop=True)
dry_down_events = dry_down_events.drop_duplicates(subset='event_start_time') # FIXED

# if summarise:
#     summary = self._summarize_events(dry_down_events, by_year=by_year, rolling=rolling, center=center)
#     return dry_down_events, summary
# else:
#     return dry_down_events

import hydrots.summary.baseflow as hbf
importlib.reload(hbf)

# Q = x['Q'].dropna().values 
# v = hbf.lh(Q)
# v = hbf.lh2(Q) 

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

_, res = hsm.dry_down_period(ts, summarise=True)

quantiles = [0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
res = hsm.flow_quantile(ts, quantile=quantiles, safe=True, by_year=False)
res = hsm.flow_quantile(ts, quantile=0.05)

res = hsm.high_flow_fraction(ts, threshold={'Q50': 0.27, 'Q80': 0.714})
res = hsm.low_flow_fraction(ts, threshold={'Q50': 0.27, 'Q80': 0.714})

# FIXME - if list of length one is given then column names incorrect
q50 = hsm.flow_quantile(ts, quantile=[0.5, 0.99], safe=True, by_year=False)['Q50'].values
res = hsm.high_flow_fraction(ts, threshold={'Q50_times_1pt5': q50 * 1.5}, by_year=True)

hsm.no_flow_fraction(ts, threshold=0.1)

hsm.no_flow_fraction(ts, threshold=0.1)

res, summary = hsm.no_flow_events(ts, summarise=True)

res, summary = hsm.high_flow_events(ts, summarise=True)
res = hsm._NDayFlowExtreme(ts).compute(by_year=True, safe=True)
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