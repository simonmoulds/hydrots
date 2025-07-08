#!/usr/bin/env python

import hydrots.timeseries as hts

def test_noflow_detects_events(timeseries_with_noflow):
    ts = hts.HydroTS(timeseries_with_noflow, metadata=None, use_water_year=False)
    ts.update_validity_criteria(start_year=ts.start.year, end_year=ts.end.year, min_tot_years=1, min_availability=0.95)
    _, summary = ts.summary.no_flow_events(summarise=True)
    assert summary["n_events"].values[0] == 2

# def test_noflow_with_valid_data(simple_timeseries):
#     summary = NoFlowEvents(simple_timeseries)
#     result = summary.compute()
#     assert result["count"] == 0
