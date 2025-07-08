#!/usr/bin/env python

import hydrots.timeseries as hts

def test_timeseries_loads(timeseries_with_noflow):
    ts = hts.HydroTS(timeseries_with_noflow, metadata=None, use_water_year=False)
    ts.update_validity_criteria(start_year=ts.start.year, end_year=ts.end.year, min_tot_years=1, min_availability=0.95)
    assert ts.start.year == 2000
    assert ts.end.year == 2000
    assert len(ts.valid_data['Q']) == 366  # Leap year
    assert ts.valid_data['Q'].isnull().sum() == 0  # No NaNs expected


def test_timeseries_validity(timeseries_with_gaps):
    ts = hts.HydroTS(timeseries_with_gaps, metadata=None, use_water_year=False)
    ts.update_validity_criteria(start_year=ts.start.year, end_year=ts.end.year, min_tot_years=1, min_availability=0.99)
    assert not ts.is_valid


def test_timeseries_is_intermittent(timeseries_with_noflow): 
    ts = hts.HydroTS(timeseries_with_noflow, metadata=None, use_water_year=False)
    ts.update_validity_criteria(start_year=ts.start.year, end_year=ts.end.year, min_tot_years=1, min_availability=0.95)
    ts.update_intermittency_criteria()
    assert ts.is_intermittent is False