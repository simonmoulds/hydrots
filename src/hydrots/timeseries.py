"""Main module."""

import pandas as pd 
import numpy as np
import warnings

from abc import ABC, abstractmethod
from typing import Optional, Dict, List

from hydrots.validator import TSValidator
from hydrots.summary import TSSummary

class HydroTS: 

    TIME_COLUMN = 'time'
    VARIABLE_COLUMNS = ['Q', 'H', 'T2M']
    COLUMN_ALIASES = {
        'date': 'time',
        'discharge': 'Q',
        'flow': 'Q',
        'flow_cumecs': 'Q'
    }

    def __init__(self, 
                 data: pd.DataFrame, 
                 metadata: pd.DataFrame,
                 freq: str, 
                 use_water_year: bool = True,
                 use_local_water_year: bool = True, 
                 wettest: bool = True):

        self.freq = freq 
        self.data = self._format_data(data, use_local_water_year, use_water_year, wettest=wettest)
        self.metadata = self._format_metadata(metadata)

        # TODO change type of validator depending on application
        self.validator = TSValidator(self.data, self.data_columns, self.freq)

    def update_validity_criteria(self, **kwargs): 
        self.validator.update_criteria(**kwargs)

    def update_intermittency_criteria(self, min_zero_flow_days: int = 5, min_zero_flow_years: int = 1): 
        if min_zero_flow_years > self.n_years: 
            warnings.warn(f'`min_zero_flow_years` exceeds total number of valid years (={self.n_years}): setting value to {self.n_years}')
            min_zero_flow_years = self.n_years 
        self.intermittency_criteria = {'min_zero_flow_days': min_zero_flow_days, 'min_zero_flow_years': min_zero_flow_years}

    @property 
    def data_columns(self): 
        return [col for col in self.data.columns if col in self.VARIABLE_COLUMNS]

    @property
    def is_valid(self):
        """Check if dataframe has the minimum number of years at the specified availability."""
        return self.validator.is_valid()

    def update_water_year(self, use_water_year: bool = True, use_local_water_year: bool = True, wettest: bool = True): 
        if use_water_year:
            self.water_year_start = self._get_water_year_start(self.data, use_local_water_year, wettest=wettest)
        else:
            self.water_year_start = (1, 1) # Jan 1 

        self.data = self._assign_water_year(self.data)

    def _format_data(self, data: pd.DataFrame, use_water_year: bool, use_local_water_year: bool, wettest: bool):
        data = self._standardize_columns(data)
        data = self._validate_columns(data)
        
        # Ensure time column is properly formatted
        data['time'] = pd.to_datetime(data['time'])
        data = data.sort_values(by='time')
        data = data.set_index('time')
        data.index.name = 'time'

        if use_water_year:
            self.water_year_start = self._get_water_year_start(data, use_local_water_year, wettest=wettest)
        else:
            self.water_year_start = (1, 1) # Jan 1 

        data = self._make_continuous(data)
        data = self._assign_water_year(data)
        return data 

    def _standardize_columns(self, data: pd.DataFrame):
        """Replace known aliases of standard column names."""
        return data.rename(columns={k: v for k, v in self.COLUMN_ALIASES.items() if k in data.columns})
    
    def _validate_columns(self, data: pd.DataFrame):
        missing_time = self.TIME_COLUMN if self.TIME_COLUMN not in data.columns else None
        missing_variable = False
        messages = []
        if missing_time:
            messages.append(f"Missing time column: '{missing_time}'")

        if missing_variable:
            messages.append(f"Missing variable column(s)")

        if messages:
            raise ValueError("\n".join(messages))

        valid_columns = [col for col in data.columns if col in [self.TIME_COLUMN] + self.VARIABLE_COLUMNS]
        data = data[valid_columns]
        return data 

    def _make_continuous(self, data: pd.DataFrame):
        """Enforce a continuous time series at the specified resolution."""
        start = data.index[0]
        end = data.index[-1]
        full_range = pd.date_range(start=start, end=end, freq=self.freq)
        data = data.reindex(full_range)
        data.index.name = 'time'
        return data

    def _format_metadata(self, metadata: pd.DataFrame): 
        return metadata 

    def _impute_missing_values(self): 
        """Impute missing data in timeseries."""
        raise NotImplementedError()

    def _compute_availability(self):
        """Compute the data availability in each year."""
        avail = self.data.groupby(self.data['water_year']).apply(lambda g: g['Q'].notna().sum() / g.shape[0], include_groups=False)
        avail = avail.sort_index()
        return avail

    def _get_local_water_year_start(self, data: pd.DataFrame, wettest: bool = True):
        """Compute the start of the local water year.

        If `wettest=True` then we assume the water year starts in the 
        first month after the highest mean monthly flow. This is 
        appropriate for computing low-flow statistics. On the other 
        hand, if `wettest=False` then we assume the water year starts 
        in the driest month on average. This is more appropriate for 
        computing high-flow statistics.  
        """
        qmean = data.groupby(data.index.month)['Q'].mean()
        if wettest: 
            # Assume water year starts in the first month after the highest mean monthly flow
            # https://doi.org/10.1029/2023WR035063 
            water_year_start_month = (int(qmean.idxmax())) % 12 + 1
        else:
            # Assume water year starts in the driest month
            water_year_start_month = int(qmean.idxmin())
        return (water_year_start_month, 1)

    def _get_water_year_start(self, data: pd.DataFrame, use_local_water_year: bool, **kwargs): 
        if use_local_water_year:
            return self._get_local_water_year_start(data, **kwargs)
        else:
            if self.metadata['latitude'] > 0:
                # Northern hemisphere starts Oct 1
                return (10, 1)
            else:
                # Southern hemisphere starts Apr 1
                return (4, 1)

    def _assign_water_year(self, data: pd.DataFrame):
        """Assign water year to timeseries dataset."""
        def get_water_year(dt, start_month, start_day):
            if (dt.month, dt.day) >= (start_month, start_day):
                return dt.year
            else:
                return dt.year - 1
        wy = pd.Series(data.index).apply(lambda x: get_water_year(x, *self.water_year_start))
        data['water_year'] = wy.values
        return data

    @property
    def is_intermittent(self):
        """
        Check the streamflow data to see if the catchment is intermittent.

        A timeseries is considered intermittent if it has, on average, more than
        `min_zero_flow_days` days of zero flow per year and if at least a 
        `min_zero_flow_years_frac` fraction of years containing one or more zero-flow days.

        Returns:
            bool: True if the timeseries is intermittent, False otherwise.
        """

        min_zero_flow_days = self.intermittency_criteria['min_zero_flow_days']
        min_zero_flow_years = self.intermittency_criteria['min_zero_flow_years']
        if min_zero_flow_days is None or min_zero_flow_years is None:
            return None

        if min_zero_flow_years > self.n_years: 
            warnings.warn(f'`min_zero_flow_years` exceeds total number of valid years (={self.n_years}): setting value to {self.n_years}')
            min_zero_flow_years = self.n_years 
        # Filter the dataframe for available years
        data = self.data[self.data.index.year.isin(self.valid_years)]
        
        # Calculate the number of zero-flow days per year
        n_zero = data.groupby(data['water_year'])['Q'].apply(lambda x: (x < 0.001).sum(), include_groups=False)
        
        # Mean number of zero flow days per year
        mean_n_zero = n_zero.mean()
        
        # Fraction of years that are intermittent
        n_intermittent = (n_zero >= min_zero_flow_days).sum() #/ len(self.valid_years)
 
        # Return True if criteria for intermittent are met
        return (float(mean_n_zero) >= min_zero_flow_days) and (float(n_intermittent) > min_zero_flow_years)

    @property 
    def start(self): 
        return self.data.index[0]
    
    @property 
    def end(self): 
        return self.data.index[-1]

    @property 
    def valid_years(self): 
        return self.validator.valid_years 

    @property 
    def n_years(self):
        return len(self.valid_years)

    @property 
    def summary(self):
        return TSSummary(self)

    @property 
    def valid_data(self): 
        return self.data[self.data['water_year'].isin(self.valid_years)]

    def __len__(self): 
        return self.data.shape[0]