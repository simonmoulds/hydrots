"""Main module."""

import calendar
import pandas as pd 
import numpy as np
import warnings

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union

# from hydrots.validator import TSValidator
from hydrots.summary.summary import TSSummary


class TSValidator:

    def __init__(self, 
                 ts, 
                #  data: pd.DataFrame, 
                #  data_columns: List[str],
                #  freq: str,
                 start_year: Optional[int] = None,
                 end_year: Optional[int] = None,
                 min_tot_years: Optional[int] = None,
                 min_consecutive_years: Optional[int] = None,
                 min_availability: Optional[float] = None, 
                 min_monthly_availability: Optional[float] = None,
                 min_valid_months_per_year: Optional[int] = None):

        # TODO check that this is updated when HydroTS object is updated (e.g. when water year is recomputed)
        self.ts = ts 
        # self.data = ts.data  # Expect data to be formatted with datetime index and single/multi column
        # self.data_columns = ts.data_columns
        # self.water_year_start = ts.water_year_start
        # self.freq = freq
        self.criteria: Dict[str, Optional[float | int]] = {}
        self._set_validity_criteria(start_year, end_year, min_tot_years, min_consecutive_years, min_availability, min_monthly_availability, min_valid_months_per_year)
        self._compute_valid_years()
        self._compute_annual_availability()

    def _set_validity_criteria(self,
                               start_year: Optional[int],
                               end_year: Optional[int],
                               min_tot_years: Optional[int],
                               min_consecutive_years: Optional[int],
                               min_availability: Optional[float],
                               min_monthly_availability: Optional[float],
                               min_valid_months_per_year: Optional[int]):

        new_values = {
            'start_year': start_year,
            'end_year': end_year,
            'min_tot_years': min_tot_years,
            'min_consecutive_years': min_consecutive_years,
            'min_availability': min_availability,
            'min_monthly_availability': min_monthly_availability,
            'min_valid_months_per_year': min_valid_months_per_year
        }

        # Only store non-None values
        self.criteria = {k: v for k, v in new_values.items() if v is not None}

        # Check consistency
        if self.criteria.get("min_tot_years") is not None or self.criteria.get("min_consecutive_years") is not None:

            has_availability = self.criteria.get("min_availability") is not None
            has_monthly_combo = (
                self.criteria.get("min_monthly_availability") is not None
                and self.criteria.get("min_valid_months_per_year") is not None
            )
            if not (has_availability or has_monthly_combo):
                raise ValueError(
                    "`min_availability` or `min_monthly_availability` with `min_valid_months_per_year` "
                    "must be specified when using `min_tot_years` or `min_consecutive_years`"
                )

    def _update_criteria(self, **kwargs):
        self._set_validity_criteria(
            kwargs.get('start_year', self.criteria.get('start_year')),
            kwargs.get('end_year', self.criteria.get('end_year')),
            kwargs.get('min_tot_years', self.criteria.get('min_tot_years')),
            kwargs.get('min_consecutive_years', self.criteria.get('min_consecutive_years')),
            kwargs.get('min_availability', self.criteria.get('min_availability')),
            kwargs.get('min_monthly_availability', self.criteria.get('min_monthly_availability')),
            kwargs.get('min_valid_months_per_year', self.criteria.get('min_valid_months_per_year'))
        )
        return self

    def update(self, **kwargs): 
        self._update_criteria(**kwargs)
        self._compute_valid_years() 
        self._compute_annual_availability()

    def _compute_valid_years(self):
        # min_avail = self.criteria.get('min_availability')
        # min_monthly_avail = self.criteria.get('min_monthly_availability')

        # if min_avail is None and min_monthly_avail is None:
        #     return None

        # # Slice by start/end year if provided
        # avail = self.availability  # pandas Series indexed by year
        # start = self.criteria.get('start_year')
        # if start is not None:
        #     avail = avail[avail.index >= start]

        # end = self.criteria.get('end_year')
        # if end is not None:
        #     avail = avail[avail.index <= end]

        # # Apply annual availability mask if needed
        # valid_mask = pd.Series(True, index=avail.index)
        # if min_avail is not None:
        #     valid_mask &= (avail >= min_avail)

        # # Apply monthly availability check if specified
        # if min_monthly_avail is not None:
        #     raise NotImplementedError('`min_monthly_avail` not currently supported')
        #     # df = self.ts.data.copy()  # DataFrame with datetime index and column 'Q'
        #     # df['year'] = df.index.year
        #     # df['month'] = df.index.month
        #     # df['is_valid'] = ~df[self.ts.data_columns].isna()

        #     # # Count valid days per (year, month)
        #     # monthly_valid_counts = df.groupby(['year', 'month'])['is_valid'].sum()
        #     # # Total days per (year, month)
        #     # monthly_total_days = df.groupby(['year', 'month'])['is_valid'].count()
        #     # # Monthly availability [unstack() puts months into columns]
        #     # monthly_avail = (monthly_valid_counts / monthly_total_days).unstack()

        #     # # Check that all months in a year meet the threshold
        #     # monthly_valid_years = (monthly_avail >= min_monthly_avail).all(axis=1)
        #     # valid_mask &= valid_mask.index.isin(monthly_valid_years[monthly_valid_years].index)

        # # return valid_mask[valid_mask].index
        # self._valid_years = valid_mask[valid_mask].index
        min_avail = self.criteria.get('min_availability')
        min_monthly_avail = self.criteria.get('min_monthly_availability')
        min_valid_months_per_year = self.criteria.get('min_valid_months_per_year')

        if min_avail is None and min_monthly_avail is None:
            return None

        # Slice by start/end year if provided
        avail = self.availability  # pandas Series indexed by year
        # start = self.criteria.get('start_year')
        # if start is not None:
        #     avail = avail[avail.index >= start]

        # end = self.criteria.get('end_year')
        # if end is not None:
        #     avail = avail[avail.index <= end]

        # Apply annual availability mask if needed
        valid_mask = pd.Series(True, index=avail.index)
        if min_avail is not None:
            valid_mask &= (avail >= min_avail)

        # Apply monthly availability check if specified
        if min_monthly_avail is not None:
            # raise NotImplementedError('`min_monthly_avail` not currently supported')
            monthly_availability = self.monthly_availability
            # monthly_availability = monthly_availability.reindex(valid_mask.index)
            monthly_availability = monthly_availability >= min_monthly_avail
            n_valid_months = monthly_availability.sum(axis=1)
            if min_valid_months_per_year is None: 
                min_valid_months_per_year = 12

            valid_mask &= n_valid_months >= int(min_valid_months_per_year)

        # Handle start/end years
        start = self.criteria.get('start_year')
        if start is not None:
            valid_mask = valid_mask[valid_mask.index >= start]

        end = self.criteria.get('end_year')
        if end is not None:
            valid_mask = valid_mask[valid_mask.index <= end]

        # return valid_mask[valid_mask].index
        self._valid_years = valid_mask[valid_mask].index

    @property 
    def availability(self): 
        return self._compute_annual_availability() 

    @property 
    def monthly_availability(self): 
        return self._compute_monthly_availability()

    def _compute_annual_availability(self) -> pd.Series:
        """Compute fraction of non-NaN values per year."""
        df = self.ts.data.copy()

        def days_in_year(year: int, year_start: tuple[int, int]) -> int:
            """Return number of days in a custom year starting on (month, day)."""
            month, day = year_start
            # If year starts on or before Feb 28, then check if *this* year is leap
            # If year starts after Feb 28, check if *next* year is leap (because Feb 29 falls in that year)
            if (month, day) <= (2, 28):
                is_leap = calendar.isleap(year)
            else:
                is_leap = calendar.isleap(year + 1)

            return 366 if is_leap else 365

        expected = pd.DataFrame({'water_year': df['water_year'].unique()})
        expected['days_in_year'] = [days_in_year(yr, self.ts.water_year_start) for yr in expected['water_year']]
        expected = expected.set_index('water_year')
        
        annual_counts = (
            df[self.ts.data_columns]
            .groupby(df['water_year'])
            .apply(lambda x: x.notna().sum())
        )
        # df = df.dropna(subset=self.ts.data_columns)
        # annual_counts = df.groupby('water_year').count()[self.ts.data_columns]
        annual_availability = expected.merge(annual_counts, how='left', left_index=True, right_index=True)
        for col in self.ts.data_columns: 
            annual_availability[col] = annual_availability[col] / annual_availability['days_in_year']

        return annual_availability[self.ts.data_columns].min(axis=1) # FIXME

    def _compute_monthly_availability(self) -> pd.Series:
        """Compute fraction of non-NaN values per water year and month."""
        df = self.ts.data.copy()

        # To calculate monthly availability we need have a complete timeseries 
        # FIXME only works for daily data
        month, day = self.ts.water_year_start 
        start_year, end_year = df['water_year'].min(), df['water_year'].max()
        
        start = pd.Timestamp(start_year, month, day)
        end = pd.Timestamp(end_year + 1, month, day) - pd.Timedelta(days=1)
        full_range = pd.date_range(start=start, end=end, freq=self.ts.freq, tz=None)
        df = df.reindex(full_range)

        df['month'] = df.index.month # Add month
        df['days_in_month'] = df.index.days_in_month
        expected_days = df.groupby(['water_year', 'month'])[['days_in_month']].first()

        monthly_counts = (
            df[self.ts.data_columns]
            .groupby([df['water_year'], df['month']])
            .apply(lambda x: x.notna().sum())
        )
        # Calculate availability
        monthly_availability = expected_days.merge(monthly_counts, how='left', left_index=True, right_index=True)
        for col in self.ts.data_columns: 
            monthly_availability[col] = monthly_availability[col] / monthly_availability['days_in_month']

        monthly_availability = monthly_availability[self.ts.data_columns].min(axis=1) # FIXME - right to take min? 
        monthly_availability = monthly_availability.unstack(level='month')
        months = [month + i for i in range(0, 12)]
        months = [m % 12 if m > 12 else m for m in months]
        for m in months: 
            if m not in monthly_availability.columns:
                monthly_availability[m] = np.nan

        monthly_availability = monthly_availability[months] # reorder 
        monthly_availability.columns = [calendar.month_abbr[m] for m in monthly_availability.columns] # Give informative names
        monthly_availability = monthly_availability.fillna(0)
        return monthly_availability

    def _get_n_tot_years(self) -> Optional[int]:
        if 'min_availability' in self.criteria or 'min_monthly_availability' in self.criteria:
            return len(self.valid_years) #(self.availability >= self.criteria['min_availability']).sum()
        return None

    def _get_n_consecutive_years(self) -> Optional[int]:
        if 'min_availability' in self.criteria or 'min_monthly_availability' in self.criteria:
            if self.valid_years.empty:
                return 0
            sorted_years = self.valid_years.sort_values()
            diffs = sorted_years.to_series().diff().fillna(1)
            group = (diffs != 1).cumsum()
            return group.value_counts().max()
        return None

    @property 
    def valid_years(self) -> pd.Series: 
        return self._valid_years

    @property 
    def max_consecutive_valid_years(self) -> int: 
        return int(self._get_n_consecutive_years())

    @property 
    def valid_years_mean_availability(self) -> float: 
        return float(self.availability[self.valid_years].mean())
    
    @property 
    def valid_years_mean_monthly_availability(self) -> pd.Series: 
        avail = self.monthly_availability.copy()
        avail = avail[avail.index.isin(self.valid_years)]
        return avail.mean()

    def is_valid(self) -> Optional[bool]:
        checks = []

        if 'min_tot_years' in self.criteria:
            checks.append(self._get_n_tot_years() >= self.criteria['min_tot_years'])

        if 'min_consecutive_years' in self.criteria:
            checks.append(self._get_n_consecutive_years() >= self.criteria['min_consecutive_years'])

        if checks:
            return all(checks)
        return None

    def summary(self) -> dict:
        return {
            "min_tot_years": self.criteria.get('min_tot_years'),
            "actual_tot_years": self._get_n_tot_years(),
            "min_consecutive_years": self.criteria.get('min_consecutive_years'),
            "actual_consecutive_years": self._get_n_consecutive_years(),
            "min_availability": self.criteria.get('min_availability')
        }

def is_regular(times: pd.Series) -> bool:
    """
    Check if a time series follows a regular frequency pattern,
    even if some periods are missing.

    Parameters
    ----------
    times : pd.Series
        A datetime-like Series of timestamps (must be sorted and unique).

    Returns
    -------
    bool
        True if the time series is regular (i.e., evenly spaced), False otherwise.
    """
    times = pd.Series(pd.to_datetime(times)).dropna().sort_values().unique()
    if len(times) < 3:
        return True  # Can't tell much with < 3 points, assume regular
    
    diffs = pd.Series(times[1:] - times[:-1])
    most_common_diff = diffs.mode().iloc[0]
    # Check if all diffs are integer multiples of the mode
    is_multiple = diffs.apply(lambda x: x / most_common_diff).apply(lambda x: x.is_integer())
    return is_multiple.all()


def read_timeseries(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    elif path.suffix in [".xls", ".xlsx"]:
        return pd.read_excel(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


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
                 data: Union[str, Path, pd.DataFrame],
                 metadata: pd.DataFrame,
                 use_water_year: bool = True,
                 use_local_water_year: bool = True, 
                 wettest: bool = True, 
                 **kwargs):

        if isinstance(data, (str, Path)): 
            data = read_timeseries(data, **kwargs)
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Expected `data` to be a Pandas DataFrame or file path")
        
        if data.shape[0] <= 1:
            raise ValueError("The input data must contain more than one row to be treated as a time series.")

        data = self._standardize_data(data)
        try:
            self.freq = pd.infer_freq(data.index)
        except ValueError:
            self.freq = None

        self.data = self._format_data(data, use_water_year, use_local_water_year, wettest=wettest)
        self.metadata = self._format_metadata(metadata)

        # TODO change type of validator depending on application
        self.validator = TSValidator(self) #.data, self.data_columns)

    def update_validity_criteria(self, **kwargs): 
        self.validator.update(**kwargs)

    def update_intermittency_criteria(self, min_zero_flow_days: int = 5, min_zero_flow_years: int = 1): 
        if min_zero_flow_years > self.n_years: 
            warnings.warn(f'`min_zero_flow_years` exceeds total number of valid years (={self.n_years}): setting value to {self.n_years}')
            min_zero_flow_years = self.n_years 
        self.intermittency_criteria = {'min_zero_flow_days': min_zero_flow_days, 'min_zero_flow_years': min_zero_flow_years}

    def update_water_year(self, 
                          use_water_year: bool = True, 
                          water_year_start: Optional[tuple[int, int]] = None,
                          use_local_water_year: bool = True, 
                          wettest: bool = True): 

        if use_water_year:
            if water_year_start is not None: 
                self.water_year_start = water_year_start 
            else:
                self.water_year_start = self._get_water_year_start(self.data, use_local_water_year, wettest=wettest)
        else:
            self.water_year_start = (1, 1) # Jan 1 

        self.data = self._assign_water_year(self.data)
        self.validator.update()

    def _standardize_data(self, data: pd.DataFrame):
        data = self._standardize_columns(data)
        data = self._validate_columns(data)

        # Ensure time column is properly formatted
        data = data.dropna(subset='time')
        data['time'] = pd.to_datetime(data['time'])
        data['time'] = data['time'].dt.tz_localize(tz=None)
        data = data.sort_values(by='time')
        
        # Make time the index
        data = data.set_index('time')
        data.index.name = 'time'
        return data 

    def _format_data(self, data: pd.DataFrame, use_water_year: bool, use_local_water_year: bool, wettest: bool):
        # data = self._standardize_columns(data)
        # data = self._validate_columns(data)

        # # Ensure time column is properly formatted
        # data = data.dropna(subset='time')
        # data['time'] = pd.to_datetime(data['time'])
        # data['time'] = data['time'].dt.tz_localize(tz=None)
        # data = data.sort_values(by='time')
        
        # # Make time the index
        # data = data.set_index('time')
        # data.index.name = 'time'

        if use_water_year:
            self.water_year_start = self._get_water_year_start(data, use_local_water_year, wettest=wettest)
        else:
            self.water_year_start = (1, 1) # Jan 1 

        if is_regular(data.index.to_series()): 
            # freq = data.index.to_series().diff().mode().iloc[0]
            data = self._make_continuous(data) #, freq)
        
        # Timestep duration (days)
        data['timestep'] = data.index.to_series().diff().shift(-1).dt.total_seconds() / 86400.
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

    def _make_continuous(self, data: pd.DataFrame): #, freq: Union[pd.Timedelta, str]):
        """Enforce a continuous time series at the specified resolution."""

        # If the frequency could not be inferred then we simply return the original dataframe
        if not self.freq:
            return data 

        start = data.index[0]
        end = data.index[-1]
        full_range = pd.date_range(start=start, end=end, freq=self.freq, tz=None)
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

        Parameters 
        ----------
        data : pd.DataFrame 
            A DataFrame containing at least the following columns:
            - 'time' (np.datetime64): Timestamps.
            - 'Q' (float): Discharge values.
        wettest : bool
            If `wettest=True` then we assume the water year starts in 
            the first month after the highest mean monthly flow. This is 
            appropriate for computing low-flow statistics. On the other 
            hand, if `wettest=False` then we assume the water year 
            starts in the driest month on average. This is more 
            appropriate for computing high-flow statistics.  

        Raises
        ------
        ValueError
            If any month has no data over the entire time period.

        References
        ----------
        .. [1]  Chagas, V. B. P., Chaffe, P. L. B., & Blöschl, G. (2024). 
                Regional low flow hydrology: Model development and 
                evaluation. Water Resources Research, 60, e2023WR035063. 
                https://doi.org/10.1029/2023WR035063 
        """
        qmean = data.groupby(data.index.month)['Q'].mean()
        n_nan = qmean.isna().sum()
        if n_nan > 0:
            msg = f'Cannot compute local water year because {n_nan} months have no data over the entire period'
            raise ValueError(msg)

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
    def data_columns(self): 
        return [col for col in self.data.columns if col in self.VARIABLE_COLUMNS]

    @property
    def is_valid(self):
        """Check if dataframe has the minimum number of years at the specified availability."""
        return self.validator.is_valid()

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
    def valid_start(self): 
        return self.valid_data.index[0] if self.is_valid else pd.NaT
    
    @property 
    def valid_end(self): 
        return self.valid_data.index[-1] if self.is_valid else pd.NaT

    @property 
    def valid_years(self): 
        return self.validator.valid_years 

    @property 
    def valid_data(self): 
        return self.data[self.data['water_year'].isin(self.validator.valid_years)]

    @property 
    def n_valid_years(self): 
        return len(self.validator.valid_years)

    @property 
    def max_consecutive_valid_years(self): 
        return self.validator.max_consecutive_valid_years

    @property 
    def valid_mean_annual_availability(self): 
        return self.validator.valid_years_mean_availability
    
    @property 
    def valid_mean_monthly_availability(self): 
        return self.validator.valid_years_mean_monthly_availability

    # FIXME - this is confusing
    @property 
    def n_years(self):
        return len(self.valid_years) if self.is_valid else 0

    @property 
    def summary(self):
        return TSSummary(self)

    # @property 
    # def signature(self): 
    #     return TSSignature(self)

    @property 
    def valid_data(self): 
        return self.data[self.data['water_year'].isin(self.valid_years)]

    def __len__(self): 
        return self.data.shape[0]