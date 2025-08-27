
import calendar
import pandas as pd 

from typing import Optional, Dict, List

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
            'min_valid_months_per_year': min_valid_months_per_year,
        }

        # Only store non-None values
        self.criteria = {k: v for k, v in new_values.items() if v is not None}

        # Check consistency
        if ((self.criteria.get('min_tot_years') is not None or self.criteria.get('min_consecutive_years') is not None)
            and self.criteria.get('min_availability') is None):
            raise ValueError(
                "`min_availability` must be specified when using `min_tot_years` or `min_consecutive_years`"
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
        start_year, end_year = self.valid_years[0], self.valid_years[-1]
        start = pd.Timestamp(start_year, month, day)
        end = pd.Timestamp(end_year, month, day) - pd.Timedelta(days=1)
        full_range = pd.date_range(start=start, end=end, freq='1D', tz=None) # FIXME
        df = df.reindex(full_range)

        df['month'] = df.index.month # Add month
        df['days_in_month'] = df.index.days_in_month
        expected_days = df.groupby(['water_year', 'month'])[['days_in_month']].first()

        # # Drop rows where any of the data columns are NaN
        # df = df.dropna(subset=self.ts.data_columns)

        # # Count valid observations per water_year and month
        # monthly_counts = df.groupby(['water_year', 'month']).count()[self.ts.data_columns]
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
        months = [7 + i for i in range(0, 12)]
        months = [m % 12 if m > 12 else m for m in months]
        monthly_availability = monthly_availability[months] # reorder 
        monthly_availability.columns = [calendar.month_abbr[m] for m in monthly_availability.columns] # Give informative names
        monthly_availability.index = monthly_availability.index.astype(int) # FIXME ensure integer
        return monthly_availability

    def _get_n_tot_years(self) -> Optional[int]:
        if 'min_availability' in self.criteria or 'min_monthly_availability' in self.criteria:
            return len(self.valid_years) #(self.availability >= self.criteria['min_availability']).sum()
        return None

    def _get_n_consecutive_years(self) -> Optional[int]:
        if 'min_availability' in self.criteria or 'min_monthly_availability' in self.criteria:
            if self.valid_years.empty:
                return 0
            # sorted_years = self.valid_years.index.sort_values()
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
        avail.name = 'availability'
        avail = avail.reset_index(drop=False)
        avail = avail[avail['water_year'].isin(self.valid_years)]
        avail = avail.groupby('month')['availability'].mean()
        return avail

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