
import pandas as pd 

from typing import Optional, Dict, List

class TSValidator:

    def __init__(self, 
                 data: pd.DataFrame, 
                 data_columns: List[str],
                #  freq: str,
                 start_year: Optional[int] = None,
                 end_year: Optional[int] = None,
                 min_tot_years: Optional[int] = None,
                 min_consecutive_years: Optional[int] = None,
                 min_availability: Optional[float] = None, 
                 min_monthly_availability: Optional[float] = None):

        # TODO check that this is updated when HydroTS object is updated (e.g. when water year is recomputed)
        self.data = data  # Expect data to be formatted with datetime index and single/multi column
        self.data_columns = data_columns
        # self.freq = freq
        self.criteria: Dict[str, Optional[float | int]] = {}
        self._set_validity_criteria(start_year, end_year, min_tot_years, min_consecutive_years, min_availability, min_monthly_availability)
        self._compute_valid_years()
        self._compute_annual_availability()

    def _set_validity_criteria(self,
                               start_year: Optional[int],
                               end_year: Optional[int],
                               min_tot_years: Optional[int],
                               min_consecutive_years: Optional[int],
                               min_availability: Optional[float],
                               min_monthly_availability: Optional[float]):

        new_values = {
            'start_year': start_year,
            'end_year': end_year,
            'min_tot_years': min_tot_years,
            'min_consecutive_years': min_consecutive_years,
            'min_availability': min_availability,
            'min_monthly_availability': min_monthly_availability
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
            kwargs.get('min_monthly_availability', self.criteria.get('min_monthly_availability'))
        )
        return self

    def update(self, **kwargs): 
        self._update_criteria(**kwargs)
        self._compute_valid_years() 
        self._compute_annual_availability()

    def _compute_valid_years(self):
        min_avail = self.criteria.get('min_availability')
        min_monthly_avail = self.criteria.get('min_monthly_availability')

        if min_avail is None and min_monthly_avail is None:
            return None

        # Slice by start/end year if provided
        avail = self.availability  # pandas Series indexed by year
        start = self.criteria.get('start_year')
        if start is not None:
            avail = avail[avail.index >= start]

        end = self.criteria.get('end_year')
        if end is not None:
            avail = avail[avail.index <= end]

        # Apply annual availability mask if needed
        valid_mask = pd.Series(True, index=avail.index)
        if min_avail is not None:
            valid_mask &= (avail >= min_avail)

        # Apply monthly availability check if specified
        if min_monthly_avail is not None:
            df = self.data.copy()  # DataFrame with datetime index and column 'Q'
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['is_valid'] = ~df[self.data_columns].isna()

            # Count valid days per (year, month)
            monthly_valid_counts = df.groupby(['year', 'month'])['is_valid'].sum()
            # Total days per (year, month)
            monthly_total_days = df.groupby(['year', 'month'])['is_valid'].count()
            # Monthly availability [unstack() puts months into columns]
            monthly_avail = (monthly_valid_counts / monthly_total_days).unstack()

            # Check that all months in a year meet the threshold
            monthly_valid_years = (monthly_avail >= min_monthly_avail).all(axis=1)
            valid_mask &= valid_mask.index.isin(monthly_valid_years[monthly_valid_years].index)

        # return valid_mask[valid_mask].index
        self._valid_years = valid_mask[valid_mask].index

    def _compute_annual_availability(self) -> pd.Series:
        """Compute fraction of non-NaN values per year."""
        df = self.data.copy()
        expected = df['water_year'].value_counts().mode()[0]
        expected = max(expected, 365) # Just in case of time series with less than one year of data
        df = df.dropna(subset=self.data_columns)
        annual_counts = df.groupby('water_year').count().iloc[:, 0]
        self.availability = annual_counts / expected

    def _get_n_tot_years(self) -> Optional[int]:
        if 'min_availability' in self.criteria:
            return len(self.valid_years) #(self.availability >= self.criteria['min_availability']).sum()
        return None

    def _get_n_consecutive_years(self) -> Optional[int]:
        if 'min_availability' in self.criteria:
            if self.valid_years.empty:
                return 0
            sorted_years = self.valid_years.index.sort_values()
            diffs = sorted_years.to_series().diff().fillna(1)
            group = (diffs != 1).cumsum()
            return group.value_counts().max()
        return None

    @property 
    def valid_years(self): 
        return self._valid_years

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