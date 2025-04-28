
import pandas as pd 

from typing import Optional, Dict, List

class TSValidator:

    def __init__(self, 
                 data: pd.DataFrame, 
                 data_columns: List[str],
                 freq: str,
                 min_tot_years: Optional[int] = None,
                 min_consecutive_years: Optional[int] = None,
                 min_availability: Optional[float] = None):
        self.data = data  # Expect data to be formatted with datetime index and single/multi column
        self.data_columns = data_columns
        self.freq = freq
        self.criteria: Dict[str, Optional[float | int]] = {}
        self._set_validity_criteria(min_tot_years, min_consecutive_years, min_availability)
        self._compute_annual_availability()

    def _set_validity_criteria(self,
                               min_tot_years: Optional[int],
                               min_consecutive_years: Optional[int],
                               min_availability: Optional[float]):

        new_values = {
            'min_tot_years': min_tot_years,
            'min_consecutive_years': min_consecutive_years,
            'min_availability': min_availability
        }

        # Only store non-None values
        self.criteria = {k: v for k, v in new_values.items() if v is not None}

        # Check consistency
        if ((self.criteria.get('min_tot_years') is not None or self.criteria.get('min_consecutive_years') is not None)
            and self.criteria.get('min_availability') is None):
            raise ValueError(
                "`min_availability` must be specified when using `min_tot_years` or `min_consecutive_years`"
            )

    def update_criteria(self, **kwargs):
        self._set_validity_criteria(
            kwargs.get('min_tot_years', self.criteria.get('min_tot_years')),
            kwargs.get('min_consecutive_years', self.criteria.get('min_consecutive_years')),
            kwargs.get('min_availability', self.criteria.get('min_availability'))
        )
        return self

    @property 
    def valid_years(self): 
        if 'min_availability' in self.criteria: 
            valid = self.availability >= self.criteria['min_availability']
            return valid[valid].index 
        else: 
            return None 

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
            return (self.availability >= self.criteria['min_availability']).sum()
        return None

    def _get_n_consecutive_years(self) -> Optional[int]:
        if 'min_availability' in self.criteria:
            valid_years = self.availability[self.availability >= self.criteria['min_availability']]
            if valid_years.empty:
                return 0
            sorted_years = valid_years.index.sort_values()
            diffs = sorted_years.to_series().diff().fillna(1)
            group = (diffs != 1).cumsum()
            return group.value_counts().max()
        return None

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