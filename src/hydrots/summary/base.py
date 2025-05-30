
import pandas as pd 
import numpy as np

from typing import Union

class BaseSummary: 

    def __init__(self, ts_or_df: Union["HydroTS", pd.DataFrame], discharge_col=None): 
        # if isinstance(ts_or_df, HydroTS):
        if hasattr(ts_or_df, "valid_data"):  # Likely a HydroTS object
            self.ts = ts_or_df
            self.data = ts_or_df.valid_data.copy()
        elif isinstance(ts_or_df, pd.DataFrame):
            self.ts = None
            self.data = ts_or_df.copy()
        else:
            raise TypeError("Input must be a HydroTS object or a pandas DataFrame.")

    def _compute_duration(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the duration of each group based on the 'time' column.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with a 'time' column and grouping labels.
        group_col : str, optional
            Name of the column to group by (default is 'group').

        Returns
        -------
        pd.DataFrame
            A DataFrame with group index and 'duration_days'.
        """
        return (
            data.groupby('group')['time']
            .agg(summary_period_duration=lambda x: x.max() - x.min() + (x.iloc[1] - x.iloc[0])) # FIXME should be able to get time resolution from HydroTS object
        )

    def _get_grouped_data(self, data, by_year=False, rolling=None, center=False):

        years = sorted(self.ts.valid_years)

        # Remove time index, because this won't work with overlapping groups (i.e. when `rolling` is set)
        data = data.reset_index(drop=False)
        if by_year:
            data['group'] = data['water_year']
            return data

        elif rolling is not None:
            group_datas = []

            for i in range(len(years) - rolling + 1):
                window_years = years[i:i + rolling]
                
                if center:
                    center_idx = i + rolling // 2
                    group_label = str(years[center_idx])
                else:
                    group_label = f"{window_years[0]}–{window_years[-1]}"
                
                window_data = data[data['water_year'].isin(window_years)].copy()
                window_data['group'] = group_label
                group_datas.append(window_data)

            return pd.concat(group_datas, ignore_index=True)

        else:
            data['group'] = f'{years[0]}-{years[-1]}'
            return data

    def compute(self): 
        raise NotImplementedError


class EventBasedSummary(BaseSummary): 
    def _compute_mean_deficit(self, events): 
        pass

    def _simple_flow_events(self, event_condition, **kwargs): 
        data = self.data.copy()
        data = data.reset_index(drop=False)
        data['time_window'] = data['time'].diff().shift(-1)
        vals = self.data['Q'].values 
        data['is_event'] = event_condition(vals) #, **kwargs)
        data['event_duration'] = data['is_event'] * data['time_window'] # timedelta64
        data = data.set_index('time')
        return data[['time', 'event_duration']]

    def _flow_events(self, event_condition, min_diff: int = 24, **kwargs) -> pd.DataFrame: 
        
        vals = self.data['Q'].values 

        # Indices where data exceeds threshold
        data_index = event_condition(vals, **kwargs)
        # data_index = np.where(vals > threshold)[0]

        if len(data_index) == 0:
            return None

        # Time difference between consecutive exceedance indices
        times = self.data.index[data_index]
        diff_hours = np.diff(times).astype('timedelta64[h]')
        
        # Where difference is greater than min_diff, it's a new event
        sep_index = np.where(diff_hours > min_diff)[0]
        
        start_index = np.concatenate(([data_index[0]], data_index[sep_index + 1]))
        end_index = np.concatenate((data_index[sep_index], [data_index[-1]]))

        start_times = self.data.index[start_index].values
        end_times = self.data.index[end_index].values
        duration = (end_times - start_times).astype('timedelta64[D]')
        water_year = self.data.iloc[start_index]['water_year'].values # i.e. take the water year at the start of the event

        n_events = len(start_index)

        if n_events == 0:
            return None

        return pd.DataFrame({'water_year': water_year, 'event_start_time': start_times, 'event_end_time': end_times, 'event_duration': duration})

    def _summarize_events(self, events, by_year, rolling, center): 
        # First calculate the duration of each period
        grouped_data = self._get_grouped_data(self.data, by_year=by_year, rolling=rolling, center=center)
        duration = self._compute_duration(grouped_data)

        # Now aggregate the events 
        if isinstance(events, pd.DataFrame):
            result = events.groupby('water_year').agg(n_events=('water_year', 'size'), mean_duration=('event_duration', 'mean'), total_duration=('event_duration', 'sum'))
            result = pd.DataFrame(result, index=self.ts.valid_years)
            result['n_events'] = result['n_events'].fillna(0).astype(int)
        else:
            result = pd.DataFrame({'water_year': self.ts.valid_years, 'n_events': 0, 'mean_duration': None, 'total_duration': None})

        result = self._get_grouped_data(result, by_year=by_year, rolling=rolling, center=center)
        result = result.groupby('group').agg(
            n_events=('n_events', 'sum'), # Take the water year of the event start
            mean_event_duration=('mean_duration', 'mean'),
            total_duration=('total_duration', 'sum')
        )
        result = pd.merge(result, duration, left_index=True, right_index=True)
        # result['frequency'] = result['total_duration'] / result['summary_period_duration']
        return result