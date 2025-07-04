
import pandas as pd 
import numpy as np
import itertools

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union

from .base import BaseSummary, EventBasedSummary
from .baseflow import lh

COMMON_KWARGS_DOC = """
    by_year : bool, optional
        Whether to group summaries by water year.
    rolling : int, optional
        Size of rolling window (in years) for summarization.
    center : bool, optional
        Whether to center the rolling window.
"""


def format_quantile(quantile: Union[float, list]) -> str:
    if isinstance(quantile, float): 
        quantile = [quantile]
    quantile_int = [int(round(q * 100)) for q in quantile]
    return [f"Q{q_int:02d}" for q_int in quantile_int]


def compute_cv(group): 
    """Coefficient of variation."""
    group = group.sort_index()
    q_std = group['Q'].std()
    q_mean = group['Q'].mean()
    if q_mean == 0:
        return np.nan 
    return q_std / q_mean


def compute_quantile(group, q):
    """Flow quantile."""
    return group['Q'].quantile(q)


def compute_mean(group):
    return group['Q'].mean() 


def compute_max(group):
    return group['Q'].max() 


def compute_min(group):
    return group['Q'].min() 


def compute_std(group): 
    return group['Q'].std()


def compute_iqr(group): 
    Q1 = group['Q'].quantile(0.25)
    Q3 = group['Q'].quantile(0.75)
    return Q3 - Q1


def compute_center_timing(group): 
    cum_flow = group['Q'].cumsum()
    total_flow = cum_flow.max()
    if total_flow == 0:
        return None  # Cannot compute for zero flow

    # Find the first date where cumulative flow >= 50% of total
    half_total = 0.5 * total_flow
    doy_50 = cum_flow[cum_flow >= half_total].index[0].dayofyear
    return doy_50


def compute_doymin(group): 
    return group['Q'].idxmin().dayofyear 


def compute_doymax(group): 
    return group['Q'].idxmax().dayofyear 


def compute_gini_coefficient(group):
    """
    Calculate the Gini coefficient of streamflow inequality within a group.
    
    Parameters:
    group (pd.DataFrame): Must contain a 'Q' column representing streamflow.
    
    Returns:
    float: Gini coefficient (between 0 and 1), or np.nan if not computable.
    """
    q = group['Q'].dropna()
    if len(q) == 0 or q.sum() == 0:
        return np.nan  # Avoid division by zero or empty group

    q_norm = q / q.sum()
    q_sorted = np.sort(q_norm)
    n = len(q_sorted)
    gini = (2 * np.sum(np.arange(1, n + 1) * q_sorted) - (n + 1)) / n
    return gini


def compute_skew(group):
    qmean = group['Q'].mean()
    q50 = group['Q'].quantile(0.5)
    return qmean / q50 if q50 > 0 else 0


def compute_rbi(group):
    """Richards-Baker Index."""
    group = group.sort_index()
    q_diff = group['Q'].diff().abs()
    total_q = group['Q'].sum()
    if total_q == 0 or np.isnan(total_q):
        return np.nan
    return q_diff.sum() / total_q


def compute_extreme_mean_flow(group, n=7, fun='min'):
    group = group.sort_index().copy()
    def custom_nanmean(window, n):
        # Check if the number of non-NaN values meets the min_periods requirement
        valid_values = window #[~np.isnan(window)]
        if len(valid_values) >= n:
            return np.nanmean(window)
        else:
            return np.nan  # Return NaN if min_periods is not met

    # Apply rolling mean
    group['QMEAN'] = (
        group['Q']
        .rolling(window=n, min_periods=1)
        .apply(custom_nanmean, args=(n,))
        .shift(-(n - 1))
    )

    # Find the extreme value
    if all(group['QMEAN'].isna()): 
        return pd.Series({f'{fun.upper()}{n}_TIME': pd.NaT, f'{fun.upper()}{n}': pd.NA}, name=None)

    if fun == 'min':
        idx = group['QMEAN'].idxmin()
    elif fun == 'max':
        idx = group['QMEAN'].idxmax()
    else:
        raise ValueError("fun must be 'min' or 'max'")

    row = group.loc[[idx], ['QMEAN']].copy()
    row = row.reset_index(drop=False)
    # Return a series
    row = row.rename(columns={'time': f'{fun.upper()}{n}_TIME', 'QMEAN': f'{fun.upper()}{n}'})
    row = row.squeeze()
    row.name = None
    return row


def compute_slope_fdc(group, lower_q=0.33, upper_q=0.66):
    """Slope of the flow duration curve."""
    if lower_q >= upper_q: 
        raise ValueError
    # qmean = group['Q'].mean() 
    # qlower = group['Q'].quantile(lower_q) / qmean
    # qupper = group['Q'].quantile(upper_q) / qmean
    qlower = group['Q'].quantile(lower_q) #/ qmean
    qupper = group['Q'].quantile(upper_q) #/ qmean

    # Determine if the fdc has a slope at this tage and return the
    # corresponding values
    if qlower == 0 and qupper == 0:
        return 0
    else:
        denominator = upper_q - lower_q
        # if qupper == 0 and not qlower == 0:
        #     # Negative slope [theoretically impossible?]
        #     return -np.log(qlower) / denominator
        if qupper > 0 and qlower == 0:
            return np.log(qupper) / denominator
        else:
            return (np.log(qupper) - np.log(qlower)) / denominator


def compute_bfi(group):
    """Baseflow index."""
    return group['Qb'].sum() / group['Q'].sum()


def compute_dvia(group): 
    q_avg = group['Q'].mean()         
    group = group.set_index('time')
    group_month = group['Q'].resample('MS').mean() 
    group_avail = group['Q'].resample('MS').count() / group['Q'].resample('MS').size()
    group_month = pd.DataFrame({'Q_mean': group_month, 'Q_month_avail': group_avail})
    group_month = group_month.groupby(group_month.index.month)['Q_mean'].mean()
    q_max = group_month.max()
    q_min = group_month.min()
    return (q_max - q_min) / q_avg


def compute_dvic(group): 
    q_avg = group['Q'].mean()         
    group = group.set_index('time')
    group_month = group['Q'].resample('MS').mean() 
    q_05 = group_month.min()
    q_95 = group_month.max()
    return (q_95 - q_05) / q_avg 


class _GSIM(BaseSummary): 

    def compute(self, annual=True, seasonal=False, monthly=False):
        data = self._get_grouped_data(self.data, by_year=True, rolling=False)
        duration = self._compute_duration(data)

        group = data.groupby('group')

        # Basic statistics
        indices = {
            'MEAN': _MeanFlow()._compute(data),
            'SD': _STDFlow()._compute(data),
            'IQR': _IQRFlow()._compute(data),
            'MIN': _MinimumFlow()._compute(data),
            'MAX': _MaximumFlow()._compute(data)
        }

        # Extreme 7-day flows
        min7 = _NDayFlowExtreme()._compute(data, n=7, fun='min')
        max7 = _NDayFlowExtreme()._compute(data, n=7, fun='max')

        indices.update({
            'MIN7': min7[['MIN7']],
            'MAX7': max7[['MAX7']],
        })

        # Only add quantiles for annual and seasonal values
        if annual or seasonal:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            indices['QUANTILES'] = _FlowQuantile()._compute(data, quantile=quantiles)

        # Other indices computed only for annual summaries
        if annual:
            indices.update({
                'CT': _CenterTiming()._compute(data),
                'DOYMIN': _MinimumFlowDOY()._compute(data), 
                'DOYMAX': _MaximumFlowDOY()._compute(data),
                'DOYMIN7': min7['MIN7_TIME'].dt.dayofyear.to_frame(name='DOYMIN7'),
                'DOYMAX7': max7['MAX7_TIME'].dt.dayofyear.to_frame(name='DOYMAX7'),
                'GINI': _GiniCoefficient()._compute(data)
            })

        result = pd.concat(indices.values(), axis=1)
        return pd.merge(result, duration, left_index=True, right_index=True)


# class _CAMELS(BaseSummary): 
#     def compute(self): 
#         data = self._get_grouped_data(self.data, by_year=False)
#         duration = self._compute_duration(data)

#         group = data.groupby('group')

#         indices = {}
#         indices['MEAN'] = group.apply(compute_mean).to_frame(name='MEAN')
#         indices['SFDC'] = group.apply(compute_slope_fdc).to_frame(name='SFDC')
#         indices['BFI'] = group.apply(compute_bfi).to_frame(name='BFI')
#         indices['Q5'] = group.apply(compute_quantile, q=[0.05]).to_frame(name='Q5')
#         indices['Q95'] = group.apply(compute_quantile, q=[0.95]).to_frame(name='Q95')
#         indices['HIGH_Q_FREQ']
#         indices['HIGH_Q_DUR']
#         indices['LOW_Q_FREQ']
#         indices['LOW_Q_DUR']
#         indices['ZERO_Q_FREQ']


class _CV(BaseSummary): 
    def _compute(self, data):
        return data.groupby('group').apply(compute_cv).to_frame(name='CV')

class _FlowQuantile(BaseSummary): 
    def _compute(self, data, quantile): 
        label = format_quantile(quantile)
        if len(label) == 1:
            label = label[0]
            result = data.groupby('group').apply(compute_quantile, q=quantile).apply(pd.Series)
        else:
            result = data.groupby('group').apply(compute_quantile, q=quantile)
            result.columns = label
        return result 

class _Skewness(BaseSummary):
    def _compute(self, data): 
        return data.groupby('group').apply(compute_skew).to_frame(name='Skew')

class _RichardsBakerIndex(BaseSummary):
    def _compute(self, data):  
        return data.groupby('group').apply(compute_rbi).to_frame(name='RBI')

class _MeanFlow(BaseSummary):
    def _compute(self, data):  
        return data.groupby('group')['Q'].mean().to_frame(name='QMEAN')

class _MaximumFlow(BaseSummary):
    def _compute(self, data):
        return data.groupby('group')['Q'].max().to_frame(name='QMAX')

class _MinimumFlow(BaseSummary):
    def _compute(self, data):
        return data.groupby('group')['Q'].min().to_frame(name='QMIN')

class _MinimumFlowDOY(BaseSummary): 
    def _compute(self, data): 
        return data.groupby('group').apply(compute_doymin).to_frame(name='DOYMIN')

class _MaximumFlowDOY(BaseSummary): 
    def _compute(self, data): 
        return data.groupby('group').apply(compute_doymax).to_frame(name='DOYMAX')

class _GiniCoefficient(BaseSummary): 
    def _compute(self, data): 
        return data.groupby('group').apply(compute_gini_coefficient).to_frame(name='GINI')

class _STDFlow(BaseSummary): 
    def _compute(self, data): 
        return data.groupby('group').apply(compute_std).to_frame(name='STD')

class _IQRFlow(BaseSummary): 
    def _compute(self, data): 
        return data.groupby('group').apply(compute_iqr).to_frame(name='IQR')

class _NDayFlowExtreme(BaseSummary):
    def _compute(self, data, n: int = 7, fun: str = 'min'):
        return data.groupby('group').apply(compute_extreme_mean_flow, n=n, fun=fun)

class _CenterTiming(BaseSummary): 
    def _compute(self, data):
        return data.groupby('group').apply(compute_center_timing).to_frame(name='CT')

class _SFDC(BaseSummary):
    def _compute(self, data):
        return data.groupby('group').apply(compute_slope_fdc).to_frame(name='SFDC')

class BFI(BaseSummary):
    def _compute(self, data, method='LH'):
        data = data.copy()
        data = data.dropna(subset='Q')
        Q = data['Q'].values
        if method.upper() == 'LH': 
            Qb = lh(Q)
        else:
            raise ValueError(f'Baseflow separation method {method} not recognised')

        data['Qb'] = Qb
        result = data.groupby('group').apply(compute_bfi).to_frame(name='BFI')
        return result 

class _DVIa(BaseSummary): 
    def _compute(self, data): 
        result = data.groupby('group').apply(compute_dvia).to_frame(name='DVIa')
        return result 

class _DVIc(BaseSummary):
    def _compute(self, data):
        return data.groupby('group').apply(compute_dvic).to_frame(name='DVIc')

def pot(vals, threshold): 
    return np.where(vals > threshold)[0] 

def lowflow(vals, threshold): 
    return np.where(vals < threshold)[0]

def highflow(value, threshold):
    return np.where(value > threshold)[0] #9 * median)[0]

class _POT(EventBasedSummary):
    def _compute(self, threshold: float, min_diff: int = 24) #, summarise=False, by_year=False, rolling=None, center=False):
        return self._flow_events(pot, min_diff, threshold=threshold)


class _LowFlowEvents(EventBasedSummary):
    def _compute(self, threshold, min_diff: int = 24):
        return self._flow_events(lowflow, min_diff, threshold=threshold)


class _HighFlowEvents(EventBasedSummary): 
    def _compute(self, threshold, min_diff: int = 24): #, summarise=False, by_year=False, rolling=None, center=False): 
        return self._flow_events(highflow, min_diff, threshold=threshold)


class _NoFlowEvents(EventBasedSummary): 
    def _compute(self, threshold: float = 0.001):
        data = self.data.copy()
        data['noflow'] = np.where(data['Q'] <= threshold, 1, 0)
        rle_no_flow = [(k, len(list(v))) for k, v in itertools.groupby(data['noflow'])]
        event_ids = [[i] * grp[1] for i, grp in enumerate(rle_no_flow)]
        event_ids = list(itertools.chain.from_iterable(event_ids))
        data['event_id'] = event_ids
        data = data.reset_index()
        events = data[data['noflow'] == 1].groupby('event_id').agg(
            water_year=('water_year', 'min'), # Take the water year of the event start
            event_start_time=('time', 'min'),
            event_end_time=('time', 'max'),
            event_duration=('time', lambda x: (x.max() - x.min())) #).days + 1)
        )
        if events.shape[0] == 0:
            events = None 

        return events 
    # def compute(self, threshold: float = 0.001, summarise: bool = False, by_year: bool = False, rolling: Optional[bool] = None, center: bool = False): 
    #     data = self.data.copy()
    #     data['noflow'] = np.where(data['Q'] <= threshold, 1, 0)
    #     rle_no_flow = [(k, len(list(v))) for k, v in itertools.groupby(data['noflow'])]
    #     event_ids = [[i] * grp[1] for i, grp in enumerate(rle_no_flow)]
    #     event_ids = list(itertools.chain.from_iterable(event_ids))
    #     data['event_id'] = event_ids
    #     data = data.reset_index()
    #     events = data[data['noflow'] == 1].groupby('event_id').agg(
    #         water_year=('water_year', 'min'), # Take the water year of the event start
    #         event_start_time=('time', 'min'),
    #         event_end_time=('time', 'max'),
    #         event_duration=('time', lambda x: (x.max() - x.min())) #).days + 1)
    #     )
    #     if events.shape[0] == 0:
    #         events = None 

    #     if summarise:
    #         summary = self._summarize_events(events, by_year=by_year, rolling=rolling, center=center)
    #         return events, summary
    #     else:
    #         return events

class _DryDownPeriod(EventBasedSummary): 
    
    def _summarize_events(self, events, by_year, rolling, center): 
        # First calculate the duration of each period
        grouped_data = self._get_grouped_data(self.data, by_year=by_year, rolling=rolling, center=center)
        duration = self._compute_duration(grouped_data)

        result = self._get_grouped_data(events, by_year=by_year, rolling=rolling, center=center)
        result = result.groupby('group').agg(
            n_events=('water_year', 'size'), # Take the water year of the event start
            mean_event_duration=('event_duration', 'mean'),
            total_duration=('event_duration', 'sum')
        )
        result = pd.merge(result, duration, left_index=True, right_index=True)

        # Compute frequency and mean duration in days 
        def get_event_frequency(row): 
            try:
                total_duration_days = row['total_duration'].days 
            except AttributeError: 
                total_duration_days = float(row['total_duration'])

            summary_period_duration_days = row['summary_period_duration'].days
            return total_duration_days / summary_period_duration_days
        
        def get_mean_event_duration_days(row): 
            n_events = row['n_events']
            if n_events == 0: 
                return 0.
            else: 
                return float(row['mean_event_duration'].days)

        result['frequency'] = result.apply(get_event_frequency, axis=1)
        result['mean_event_duration_days'] = result.apply(get_mean_event_duration_days, axis=1)
        return result

    def _compute(self, quantile: float = 0.25): #, summarise=False, by_year=False, rolling=None, center=False) -> float: 

        threshold = self.data['Q'].quantile(quantile)
        if threshold == 0.:
            return None #if not summarise else (None, None)

        pot_events = _POT(self.ts).compute(threshold=threshold)
        noflow_events = _NoFlowEvents(self.ts).compute(threshold=0.)
        if noflow_events is None or pot_events is None:
            return None #if not summarise else (None, None)

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
                water_year = self.data.loc[most_recent_high_end, 'water_year']
                dry_down_period = (n - most_recent_high_end).astype("timedelta64[D]").item()
                dry_down_events.append(pd.DataFrame({'water_year': [water_year], 'event_start_time': [most_recent_high_end], 'event_end_time': [n], 'event_duration': [dry_down_period]}))

        if len(dry_down_events) == 0: 
            return None #if not summarise else (None, None)

        dry_down_events = pd.concat(dry_down_events, axis=0).reset_index(drop=True)
        dry_down_events = dry_down_events.drop_duplicates(subset='event_start_time')
        return dry_down_events

    def compute(self, summarise=False, by_year=False, rolling=None, center=False, **kwargs):
        events = self._compute(**kwargs)
        if not events: 
            return None if not summarise else (None, None)

        if summarise:
            summary = self._summarize_events(events, by_year=by_year, rolling=rolling, center=center)
            return events, summary
        else:
            return events


class _NoFlowFraction(EventBasedSummary): 
    def compute(self, threshold: float = 0.001, by_year=False, rolling=None, center=False): 
        def noflow(vals, threshold): 
            return vals < threshold

        data = self._simple_flow_events(noflow, threshold=threshold)
        data = self._get_grouped_data(data, by_year=by_year, rolling=rolling, center=center)
        result = data.groupby('group').agg(
            event_duration=('event_duration', 'sum'),
            summary_period_duration=('timestep', 'sum')
        )
        result['noflow_fraction'] = result['event_duration'] / result['summary_period_duration']
        return result[['noflow_fraction']]


class _HighFlowFraction(EventBasedSummary): 
    def compute(self, threshold, by_year=False, rolling=None, center=False): 

        def highflow(vals, threshold): 
            return vals > threshold

        def compute_high_flow_fraction(threshold):
            data = self._simple_flow_events(highflow, threshold=threshold)
            data['event_volume_above_threshold'] = (data['Q'] - threshold) * data['event_duration'] * 86400.
            data = self._get_grouped_data(data, by_year=by_year, rolling=rolling, center=center)
            result = data.groupby('group').agg(
                event_duration=('event_duration', 'sum'),
                event_volume_above_threshold=('event_volume_above_threshold', 'sum'),
                summary_period_duration=('timestep', 'sum')
            )
            result['highflow_fraction'] = result['event_duration'] / result['summary_period_duration']
            return result
        
        if isinstance(threshold, dict): 
            result_list = []
            for threshold_name, threshold_value in threshold.items():
                result = compute_high_flow_fraction(threshold_value)
                result_list.append(result)
            result = pd.concat(result_list, axis=0, keys=threshold.keys(), names=('threshold', 'group'))
        else:
            result = compute_high_flow_fraction(threshold)

        return result[['highflow_fraction', 'event_duration', 'event_volume_above_threshold']]

class _LowFlowFraction(EventBasedSummary): 
    def compute(self, threshold, by_year=False, rolling=None, center=False): 
        def lowflow(vals, threshold): 
            return vals < threshold

        def compute_low_flow_fraction(threshold):
            data = self._simple_flow_events(lowflow, threshold=threshold)
            data = self._get_grouped_data(data, by_year=by_year, rolling=rolling, center=center)
            result = data.groupby('group').agg(
                event_duration=('event_duration', 'sum'),
                summary_period_duration=('timestep', 'sum')
            )
            result['lowflow_fraction'] = result['event_duration'] / result['summary_period_duration']
            return result
        
        if isinstance(threshold, dict): 
            result_list = []
            for threshold_name, threshold_value in threshold.items():
                result = compute_low_flow_fraction(threshold_value)
                result_list.append(result)
            result = pd.concat(result_list, axis=0, keys=threshold.keys(), names=('threshold', 'group'))
        else:
            result = compute_low_flow_fraction(threshold)
        return result[['lowflow_fraction', 'event_duration']]


summary_method_registry = {}

def register_summary_method(func):
    summary_method_registry[func.__name__] = func
    return func

@register_summary_method 
def gsim_annual(ts_or_df): 
    return _GSIM(ts_or_df).compute(annual=True)

@register_summary_method 
def gsim_seasonal(ts_or_df): 
    return _GSIM(ts_or_df).compute(seasonal=True)

@register_summary_method 
def gsim_monthly(ts_or_df): 
    return _GSIM(ts_or_df).compute(monthly=True)

@register_summary_method
def richards_baker_index(ts_or_df, **kwargs): 
    return _RichardsBakerIndex(ts_or_df).compute(**kwargs) 

@register_summary_method
def flow_quantile(ts_or_df, **kwargs):
    return _FlowQuantile(ts_or_df).compute(**kwargs)

@register_summary_method
def maximum_flow(ts_or_df, **kwargs):
    return _MaximumFlow(ts_or_df).compute(**kwargs)

@register_summary_method
def maximum_flow(ts_or_df, **kwargs):
    return _MinimumFlow(ts_or_df).compute(**kwargs)

@register_summary_method
def mean_flow(ts_or_df, **kwargs):
    return _MeanFlow(ts_or_df).compute(**kwargs)

@register_summary_method
def annual_maximum_flow(ts_or_df):
    return _MaximumFlow(ts_or_df).compute(by_year=True)

@register_summary_method
def n_day_low_flow_extreme(ts_or_df, n: int = 7, **kwargs):
    return _NDayFlowExtreme(ts_or_df).compute(n=n, fun='min', **kwargs)

@register_summary_method
def n_day_high_flow_extreme(ts_or_df, n: int = 7, **kwargs):
    return _NDayFlowExtreme(ts_or_df).compute(n=n, fun='max', **kwargs)

@register_summary_method
def peaks_over_threshold(ts_or_df, threshold: float, min_diff: int = 24, summarise=False, **kwargs):
    return _POT(ts_or_df).compute(threshold=threshold, min_diff=min_diff, summarise=summarise, **kwargs)

@register_summary_method
def no_flow_events(ts_or_df, threshold: float = 0.001, summarise=False, **kwargs): 
    return _NoFlowEvents(ts_or_df).compute(threshold=threshold, summarise=summarise, **kwargs)

@register_summary_method
def low_flow_events(ts_or_df, summarise=False, **kwargs):

    if hasattr(ts_or_df, "valid_data"):
        threshold = ts_or_df.valid_data['Q'].mean() * 0.2
    else:
        threshold = ts_or_df['Q'].mean() * 0.2 

    return _LowFlowEvents(ts_or_df).compute(threshold=threshold, summarise=summarise, **kwargs)

@register_summary_method
def high_flow_events(ts_or_df, summarise=False, **kwargs):
    """
    Identify high flow events in a time series.

    Parameters
    ----------
    ts_or_df : HydroTS or pd.DataFrame
        Input data containing discharge (Q).
    summarise : bool, optional
        Whether to return summary statistics.
    {kwargs_doc}

    Returns
    -------
    events : pd.DataFrame
        Event-level data.
    summary : pd.DataFrame, optional
        Summary statistics if summarise=True.
    """.format(kwargs_doc=COMMON_KWARGS_DOC)

    if hasattr(ts_or_df, "valid_data"):
        threshold = ts_or_df.data['Q'].median() * 9
    else:
        threshold = ts_or_df['Q'].median() * 9
    return _HighFlowEvents(ts_or_df).compute(threshold=threshold, summarise=summarise, **kwargs)

@register_summary_method
def dry_down_period(ts_or_df, quantile: float = 0.25, summarise=False, **kwargs): 
    return _DryDownPeriod(ts_or_df).compute(quantile=quantile, summarise=summarise, **kwargs)

@register_summary_method
def coefficient_of_variation(ts_or_df, **kwargs):
    return _CV(ts_or_df).compute(**kwargs)

@register_summary_method
def skewness(ts_or_df, **kwargs): 
    return _Skewness(ts_or_df).compute(**kwargs)

@register_summary_method
def slope_flow_duration_curve(ts_or_df, **kwargs): 
    return _SFDC(ts_or_df).compute(**kwargs)

@register_summary_method
def baseflow_index(ts_or_df, method='LH', **kwargs): 
    return BFI(ts_or_df).compute(method=method, **kwargs)

@register_summary_method 
def no_flow_fraction(ts_or_df, threshold, **kwargs):
    return _NoFlowFraction(ts_or_df).compute(threshold=threshold, **kwargs)

@register_summary_method 
def high_flow_fraction(ts_or_df, threshold=None, **kwargs):
    if not threshold:
        Q = ts_or_df.valid_data['Q'] if hasattr(ts_or_df, "valid_data") else ts_or_df['Q']
        threshold = Q.median() * 9.
    return _HighFlowFraction(ts_or_df).compute(threshold=threshold, **kwargs)

@register_summary_method 
def low_flow_fraction(ts_or_df, threshold=None, **kwargs):
    if not threshold:
        Q = ts_or_df.valid_data['Q'] if hasattr(ts_or_df, "valid_data") else ts_or_df['Q']
        threshold = Q.mean() * 0.2
    return _LowFlowFraction(ts_or_df).compute(threshold=threshold, **kwargs)

@register_summary_method 
def discharge_variability_index(ts_or_df, **kwargs): 
    return _DVIa(ts_or_df).compute(**kwargs)

@register_summary_method 
def cumulative_discharge_variability_index(ts_or_df, **kwargs): 
    return _DVIc(ts_or_df).compute(**kwargs)

# FIXME this currently makes assumption that timeseries has daily resolution
class TSSummary:

    _custom_summary_functions = {}

    def __init__(self, ts: "HydroTS"): 
        self.ts = ts

    # @staticmethod
    # def _circular_statistics(dates, hemisphere="Northern"):
    #     # Function to compute circular statistics (mean direction and regularity)
    #     day_of_year = dates.dt.dayofyear
    #     angles = 2 * np.pi * day_of_year / 365
    #     mean_cos = np.cos(angles).mean()
    #     mean_sin = np.sin(angles).mean()
        
    #     theta = np.arctan2(mean_sin, mean_cos)
        
    #     if hemisphere == "Southern":
    #         if any((day_of_year >= 182) & (day_of_year <= 365)):
    #             theta -= np.pi
    #         elif any((day_of_year >= 1) & (day_of_year <= 181)):
    #             theta += np.pi
        
    #     if theta < 0:
    #         theta += 2 * np.pi
        
    #     r = np.sqrt(mean_cos**2 + mean_sin**2)
    #     return theta, r

    # @classmethod
    # def register_summary_function(cls, name: str, func):
    #     """Register a new summary function dynamically."""
    #     cls._custom_summary_functions[name] = func

    # def __getattr__(self, name):
    #     """Dynamically return callable wrappers for custom summary functions."""
    #     if name in self._custom_summary_functions:
    #         # If a registered summary function is accessed, return a callable wrapper
    #         return self.SummaryFunctionWrapper(self._custom_summary_functions[name], self.ts)
    #     else:
    #         raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # class SummaryFunctionWrapper:
    #     def __init__(self, func, ts):
    #         self.func = func
    #         self.ts = ts
        
    #     def __call__(self, *args, **kwargs):
    #         """Allow the wrapper to be called like the original function."""
    #         # Call the function with self.hydrots as the first argument
    #         return self.func(self.ts, *args, **kwargs)

def make_method(func):
    def method(self, *args, **kwargs):
        return func(self.ts, *args, **kwargs)
    method.__name__ = func.__name__
    method.__doc__ = func.__doc__
    return method

for name, func in summary_method_registry.items():
    setattr(TSSummary, name, make_method(func))

# def _add_summary_method(func, name=None):
#     """Attach a function to TSSummary as a method."""
#     method_name = name or func.__name__

#     def method(self, *args, **kwargs):
#         return func(self.ts, *args, **kwargs)

#     setattr(TSSummary, method_name, method)
# _add_summary_method(richards_baker_index)
# _add_summary_method(maximum_flow)
# _add_summary_method(annual_maximum_flow)
# _add_summary_method(n_day_low_flow_extreme)
# _add_summary_method(n_day_high_flow_extreme)
# _add_summary_method(peaks_over_threshold)
# _add_summary_method(no_flow_events)
# _add_summary_method(low_flow_events)
# _add_summary_method(high_flow_events)
# _add_summary_method(dry_down_period)
# _add_summary_method(coefficient_of_variation)

# NOT USED:

# def circular_stats(dates, hemisphere="Northern"):
#     # Function to compute circular statistics (mean direction and regularity)
#     day_of_year = dates.dt.dayofyear
#     angles = 2 * np.pi * day_of_year / 365
#     mean_cos = np.cos(angles).mean()
#     mean_sin = np.sin(angles).mean()
#     theta = np.arctan2(mean_sin, mean_cos)
#     if hemisphere == "Southern":
#         if any((day_of_year >= 182) & (day_of_year <= 365)):
#             theta -= np.pi
#         elif any((day_of_year >= 1) & (day_of_year <= 181)):
#             theta += np.pi
#     if theta < 0:
#         theta += 2 * np.pi
#     r = np.sqrt(mean_cos**2 + mean_sin**2)
#     # return {'theta': theta, 'regularity': r}
#     return theta, r

# def local_water_year(df, wettest=True):
#     df = df.copy()
#     df = df.set_index('date')
#     df = df.rolling('30d', min_periods=7).mean()
#     df = df.reset_index()
#     df['day'] = df['date'].dt.dayofyear
#     df = df.groupby('day')['Q'].mean()
#     df = df.loc[1:365]
#     if wettest:
#         water_year_start = df.idxmax()
#     else:
#         water_year_start = df.idxmin()
#     return water_year_start

# class MaxLowFlowDuration(LowFlowMetric): 
#     def compute(self, quantile=0.9):
#         df = self.data.copy()
#         df = df[df.groupby('water_year').transform('size') >= 365]
#         q_str = format_quantile(quantile)
#         threshold = df['Q'].quantile((1-quantile))
#         df['below'] = df['Q'] <= threshold 
#         df['below'] = df['below'].astype(int)
#         years = df['water_year'].unique()
#         df_list = []
#         for year in years: 
#             df_year = df[df['water_year'] == year].copy()
#             # Calculate the lengths of consecutive periods where 'below' == 1
#             df_year['below_shifted'] = df_year['below'].shift(1, fill_value=0)
#             df_year['period'] = (df_year['below'] != df_year['below_shifted']).cumsum()
#             # Filter only the periods where the value is below the threshold
#             below_periods = df_year[df_year['below'] == 1].groupby('period').size()
#             # If there are any periods, find the maximum length of consecutive days below threshold
#             if not below_periods.empty:
#                 max_consecutive_days = below_periods.max()
#                 # Find the start time of the longest below-threshold period
#                 longest_period = below_periods.idxmax()
#                 start_of_longest_period = df_year[df_year['period'] == longest_period]['time'].min()
#                 df_list.append(pd.DataFrame.from_dict({
#                     'water_year': [year],
#                     'longest_period_below_' + q_str: [max_consecutive_days], 
#                     'longest_period_below_' + q_str + '_start_time': [start_of_longest_period], 
#                 }))

#         out = pd.concat(df_list, axis=0).reset_index(drop=True)
#         return out

# class MaxLowFlowDeficit(LowFlowMetric):
#     def compute(self, quantile=0.9):
#         df = self.data.copy()
#         q_str = format_quantile(quantile)
#         threshold = df['Q'].quantile((1-quantile))
#         df['below'] = df['Q'] <= threshold 
#         df['below'] = df['below'].astype(int)
#         df['deficit'] = df['Q'].apply(lambda x: threshold - x if x <= threshold else 0)
#         df = df[df.groupby('water_year').transform('size') >= 365]
#         years = df['water_year'].unique()
#         df_list = []
#         for year in years: 
#             df_year = df[df['water_year'] == year].copy()
#             # Calculate the lengths of consecutive periods where 'below' == 1
#             df_year['below_shifted'] = df_year['below'].shift(1, fill_value=0)
#             df_year['period'] = (df_year['below'] != df_year['below_shifted']).cumsum()
#             # Filter only the periods where the value is below the threshold
#             below_periods = df_year[df_year['below'] == 1].groupby('period')['deficit'].sum()
#             # If there are any periods, find the maximum length of consecutive days below threshold
#             if not below_periods.empty:
#                 max_deficit = below_periods.max()
#                 # Find the start time of the longest below-threshold period
#                 max_deficit_period = below_periods.idxmax()
#                 start_of_max_deficit_period = df_year[df_year['period'] == max_deficit_period]['time'].min()
#                 df_list.append(pd.DataFrame.from_dict({
#                     'water_year': [year],
#                     'max_deficit_below_' + q_str: [max_deficit], 
#                     'max_deficit_below_' + q_str + '_start_time': [start_of_max_deficit_period]
#                 }))

#         out = pd.concat(df_list, axis=0).reset_index(drop=True)
#         return out

# class NoFlowEventDuration(LowFlowMetric):
#     def compute(self): 
#         df = df.copy()
#         # Function to calculate the no-flow duration recurrence period
#         df['noflow'] = np.where(df['Q'] <= 0.001, 1, 0)
#         # rle_no_flow = df['noflow'].values
#         rle_no_flow = [(k, len(list(v))) for k, v in itertools.groupby(df['noflow'])]
#         event_ids = [[i] * grp[1] for i, grp in enumerate(rle_no_flow)]
#         event_ids = list(itertools.chain.from_iterable(event_ids))
#         df['event_id'] = event_ids
#         no_flow_events = df[df['noflow'] == 1].groupby('event_id').agg(
#             start_time=('time', 'min'),
#             end_time=('time', 'max'),
#             duration=('time', lambda x: (x.max() - x.min()).days + 1)
#         )
#         # no_flow_events['end_year'] = no_flow_events['end_time'].dt.year
#         # max_durations = no_flow_events.groupby('end_year')['duration'].max()
#         # D80 = max_durations.quantile(quantile)
#         # return D80
#         return no_flow_events

# class NoFlowFrequency(LowFlowMetric):
#     def compute(self): 
#         df = df.copy()
#         # Function to calculate no-flow frequency
#         df['noflow'] = np.where(df['Q'] <= 0.001, 1, 0)
#         rle_no_flow = [(k, len(list(v))) for k, v in itertools.groupby(df['noflow'])]
#         event_ids = [[i] * grp[1] for i, grp in enumerate(rle_no_flow)]
#         event_ids = list(itertools.chain.from_iterable(event_ids))
#         df['event_id'] = event_ids
#         df['year'] = df['time'].dt.year
#         no_flow_events = df[df['noflow'] == 1].groupby(['year', 'event_id']).agg(
#             start_time=('time', 'min'),
#             end_time=('time', 'max'),
#             duration=('time', lambda x: (x.max() - x.min()).days + 1)
#         )
#         yearly_events = no_flow_events.groupby('year').size().reset_index(name='n')
#         all_years = df['year'].drop_duplicates().sort_values()
#         yearly_events = pd.merge(pd.DataFrame({'year': all_years}), yearly_events, how='left', on='year').fillna(0)
#         yearly_events['n'] = yearly_events['n'].astype(int)
#         return yearly_events

# class NoFlowTiming(LowFlowMetric):
#     # Function to calculate timing of no-flow events
#     # TODO not looked in detail at this function
#     def compute(self):
#         df = self.data.copy()
#         df['noflow'] = np.where(df['Q'] <= 0.001, 1, 0)
#         no_flow_dates = df[df['noflow'] == 1]['time']
#         return circular_stats(no_flow_dates)

# class BaseflowIndexDecadal(LowFlowMetric): 
#     # TODO not sure what the precedent is for this method?
#     def compute(self, nyears=10):
#         df = self.data.copy()
#         df_base = baseflow.separation(df[['time', 'Q']].set_index('time'), method = 'LH')
#         df_base = df_base['LH'].reset_index().rename({'Q': 'Qb'}, axis=1)
#         df_base = pd.merge(df, df_base, how='left', on='time')
#         df_base['month'] = df_base['time'].dt.month
#         years = df_base['water_year'].unique()
#         bfi_year_list = []
#         bfi_month_list = []
#         for start_year in years:
#             decade_range = (df_base['water_year'] >= start_year) & (df_base['water_year'] <= start_year + (nyears - 1))
#             df_base_decade = df_base[decade_range].copy()
#             df_base_decade['decade_start_year'] = start_year 
#             bfi_year = df_base_decade.groupby('decade_start_year').apply(lambda x: x['Qb'].sum() / x['Q'].sum(), include_groups=False).reset_index(name='bfi')
#             bfi_month = df_base_decade.groupby(['decade_start_year', 'month']).apply(lambda x: x['Qb'].sum() / x['Q'].sum(), include_groups=False).reset_index(name='bfi')
#             bfi_year_list.append(bfi_year)
#             bfi_month_list.append(bfi_month)

#         bfi_year = pd.concat(bfi_year_list).reset_index(drop=True).rename({'bfi': 'bfi_year'}, axis=1)
#         bfi_month = pd.concat(bfi_month_list).reset_index(drop=True)
#         bfi_month = bfi_month.pivot(index='decade_start_year', columns='month', values='bfi')
#         bfi_month.columns = ['bfi_month_' + str(m) for m in bfi_month.columns]
#         bfi = pd.merge(bfi_year, bfi_month, how='left', on='decade_start_year')
#         bfi = bfi.rename({'decade_start_year': 'water_year'}, axis=1)
#         return bfi

# class SeasonalPredictability(LowFlowMetric):
#     # Function to calculate seasonal predictability of dry periods
#     # TODO find precedent for this approach
#     def compute(self): 
#         df = self.data.copy()
#         df['noflow'] = np.where(df['Q'] <= 0.001, 1, 0)
#         df['month'] = df['time'].dt.month
#         df['year'] = df['time'].dt.year
#         monthly_no_flow = df.groupby('month')['noflow'].mean()
        
#         rolling_sums = []
#         for i in range(12):
#             indices = np.arange(i, i + 6) % 12
#             rolling_sums.append({
#                 'start_month': monthly_no_flow.index[i],
#                 'end_month': monthly_no_flow.index[indices[-1]],
#                 'total_no_flow_freq': monthly_no_flow.iloc[indices].sum()
#             })
        
#         rolling_sums_df = pd.DataFrame(rolling_sums)
#         driest_six_months = rolling_sums_df.loc[rolling_sums_df['total_no_flow_freq'].idxmax()]
#         wettest_six_months = rolling_sums_df.loc[rolling_sums_df['total_no_flow_freq'].idxmin()]
        
#         def average_monthly_zero_flow(df, start_month):
#             months = np.arange(start_month, start_month + 6) % 12 + 1
#             mean_F06 = df[df['month'].isin(months)].groupby('year')['noflow'].mean().mean()
#             return mean_F06
        
#         Fd_dry = average_monthly_zero_flow(df, driest_six_months['start_month'])
#         Fd_wet = average_monthly_zero_flow(df, wettest_six_months['start_month'])
#         SD6 = 1 - Fd_wet / Fd_dry
#         return SD6

# class SeasonalRecessionTimescale(LowFlowMetric):
#     # Function to calculate seasonal recession timescale (tentative)
#     # TODO find precedent for this method 
#     def compute(self):
#         df = self.data.copy()
#         df['day'] = df['date'].dt.dayofyear
#         daily_mean = df.groupby('day')['Q'].mean().reset_index()
#         daily_mean = pd.concat([daily_mean, daily_mean.head(30)], ignore_index=True)
#         daily_mean['Q_30d'] = daily_mean['Q'].rolling(window=30, min_periods=1).mean()
        
#         max_idx = daily_mean['Q_30d'].idxmax()
#         daily_mean = pd.concat([daily_mean[max_idx:], daily_mean[:max_idx]], ignore_index=True)
        
#         min_idx = daily_mean['Q_30d'].idxmin()
#         q50 = daily_mean['Q_30d'].quantile(0.5)
#         q90 = daily_mean['Q_30d'].quantile(0.9)
        
#         Drec = len(daily_mean[(daily_mean['Q_30d'] >= q50) & (daily_mean['Q_30d'] <= q90)])
#         return Drec

# class ConcavityIndex(Metric):
#     # TODO find precedent for this method 
#     def compute(self):
#         df = self.data.copy()
#         # Function to calculate concavity index
#         Q = df['Q'].dropna()
#         q = np.quantile(Q, [0.01, 0.1, 0.99])
#         IC = (q[1] - q[2]) / (q[0] - q[2])
#         return IC

# class BaseflowIndex(Metric): 
#     # TODO per year / rolling?
#     def compute(self, method = 'UKIH'):
#         df = self.data.copy()
#         df = df[['time', 'Q']].set_index('time')
#         with contextlib.redirect_stdout(io.StringIO()):
#             bfi = baseflow.separation(df, method='UKIH')
#         bfi = bfi['UKIH']
#         Qt = df['Q'].sum()
#         Qb = bfi['Q'].sum()
#         bfi = Qb / Qt 
#         return bfi

# class RunoffEventDuration(Metric): 
#     # TODO find precedent for this method
#     def compute(self, n_peaks = 5): 
#         df = self.data.copy()
#         df = df[['time', 'Q']].set_index('time')
#         with contextlib.redirect_stdout(io.StringIO()):
#             bfi = baseflow.separation(df, method='UKIH')
#         bfi = bfi['UKIH']
#         bfi = bfi.fillna(0)
#         bfi = bfi.rename({'Q': 'Qb'}, axis=1)
#         df = pd.merge(df, bfi, left_index=True, right_index=True).reset_index()
#         df['Qs'] = df['Q'] - df['Qb']
#         df['runoff'] = (df['Qs'] >= 0.001).astype(int)
#         rle_runoff = [(k, len(list(v))) for k, v in itertools.groupby(df['runoff'])]
#         event_ids = [[i] * grp[1] for i, grp in enumerate(rle_runoff)]
#         event_ids = list(itertools.chain.from_iterable(event_ids))
#         df['event_id'] = event_ids
#         df['year'] = df['time'].dt.year
#         df = df[df['runoff'] == 1]
#         # Select the most severe runoff events using POT approach
#         df_peak = df.groupby(['event_id'])['Q'].max() 
#         # Calculate the target number of peaks
#         n_years = len(df['time'].dt.year.unique())
#         n_target_peaks = n_years * n_peaks
#         n_target_peaks = min(n_target_peaks, df_peak.shape[0])
#         # Sort the flood maxima in descending order
#         sorted_peaks = df_peak.sort_values(ascending=False).reset_index(drop=True)
#         # Set the threshold to select the top 'target_peaks' maximums
#         threshold_value = sorted_peaks.iloc[n_target_peaks - 1]
#         # Identify the peaks over threshold and limit dataframe to these events
#         event_ids = list(df_peak[df_peak > threshold_value].index)
#         df = df[df['event_id'].isin(event_ids)]
#         # Now compute the event duration
#         runoff_events = df['event_id'].unique()
#         Dr = []
#         for event in runoff_events: 
#             df_event = df[df['event_id'] == event]
#             Q_max_index = df_event['Qs'].idxmax()
#             Q_max = df_event['Qs'].max()
#             df_event = df_event.loc[Q_max_index:]
#             event_duration = len([i for i in range(df_event.shape[0]) if df_event['Qs'].values[i] > (Q_max / 2)])
#             Dr.append(event_duration)

#         return Dr

# # Function to calculate seasonal predictability of dry periods
# def seasonal_predictability_flow(df, n=6):
#     df = df.copy()
#     # df['noflow'] = np.where(df['Q'] <= 0.001, 1, 0)
#     df['month'] = df['date'].dt.month
#     df['year'] = df['date'].dt.year
#     monthly_no_flow = df.groupby('month')['Q'].mean()
#     rolling_sums = []
#     for i in range(12):
#         indices = np.arange(i, i + n) % 12
#         rolling_sums.append({
#             'start_month': monthly_no_flow.index[i],
#             'end_month': monthly_no_flow.index[indices[-1]],
#             'total_no_flow_freq': monthly_no_flow.iloc[indices].sum()
#         })
#     rolling_sums_df = pd.DataFrame(rolling_sums)
#     driest_six_months = rolling_sums_df.loc[rolling_sums_df['total_no_flow_freq'].idxmax()]
#     wettest_six_months = rolling_sums_df.loc[rolling_sums_df['total_no_flow_freq'].idxmin()]
#     def average_monthly_flow(df, start_month):
#         months = np.arange(start_month, start_month + n) % 12 + 1
#         mean_F0 = df[df['month'].isin(months)].groupby('year')['Q'].mean().mean()
#         return mean_F0
#     Fd_dry = average_monthly_flow(df, driest_six_months['start_month'])
#     Fd_wet = average_monthly_flow(df, wettest_six_months['start_month'])
#     SD = 1 - Fd_wet / Fd_dry
#     return SD