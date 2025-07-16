
import pandas as pd 
import numpy as np
import itertools
import copy

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union

# from .base import BaseSummary, EventBasedSummary
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


def compute_seasonality(group): 
    # https://doi.org/10.5194/hess-29-2851-2025
    # Here we calculate the concentration, R, described in Berghuijs et al. (2025)
    group = group.sort_index()
    dates = group['time'] #pd.Series(group.index)
    group['day'] = round(365*dates.dt.dayofyear / dates.dt.is_leap_year.apply(lambda x: 366 if x else 365)).values
    group['day_scaled'] = (group['day'] / 365) * 2 * np.pi
    group['cos_day_scaled'] = np.cos(group['day_scaled'])
    group['sin_day_scaled'] = np.sin(group['day_scaled'])
    mean_x_coord = (group['cos_day_scaled'] * group['Q']).sum() / group['Q'].sum()
    mean_y_coord = (group['sin_day_scaled'] * group['Q']).sum() / group['Q'].sum()
    R =  (mean_x_coord**2 + mean_y_coord **2)**0.5  
    return R


def compute_cv(group): 
    """Coefficient of variation."""
    group = group.sort_index()
    q_std = group['Q'].std()
    q_mean = group['Q'].mean()
    if q_mean == 0:
        return np.nan 
    return q_std / q_mean


def compute_qcv(group): 
    """Quartile-based coefficient of variation."""
    q_iqr = compute_iqr(group)
    q_median = group['Q'].median()
    if q_median == 0: 
        return np.nan 
    return q_iqr / q_median 


def compute_quantile(group, q):
    """Flow quantile."""
    return group['Q'].quantile(q)


def compute_autocorrelation(group, lag):
    """Lag-x autocorrelation of flow."""
    x = group['Q']
    x_lagged = x.shift(lag)
    valid = x.notna() & x_lagged.notna()
    if valid.sum() == 0:
        return np.nan  # Not enough valid pairs

    x_valid = x[valid]
    x_lagged_valid = x_lagged[valid]
    return x_valid.corr(x_lagged_valid)


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
    group = group.copy() 
    if 'time' in group.columns:
        group = group.set_index('time')

    group = group.sort_index()
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
    Q_sum = group['Q'].sum()
    Qb_sum = group['Qb'].sum() 
    return Qb_sum / Q_sum if Q_sum > 0 else np.nan


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

    GSIM_SEASONS = ['MAM', 'JJA', 'SON', 'DJF']

    def compute(self, annual=True, seasonal=False, monthly=False):

        self.ts.update_water_year(use_water_year=False)
        data = self._format_data(by_season=seasonal, seasons=self.GSIM_SEASONS, by_month=monthly)
        if annual:
            data = self._get_grouped_data(data, by_year=True)
        if seasonal:
            data = self._get_grouped_data(data, by_year=True, by_season=True, seasons=self.GSIM_SEASONS)
        if monthly: 
            data = self._get_grouped_data(data, by_year=True, by_month=True) 

        duration = self._compute_duration(data)

        result = {} 
        result['MEAN'] = _MeanFlow()._compute(data) 
        result['STD'] = _STDFlow()._compute(data)
        result['IQR'] = _IQRFlow()._compute(data)
        result['MIN'] = _MinimumFlow()._compute(data)
        result['MAX'] = _MaximumFlow()._compute(data)

        # Only add quantiles for annual and seasonal values
        if annual or seasonal:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            result['QUANTILES'] = _FlowQuantile()._compute(data, quantile=quantiles)

        # Other indices computed only for annual summaries
        if annual:
            # result['CT'] = _CenterTiming()._compute(data).to_frame('CT')
            min7 = _NDayFlowExtreme()._compute(data, n=7, fun='min')
            max7 = _NDayFlowExtreme()._compute(data, n=7, fun='max')
            result['MIN7'] = min7[['MIN7']]
            result['MAX7'] = max7[['MAX7']] 
            result['DOYMIN'] = min7['MIN7_TIME'].dt.dayofyear.to_frame(name='DOYMIN7')
            result['DOYMAX'] = max7['MAX7_TIME'].dt.dayofyear.to_frame(name='DOYMAX7')
            result['GINI'] = _GiniCoefficient()._compute(data)

        result = pd.concat(result.values(), axis=1)
        return pd.merge(result, duration, left_index=True, right_index=True)


class _StreamflowIndices(BaseSummary): 

    def compute(self, by_year=False, rolling=None):

        # if self.data is None:
        #     raise ValueError("No data was provided. You must supply data at initialization or call _compute directly.")

        data = self._format_data() 

        # Compute indices over full record
        data = self._get_grouped_data(data, by_year=by_year, rolling=rolling)
        duration = self._compute_duration(data)

        result = {}

        result['MEAN'] = _MeanFlow()._compute(data)
        result['STD'] = _STDFlow()._compute(data)
        result['IQR'] = _IQRFlow()._compute(data)
        result['QUANTILES'] = _FlowQuantile()._compute(data, quantile=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        result['BFI'] = _BFI()._compute(data, method='LH')

        # Streamflow variability
        result['SFDC'] = _SFDC()._compute(data)
        result['CV'] = _CV()._compute(data) 
        result['QCV'] = _QCV()._compute(data)
        result['RBI'] = _RichardsBakerIndex()._compute(data)
        result['DVIa'] = _DVIa()._compute(data)
        result['DVIc'] = _DVIc()._compute(data)
        result['GINI'] = _GiniCoefficient()._compute(data)
        result['AC1'] = _Autocorrelation()._compute(data, lag=1) 
        result['CONC'] = _Concentration()._compute(data)

        # Event-based summaries
        hf_threshold = 9. * self.ts.summary.flow_quantile(quantile=0.5)['Q50'].iloc[0]
        lf_threshold = 0.2 * self.ts.summary.mean_flow()['MEAN'].iloc[0]
        _, hf_event_summary = _HighFlowEvents(self.ts).compute(threshold=hf_threshold, summarise=True, by_year=by_year, rolling=rolling)
        _, lf_event_summary = _LowFlowEvents(self.ts).compute(threshold=lf_threshold, summarise=True, by_year=by_year, rolling=rolling)
        _, zf_event_summary = _NoFlowEvents(self.ts).compute(summarise=True, by_year=by_year, rolling=rolling)

        # Frequency  
        result['HIGH_Q_FREQ'] = hf_event_summary['frequency'].to_frame(name='HIGH_Q_FREQ')
        result['LOW_Q_FREQ'] = lf_event_summary['frequency'].to_frame(name='LOW_Q_FREQ')
        result['ZERO_Q_FREQ'] = zf_event_summary['frequency'].to_frame(name='ZERO_Q_FREQ')

        # Duration
        def convert_duration(event_summary, colname, key):
            try:
                return (event_summary['mean_event_duration'].dt.total_seconds() / 86400.).to_frame(name=key)
            except AttributeError:
                return event_summary['mean_event_duration'].to_frame(name=key)

        result['HIGH_Q_DUR'] = convert_duration(hf_event_summary, 'mean_event_duration', 'HIGH_Q_DUR')
        result['LOW_Q_DUR'] = convert_duration(lf_event_summary, 'mean_event_duration', 'LOW_Q_DUR')
        result['ZERO_Q_DUR'] = convert_duration(zf_event_summary, 'mean_event_duration', 'ZERO_Q_DUR')

        result = pd.concat(result.values(), axis=1)
        return pd.merge(result, duration, left_index=True, right_index=True)

class _CV(BaseSummary): 
    def _compute(self, data):
        return data.groupby('group').apply(compute_cv).to_frame(name='CV')

class _QCV(BaseSummary): 
    def _compute(self, data):
        return data.groupby('group').apply(compute_qcv).to_frame(name='QCV')

class _Autocorrelation(BaseSummary):
    def _compute(self, data, lag): 
        lag = int(lag)
        return data.groupby('group').apply(compute_autocorrelation, lag=lag).to_frame(f'AC{lag}')

class _Concentration(BaseSummary): 
    def _compute(self, data): 
        return data.groupby('group').apply(compute_seasonality).to_frame('CONC')

class _FlowQuantile(BaseSummary): 
    def _compute(self, data, quantile): 
        label = format_quantile(quantile)
        if len(label) == 1:
            # label = label[0]
            result = data.groupby('group').apply(compute_quantile, q=quantile).apply(pd.Series)
        else:
            result = data.groupby('group').apply(compute_quantile, q=quantile)
        result.columns = label
        return result 

class _IQR(BaseSummary): 
    def _compute(self, data):
        return data.groupby('group').apply(compute_iqr).to_frame(name='IQR')

class _Skewness(BaseSummary):
    def _compute(self, data): 
        return data.groupby('group').apply(compute_skew).to_frame(name='SKEW')

class _RichardsBakerIndex(BaseSummary):
    def _compute(self, data):  
        return data.groupby('group').apply(compute_rbi).to_frame(name='RBI')

class _MeanFlow(BaseSummary):
    def _compute(self, data):  
        return data.groupby('group')['Q'].mean().to_frame(name='MEAN')

class _MaximumFlow(BaseSummary):
    def _compute(self, data):
        return data.groupby('group', sort=False)['Q'].max().to_frame(name='MAX')

class _MinimumFlow(BaseSummary):
    def _compute(self, data):
        return data.groupby('group')['Q'].min().to_frame(name='MIN')

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

def compute_baseflow(data, method='LH'): 
    data = data.copy()
    complete_data = data.dropna(subset='Q').copy()
    Q = complete_data['Q'].values
    if method.upper() == 'LH': 
        Qb = lh(Q)
    else:
        raise ValueError(f'Baseflow separation method {method} not recognised')

    complete_data.loc[:, 'Qb'] = Qb
    data = data.merge(complete_data[['Qb']], left_index=True, right_index=True)
    return data

class _BFI(BaseSummary):
    def _compute(self, data, method='LH'):
        data = compute_baseflow(data, method=method)
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
    def _compute(self, data, threshold: float, min_diff: int = 24): #, summarise=False, by_year=False, rolling=None, center=False):
        return self._flow_events(data, pot, min_diff, threshold=threshold)


class _LowFlowEvents(EventBasedSummary):
    def _compute(self, data, threshold, min_diff: int = 24):
        return self._flow_events(data, lowflow, min_diff, threshold=threshold)


class _HighFlowEvents(EventBasedSummary): 
    def _compute(self, data, threshold, min_diff: int = 24): #, summarise=False, by_year=False, rolling=None, center=False): 
        return self._flow_events(data, highflow, min_diff, threshold=threshold)


class _NoFlowEvents(EventBasedSummary): 
    def _compute(self, data, threshold: float = 0.001):
        # data = self.data.copy()
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
    
    def _summarize_events(self, data, events, by_year, rolling, center): 
        # First calculate the duration of each period
        grouped_data = self._get_grouped_data(data, by_year=by_year, rolling=rolling, center=center)
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

    def _compute(self, data, quantile: float = 0.25): #, summarise=False, by_year=False, rolling=None, center=False) -> float: 

        threshold = data['Q'].quantile(quantile)
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
        data = self._format_data(by_year=by_year)
        events = self._compute(data, **kwargs)
        if events is None or events.empty:
            return None if not summarise else (None, None)

        if summarise:
            summary = self._summarize_events(data, events, by_year=by_year, rolling=rolling, center=center)
            return events, summary
        else:
            return events


class _NoFlowFraction(EventBasedSummary): 
    def compute(self, threshold: float = 0.001, by_year=False, rolling=None, center=False): 

        data = self._format_data(by_year=by_year)
        def noflow(vals, threshold): 
            return vals < threshold

        data = self._simple_flow_events(data, noflow, threshold=threshold)
        data = self._get_grouped_data(data, by_year=by_year, rolling=rolling, center=center)
        result = data.groupby('group').agg(
            event_duration=('event_duration', 'sum'),
            summary_period_duration=('timestep', 'sum')
        )
        result['noflow_fraction'] = result['event_duration'] / result['summary_period_duration']
        return result[['noflow_fraction']]


class _HighFlowFraction(EventBasedSummary): 
    def compute(self, threshold, by_year=False, rolling=None, center=False): 

        data = self._format_data(by_year=by_year)
        def highflow(vals, threshold): 
            return vals > threshold

        def compute_high_flow_fraction(threshold):
            data = self._simple_flow_events(data, highflow, threshold=threshold)
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
        data = self._format_data(by_year=by_year)
        def lowflow(vals, threshold): 
            return vals < threshold

        def compute_low_flow_fraction(threshold):
            data = self._simple_flow_events(data, lowflow, threshold=threshold)
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

# @register_summary_method 
# def gsim_annual(ts_or_df): 
#     return _GSIM(ts_or_df).compute(annual=True)

# @register_summary_method 
# def gsim_seasonal(ts_or_df): 
#     return _GSIM(ts_or_df).compute(seasonal=True)

# @register_summary_method 
# def gsim_monthly(ts_or_df): 
#     return _GSIM(ts_or_df).compute(monthly=True)

@register_summary_method 
def streamflow_indices(ts_or_df, **kwargs): 
    return _StreamflowIndices(ts_or_df).compute(**kwargs)

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
def minimum_flow(ts_or_df, **kwargs):
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

# # Helpers for grouping
# def get_season_period(date):
#     """Return start and end date for a given Timestamp."""
#     year = date.year
#     month = date.month
#     if month in [12, 1, 2]: # DJF
#         start_year = year if month == 12 else year - 1
#         start_date = pd.Timestamp(f"{start_year}-12-01")
#         end_date = pd.Timestamp(f"{start_year + 1}-02-28")
#         if end_date.is_leap_year:
#             end_date = end_date + pd.Timedelta(days=1)
#     elif month in [3, 4, 5]: # MAM
#         start_date = pd.Timestamp(f"{year}-03-01")
#         end_date = pd.Timestamp(f"{year}-05-31")
#     elif month in [6, 7, 8]: # JJA
#         start_date = pd.Timestamp(f"{year}-06-01")
#         end_date = pd.Timestamp(f"{year}-08-31")
#     elif month in [9, 10, 11]: # SON
#         start_date = pd.Timestamp(f"{year}-09-01")
#         end_date = pd.Timestamp(f"{year}-11-30")
#     return start_date, end_date

def get_season_period(date, seasons_dict):
    """
    Return start and end date for the season that `date` belongs to,
    based on a custom `seasons_dict` like {'DJF': [12,1,2], 'MAM': [3,4,5], ...}.
    """
    month = date.month
    year = date.year

    # Identify the season
    for season, months in seasons_dict.items():
        if month in months:
            season_months = months
            season_name = season
            break
    else:
        return (pd.NaT, pd.NaT)
        # raise ValueError(f"Month {month} not found in any season definition.")

    # Determine start year
    first_month = season_months[0]
    start_year = year
    if month < first_month:
        # The season starts in the previous year
        start_year -= 1

    # Build start and end dates
    start_date = pd.Timestamp(f"{start_year}-{first_month:02d}-01")

    last_month = season_months[-1]
    end_year = start_year if last_month >= first_month else start_year + 1
    end_day = pd.Timestamp(f"{end_year}-{last_month:02d}-01") + pd.offsets.MonthEnd(0)
    end_date = end_day

    return start_date, end_date

def assign_season_label(date, seasons_dict):
    """Return season label for a given date."""
    month = date.month
    for season, months in seasons_dict.items():
        if month in months:
            return season
    return None
    # if month in [3, 4, 5]:
    #     season = 'MAM'
    # elif month in [6, 7, 8]:
    #     season = 'JJA'
    # elif month in [9, 10, 11]:
    #     season = 'SON'
    # else:
    #     season = 'DJF'
    # return f"{season}"

def assign_season_year(date):
    """Return season start year for a given date."""
    year = date.year
    month = date.month
    if month in [1, 2]: # DJF
        return year-1 #f'{year - 1}'
    else:
        return year #f'{year}'

def assign_month_label(date): 
    month = date.month 
    return f"{month:02d}"

# def season_string_to_months(season_str):
#     """
#     Convert a season string like 'MAM' or 'NDJF' to a list of month indices.
#     E.g., 'MAM' -> [3, 4, 5], 'NDJF' -> [11, 12, 1, 2]
#     """
#     month_abbr_to_num = {month.upper(): i for i, month in enumerate(calendar.month_abbr) if month}
#     months = []
#     i = 0
#     while i < len(season_str):
#         abbr = season_str[i:i+3].upper()
#         if abbr not in month_abbr_to_num:
#             raise ValueError(f"Invalid month abbreviation: '{abbr}'")
#         months.append(month_abbr_to_num[abbr])
#         i += 3

#     return months
# def season_to_months(season):
#     """Convert a season string like 'NDJF' to list of month numbers."""
#     month_str_to_num = {month[:3].upper(): i for i, month in enumerate(calendar.month_abbr) if month}
#     months = []
#     i = 0
#     while i < len(season):
#         abbr = season[i:i+3]
#         if abbr not in month_str_to_num:
#             raise ValueError(f"Invalid month abbreviation: {abbr}")
#         months.append(month_str_to_num[abbr])
#         i += 3
#     return months

MONTHS = 'JFMAMJJASONDJFMAMJJASONDJF'  # 24 months for wraparound

# def reorder_by_earliest_start(lists):
#     """
#     Reorder a list of lists so that the sublist with the smallest
#     first element is first. Relative order of the rest is preserved.
#     """
#     if not lists:
#         return lists

#     # Find index of the sublist with the smallest first element
#     start_idx = min(range(len(lists)), key=lambda i: lists[i][0])
    
#     # Rotate the list
#     return lists[start_idx:] + lists[:start_idx]
def reorder_season_dict_by_start_month(season_dict):
    """
    Reorder a season dictionary so the entry with the smallest starting month comes first.
    Preserves the order of the remaining items.
    """
    if not season_dict:
        return season_dict

    items = list(season_dict.items())
    # Find the index of the entry with the smallest first month
    start_idx = min(range(len(items)), key=lambda i: items[i][1][0])
    # Rotate the items
    rotated_items = items[start_idx:] + items[:start_idx]
    # Reconstruct dictionary (order preserved in Python 3.7+)
    return dict(rotated_items)

def parse_season_string(season_str):
    """
    Given a season string like 'MAM', 'JASO', 'NDJF', return:
    - the list of month indices (1–12)
    - the starting month index (1-based)
    
    Raises ValueError if the season string doesn't represent consecutive months.
    """
    season_str = season_str.upper()

    idx = MONTHS.find(season_str)
    if idx == -1: 
        raise ValueError(f'{season_str} is not a valid season')
    n_months = len(season_str)

    # Build list of month indices, wrapping around year end
    month_indices = [(idx + i) % 12 + 1 for i in range(n_months)]
    return month_indices

def validate_seasons(seasons):
    """Ensure seasons form a continuous sequence of months, return the start month."""
    season_index = [parse_season_string(season) for season in seasons]
    merged_season_index = sum(season_index, [])
    if len(merged_season_index) > len(set(merged_season_index)): 
        raise ValueError(f'Invalid seasons: some months are duplicated')

    seasons_dict = {season:parse_season_string(season) for season in seasons}
    seasons_dict = reorder_season_dict_by_start_month(seasons_dict)
    return seasons_dict

class BaseSummary: 

    def __init__(self, ts_or_df: Optional["HydroTS"]=None):
        # if isinstance(ts_or_df, HydroTS):
        if ts_or_df is None: 
            self.ts = None 
            self.data = None 
        elif hasattr(ts_or_df, "valid_data"):  # Likely a HydroTS object
            self.ts = copy.deepcopy(ts_or_df)
            self.data = None #ts_or_df.valid_data.copy()
        # elif isinstance(ts_or_df, pd.DataFrame):
        #     self.ts = None
        #     self.data = ts_or_df.copy()
        else:
            raise TypeError("Input must be a HydroTS object or a pandas DataFrame or None.")
    # def __init__(self, ts_or_df: Optional[Union["HydroTS", pd.DataFrame]]=None, discharge_col=None): 
    #     # if isinstance(ts_or_df, HydroTS):
    #     if ts_or_df is None: 
    #         self.ts = None 
    #         self.data = None 
    #     elif hasattr(ts_or_df, "valid_data"):  # Likely a HydroTS object
    #         self.ts = ts_or_df
    #         self.data = ts_or_df.valid_data.copy()
    #     elif isinstance(ts_or_df, pd.DataFrame):
    #         self.ts = None
    #         self.data = ts_or_df.copy()
    #     else:
    #         raise TypeError("Input must be a HydroTS object or a pandas DataFrame or None.")

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
            data.groupby('group')[['time', 'Q']] # FIXME should be `value` not `Q`
            .agg(
                summary_period_duration=('time', lambda x: x.max() - x.min() + ((x.iloc[1] - x.iloc[0]) if len(x) > 1 else pd.Timedelta(0))),
                data_availability=('Q', lambda x: x.notna().sum() / len(x) if len(x) > 0 else 0)
            ) # FIXME should be able to get time resolution from HydroTS object
        )

    def _get_grouped_data(self, data, by_year=False, rolling=None, center=False, by_season=False, seasons=None, by_month=False):

        data = data.reset_index(drop=False)
        if by_season or by_month: 

            if by_season:
                seasons_dict = validate_seasons(seasons)

            # Remove index, because this won't work with overlapping groups (i.e. when `rolling` is set)
            data = data.reset_index(drop=False) 
            data['group0'] = data['time'].apply(lambda x: assign_season_label(x, seasons_dict)) if by_season else data['time'].apply(assign_month_label)
            data['group1'] = data['water_year'] # This works because we have changed the water year, above

            # If by_season/by_month and by_year we use the calendar year to create groups
            if by_year:
                group_datas = []
                if rolling is not None: 
                    years = data['group1'].unique()
                    group_datas = []
                    for i in range(len(years) - rolling + 1):
                        window_years = years[i:i + rolling]
                        if center:
                            center_idx = i + rolling // 2
                            group_label = str(years[center_idx])
                        else:
                            group_label = f"{window_years[0]}–{window_years[-1]}"
                        window_data = data[data['group1'].isin(window_years)].copy()
                        window_data['group'] = group_label + '-' + window_data['group0']
                        group_datas.append(window_data)

                    data = pd.concat(group_datas, ignore_index=True)
                else: 
                    data['group'] = data['group1'].astype(str) + '-' + data['group0']

            else:
                data['group'] = data['group0']

            data = data.drop(['group0', 'group1'], axis=1)

        if not by_month and not by_season:

            data = data.reset_index(drop=False)
            years = sorted(self.ts.valid_years)

            if by_year:
                if rolling is not None:
                    group_datas = []
                    for i in range(len(years) - rolling + 1):
                        window_years = years[i:i + rolling]
                        if center:
                            center_idx = i + rolling // 2
                            group_label = str(years[center_idx])
                        else:
                            group_label = f"{window_years[0]}-{window_years[-1]}"
                        window_data = data[data['water_year'].isin(window_years)].copy()
                        window_data['group'] = group_label
                        group_datas.append(window_data)
                    data = pd.concat(group_datas, ignore_index=True)
                else:
                    data['group'] = data['water_year']

            else:
                data['group'] = f'{years[0]}-{years[-1]}'
        
        return data

    def _compute(self, data, **kwargs): 
        raise NotImplementedError

    def _format_data(self, by_season=False, seasons=['MAM', 'JJA', 'SON', 'DJF'], by_month=False): 
        if by_season:
            seasons_dict = validate_seasons(seasons)
            first_season = next(iter(seasons_dict))
            season_start_index = seasons_dict[first_season][0]
            # Check the water year and update if necessary
            if self.ts.water_year_start[0] != season_start_index: 
                self.ts.update_water_year(use_water_year=True, water_year_start=(season_start_index, 1))

            data = self.ts.valid_data.copy()
            # This ensures that data availability is correctly calculated 
            # when the timeseries start and end occurs in the middle of a season.
            valid_seasons = data.index.map(lambda x: get_season_period(x, seasons_dict)).unique()
            valid_mask = [(pd.notna(start) and pd.notna(end)) for (start, end) in valid_seasons]
            valid_seasons = valid_seasons[valid_mask]
            new_index = pd.concat([pd.Series(pd.date_range(start, end, freq=self.ts.freq, tz=None), name='time') for start, end in valid_seasons])
            new_index = pd.DatetimeIndex(new_index)
            data = data.reindex(new_index)

        elif by_month: 
            if self.ts.water_year_start[0] != 1: 
                self.ts.update_water_year(use_water_year=False)
            data = self.ts.valid_data.copy()
        else: 
            data = self.ts.valid_data.copy()
        return data 

    def compute(self, 
                by_year=False, 
                rolling=None, 
                center=False, 
                by_season=False, 
                seasons=['MAM', 'JJA', 'SON', 'DJF'],
                by_month=False, 
                include_duration=True, 
                **kwargs):

        data = self._format_data(by_season=by_season, seasons=seasons, by_month=by_month)
        data = self._get_grouped_data(
            data, by_year=by_year, rolling=rolling, 
            center=center, by_season=by_season, seasons=seasons,
            by_month=by_month
        )
        duration = self._compute_duration(data)
        result = self._compute(data, **kwargs)
        if include_duration:
            return pd.merge(result, duration, left_index=True, right_index=True)
        else:
            return result 

class EventBasedSummary(BaseSummary): 
    def _compute_mean_deficit(self, events): 
        pass

    def _simple_flow_events(self, data, event_condition, **kwargs): 
        data = data.reset_index(drop=False)
        vals = self.data['Q'].values 
        is_event = event_condition(vals, **kwargs)
        data['event_duration'] = is_event * data['timestep'] # timestep has units of days
        data = data.set_index('time')
        return data

    def _flow_events(self, data, event_condition, min_diff: int = 24, **kwargs) -> pd.DataFrame: 

        vals = data['Q'].values 

        # Indices where data exceeds threshold
        data_index = event_condition(vals, **kwargs)
        if len(data_index) == 0:
            return None

        # Time difference between consecutive exceedance indices
        times = data.index[data_index]
        diff_hours = np.diff(times).astype('timedelta64[h]')
        
        # Where difference is greater than min_diff, it's a new event
        sep_index = np.where(diff_hours > min_diff)[0]
        
        start_index = np.concatenate(([data_index[0]], data_index[sep_index + 1]))
        end_index = np.concatenate((data_index[sep_index], [data_index[-1]]))

        start_times = data.index[start_index].values
        end_times = data.index[end_index].values
        duration = (end_times - start_times).astype('timedelta64[D]')
        water_year = data.iloc[start_index]['water_year'].values # i.e. take the water year at the start of the event

        n_events = len(start_index)

        if n_events == 0:
            return None

        return pd.DataFrame({'water_year': water_year, 'event_start_time': start_times, 'event_end_time': end_times, 'event_duration': duration})

    def _summarize_events(self, data, events, by_year, rolling, center): 
        
        # First calculate the duration of each period
        grouped_data = self._get_grouped_data(data, by_year=by_year, rolling=rolling, center=center)
        duration = self._compute_duration(grouped_data)

        # Now aggregate the events 
        default_events = pd.DataFrame({
            'water_year': self.ts.valid_years, 
            'n_events': 0, 
            'mean_duration': None, 
            'total_duration': None}
        )
        if isinstance(events, pd.DataFrame):
            result = events.groupby('water_year').agg(
                n_events=('water_year', 'size'), 
                mean_duration=('event_duration', 'mean'), 
                total_duration=('event_duration', 'sum')
            )
            result = pd.DataFrame(result, index=self.ts.valid_years)
            result['n_events'] = result['n_events'].fillna(0).astype(int)
        else:
            result = default_events 

        result = self._get_grouped_data(result, by_year=by_year, rolling=rolling, center=center)
        result = result.groupby('group').agg(
            n_events=('n_events', 'sum'), # Take the water year of the event start
            mean_event_duration=('mean_duration', 'mean'),
            total_duration=('total_duration', 'sum')
        )
        result = pd.merge(result, duration, left_index=True, right_index=True)

        result['mean_event_duration'] = result['mean_event_duration'].to_numpy()
        result['total_duration'] = result['total_duration'].to_numpy()

        # Compute frequency and mean duration in days 
        def get_event_frequency(row): 
            try:
                total_duration_seconds = row['total_duration'].total_seconds()
            except AttributeError: 
                total_duration_seconds = float(row['total_duration']) # CHECK

            summary_period_duration_seconds = row['summary_period_duration'].total_seconds()
            return total_duration_seconds / summary_period_duration_seconds
        
        def get_mean_event_duration_days(row): 
            n_events = row['n_events']
            if n_events == 0: 
                return 0.
            else: 
                return float(row['mean_event_duration'].days)

        result['frequency'] = result.apply(get_event_frequency, axis=1)
        result['mean_event_duration_days'] = result.apply(get_mean_event_duration_days, axis=1)

        return result

    def _compute(self, data, **kwargs): 
        raise NotImplementedError 

    def compute(self, summarise=False, by_year=False, rolling=None, center=False, **kwargs):
        data = self._format_data()
        events = self._compute(data, **kwargs)
        if summarise:
            summary = self._summarize_events(data, events, by_year=by_year, rolling=rolling, center=center)
            return events, summary
        else:
            return events
