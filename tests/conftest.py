
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def simple_timeseries():
    """Simple daily timeseries with no missing values."""
    date_range = pd.date_range(start="2000-01-01", end="2000-12-31", freq="D")
    flow = np.random.rand(len(date_range))
    return pd.DataFrame({'time': date_range, 'Q': flow})

@pytest.fixture
def timeseries_with_noflow():
    """Timeseries with extended zero-flow periods."""
    date_range = pd.date_range(start="2000-01-01", end="2000-12-31", freq="D")
    flow = np.random.rand(len(date_range))
    flow[50:60] = 0.0
    flow[200:220] = 0.0
    return pd.DataFrame({'time': date_range, 'Q': flow})

@pytest.fixture
def timeseries_with_gaps():
    """Timeseries with NaNs."""
    date_range = pd.date_range(start="2000-01-01", end="2000-12-31", freq="D")
    flow = np.random.rand(len(date_range))
    flow[100:110] = np.nan
    return pd.DataFrame({'time': date_range, 'Q': flow})
