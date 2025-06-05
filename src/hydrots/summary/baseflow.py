
import numpy as np

def lh(Q, beta=0.925, return_exceed=False):
    """LH digital filter (Lyne & Hollick, 1979)
    Lyne, V. and Hollick, M. (1979) Stochastic Time-Variable Rainfall-Runoff Modeling. Institute of Engineers Australia National Conference, 89-93.
    
    Args:
        Q (np.array): streamflow
        beta (float): filter parameter, 0.925 recommended by (Nathan & McMahon, 1990)
    """
    if return_exceed:
        b = np.zeros(Q.shape[0] + 1)
    else:
        b = np.zeros(Q.shape[0])

    # first pass
    b[0] = Q[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = beta * b[i] + (1 - beta) / 2 * (Q[i] + Q[i + 1])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1

    # second pass
    b1 = np.copy(b)
    for i in range(Q.shape[0] - 2, -1, -1):
        b[i] = beta * b[i + 1] + (1 - beta) / 2 * (b1[i + 1] + b1[i])
        if b[i] > b1[i]:
            b[i] = b1[i]
            if return_exceed:
                b[-1] += 1
    return b

# def baseflow_separation(streamflow, filter_parameter=0.925, passes=3):
def lh2(Q, beta=0.925, passes=3):
    """
    Separate streamflow into baseflow and quickflow components using an iterative filter.

    Parameters
    ----------
    Q : array-like
        Time series of streamflow values (must be 1D).
    beta : float, optional
        The recursive filter parameter (default is 0.925).
    passes : int, optional
        Number of filter passes (default is 3).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'bt': baseflow
        - 'qft': quickflow (streamflow - baseflow)
    """
    Q = np.asarray(Q, dtype=float)
    n = len(Q)

    # Indices for filtering
    ends = np.array([0, n - 1] * (passes + 1))[:passes + 2]
    add_to_start = np.array([1, -1] * passes)[:passes]

    btP = Q.copy()  # Previous pass's baseflow estimate
    bt = np.zeros_like(Q)

    # Guess initial baseflow value
    bt[0] = Q[0] if Q[0] < np.quantile(Q, 0.25) else np.mean(Q) / 1.5

    for j in range(passes):
        start = ends[j] + add_to_start[j] if j < len(add_to_start) else ends[j]
        end = ends[j + 1] + 1  # include end point
        step = 1 if start < end else -1

        for i in range(start, end, step):
            i_prev = i - add_to_start[j]
            term = (beta * bt[i_prev] +
                    ((1 - beta) / 2) * (btP[i] + btP[i_prev]))
            bt[i] = btP[i] if term > btP[i] else term

        if j < passes - 1:
            btP = bt.copy()
            end_idx = ends[j + 1]
            bt[end_idx] = (Q[end_idx] / 1.2
                           if Q[end_idx] < np.mean(btP)
                           else np.mean(btP))

    return bt