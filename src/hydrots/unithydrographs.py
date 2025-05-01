
import numpy as np

def route(flux_in: float, uh: np.ndarray) -> float:
    """
    Routes a flux through a unit hydrograph at the current timestep.

    Parameters:
    - flux_in (float): Input flux [mm/d]
    - uh (np.ndarray): Unit hydrograph of shape [2, n], where
        - uh[0, :] contains coefficients to split flow at each timestep
        - uh[1, :] contains still-to-flow values

    Returns:
    - float: Flux routed through the UH at this step
    """
    return uh[0, 0] * flux_in + uh[1, 0]

def uh_1_half(d_base, delta_t):
    """
    Unit Hydrograph [days] with half a bell curve. GR4J-based.
    
    Parameters
    ----------
    d_base : float
        Time base of routing delay [days]
    delta_t : float
        Time step size [days]

    Returns
    -------
    UH : ndarray
        Unit hydrograph [2 x n array]
        First row contains the UH coefficients,
        Second row is all zeros (still-to-flow values placeholder)
    """
    
    # Calculate routing delay in time steps
    delay = d_base / delta_t
    if delay == 0:
        delay = 1  # Avoid divide-by-zero or empty time series
    
    tt = np.arange(1, int(np.ceil(delay)) + 1)  # Time steps from 1 to ceil(delay)
    
    SH = np.zeros(len(tt) + 1)  # Cumulative hydrograph
    UH = np.zeros(len(tt))      # Unit hydrograph
    
    for t in tt:
        if t < delay:
            SH[t] = (t / delay) ** (5 / 2)
        else:
            SH[t] = 1.0
        UH[t - 1] = SH[t] - SH[t - 1]
    
    UH_out = np.vstack([UH, np.zeros_like(UH)])  # Second row: placeholder zeros
    
    return UH_out

