
import numpy as np

def baseflow_1(p1: float, S: float) -> float:
    """
    Outflow from a linear reservoir.

    Parameters:
        p1 (float): time scale parameter [d⁻¹]
        S (float): current storage [mm]

    Returns:
        float: baseflow [mm/d]
    """
    return p1 * S

def evap_7(S: float, Smax: float, Ep: float, dt: float) -> float:
    """
    Evaporation based on scaled current water storage, limited by potential rate.

    Parameters:
        S (float): current storage [mm]
        Smax (float): maximum contributing storage [mm]
        Ep (float): potential evapotranspiration rate [mm/d]
        dt (float): time step size [d]

    Returns:
        float: evaporation flux [mm/d]
    """
    return min((S / Smax) * Ep, S / dt)


def evap_12(S: float, p1: float, Ep: float) -> float:
    """
    Evaporation from deficit store, with exponential decline as
    deficit goes below a threshold.

    Parameters:
    - S (float): current storage [mm]
    - p1 (float): wilting point [mm]
    - Ep (float): potential evapotranspiration rate [mm/d]

    Returns:
    - float: actual evaporation [mm/d]
    """
    return min(1, np.exp(2 * (1 - S / p1))) * Ep


def saturation_2(S: float, Smax: float, p1: float, In: float) -> float:
    """
    Saturation excess from a store with varying degrees of saturation.

    Parameters:
        S (float): current storage [mm]
        Smax (float): maximum contributing storage [mm]
        p1 (float): non-linear scaling parameter [-]
        In (float): incoming flux [mm/d]

    Returns:
        float: outgoing flux [mm/d]
    """
    saturation_deficit = max(0.0, min(1.0, 1 - S / Smax))
    return (1 - saturation_deficit ** p1) * In


def saturation_5(S: float, p1: float, p2: float, In: float) -> float:
    """
    Exponential saturation excess based on current storage and a threshold.

    Parameters:
    - S (float): current deficit [mm]
    - p1 (float): deficit threshold above which no flow occurs [mm]
    - p2 (float): exponential scaling parameter [-]
    - In (float): incoming flux [mm/d]

    Returns:
    - float: outgoing flux [mm/d]
    """
    S = max(S, 0)  # ensure S >= 0
    saturation_ratio = min(1, (S / p1) ** p2)
    return (1 - saturation_ratio) * In

def split_1(p1: float, In: float) -> float:
    """
    Split flow by a given fraction.

    Parameters:
    - p1 (float): fraction of flux to be diverted [-]
    - In (float): incoming flux [mm/d]

    Returns:
    - float: diverted flux [mm/d]
    """
    return p1 * In

