
import numpy as np
from numba import njit

from hydrots import fluxes 

@njit
def hymod_5p5s(forcing, params, stores, delta_t): 

    smax = params[0] 
    b = params[1]
    a = params[2]
    kf = params[3]
    ks = params[4] 

    S1 = stores[0] 
    S2 = stores[1]
    S3 = stores[2]
    S4 = stores[3]
    S5 = stores[4]

    P = forcing[0]
    Ep = forcing[1]
    T = forcing[2]

    # Fluxes functions
    flux_ea = fluxes.evap_7(S1, smax, Ep, delta_t)
    flux_pe = fluxes.saturation_2(S1, smax, b, P)
    flux_pf = fluxes.split_1(a, flux_pe)
    flux_ps = fluxes.split_1(1-a, flux_pe)
    flux_qf1 = fluxes.baseflow_1(kf, S2)
    flux_qf2 = fluxes.baseflow_1(kf, S3)
    flux_qf3 = fluxes.baseflow_1(kf, S4)
    flux_qs = fluxes.baseflow_1(ks, S5)

    # Stores 
    dS1 = P - flux_ea - flux_pe 
    dS2 = flux_pf - flux_qf1 
    dS3 = flux_qf1 - flux_qf2 
    dS4 = flux_qf2 - flux_qf3 
    dS5 = flux_ps - flux_qs

    # Outputs 
    dS = np.array([dS1, dS2, dS3, dS4, dS5]).flatten()
    flux = np.array([flux_ea, flux_pe, flux_pf, flux_ps, flux_qf1, flux_qf2, flux_qf3, flux_qs]).flatten()
    return dS, flux