import numpy as np

def of_KGE(obs, sim, w=None):
    """
    Calculates Kling-Gupta Efficiency (Gupta et al., 2009).
    
    Parameters:
        obs (array_like): Time series of observations [n]
        sim (array_like): Time series of simulations [n]
        idx (array_like, optional): Indices to use for calculation
        w (array_like, optional): Weights for [r, alpha, beta] components [3]
    
    Returns:
        val (float): KGE value
        c (ndarray): Components [r, alpha, beta]
        idx (ndarray): Indices used for calculation
        w (ndarray): Weights used [3]
    """
    obs = np.asarray(obs)
    sim = np.asarray(sim)

    # if idx is not None:
    #     idx = np.asarray(idx)
    #     if idx.dtype == bool:
    #         obs = obs[idx]
    #         sim = sim[idx]
    #     else:
    #         obs = obs[idx]
    #         sim = sim[idx]
    # else:
    # idx = np.arange(len(obs))

    # Remove any invalid values (e.g., negative flows if needed)
    valid = (obs >= 0) & (sim >= 0)
    obs = obs[valid]
    sim = sim[valid]
    # idx = idx[valid]

    # Default weights
    if w is None:
        w = np.array([1.0, 1.0, 1.0])
    else:
        w = np.asarray(w).flatten()
        if w.shape != (3,):
            raise ValueError("Weights should be a 3-element vector.")

    # Calculate components
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta = np.mean(sim) / np.mean(obs)
    c = np.array([r, alpha, beta])

    # Calculate KGE value
    val = 1 - np.sqrt(
        (w[0] * (c[0] - 1))**2 +
        (w[1] * (c[1] - 1))**2 +
        (w[2] * (c[2] - 1))**2
    )

    return val #, c, w
