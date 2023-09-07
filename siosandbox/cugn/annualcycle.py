""" Methods related to the Annual Cycle """

import numpy as np

def prep_m(Avar:np.ndarray, level:int, ip:int, nvals:int):
    maxharmonic = Avar['sin'].shape[2]

    m = np.zeros((nvals, 1 + 2*maxharmonic))
    m[:,0] = Avar['constant'][level, ip:ip+nvals]
    m[:,1:1+maxharmonic] = Avar['sin'][level, ip:ip+nvals]
    m[:,1+maxharmonic:] = Avar['cos'][level, ip:ip+nvals] 

    return m

def evaluate(Aarray:np.ndarray, variable:str, level:int, time:np.ndarray, dist:np.ndarray):

    evals = np.zeros(time.size)
    
    # Unpack A for convenience
    A = {}
    A['t'] = {}
    for key in ['constant', 'sin', 'cos']:
        A['t'][key] = Aarray[0][0][12][key][0][0]
    A['xcenter'] = Aarray[0][0][5][:,0].astype(float)

    # Prep
    timebin=2*np.pi*time/86400/365.25
    maxharmonic = A[variable]['sin'].shape[2]

    # Build G
    G = np.ones((time.size, 1 + 2*maxharmonic))
    for kk in range(maxharmonic):
        G[:,kk+1] = np.sin(timebin * (kk+1))
        G[:,kk+1+maxharmonic] = np.cos(timebin * (kk+1))

    # Beyond the grid
    ii = dist <= np.min(A['xcenter'])
    if np.any(ii):
        m = prep_m(A[variable], level, 0, 1)
        evals[ii] = G[ii,:] @ m

    jj = dist >= np.max(A['xcenter'])
    if np.any(jj):
        m = prep_m(A[variable], level, -1, 1)
        evals[jj] = G[jj,:] @ m

    # Within the grid
    dx = np.diff(A['xcenter'])

    ip = np.where(~ii & ~jj)[0]
    for n in ip:
        xx = A['xcenter'] - dist[n]
        # Find the zero crossing
        ip = np.where(xx[:-1] * xx[1:] <= 0)[0][0]
        # Prep
        m2 = prep_m(A[variable], level, ip, 2)
        # Evaluate
        bracket = G[n:n+1,:] @ m2.T

        evals[n] =  float(bracket @ [xx[ip+1], -xx[ip]]) / dx[ip]


    return evals