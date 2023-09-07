""" Methods related to the Annual Cycle """
import os

import numpy as np
from scipy.io import loadmat

import pandas

from siosandbox.cugn import utils as cugn_utils

from IPython import embed

def prep_m(Avar:np.ndarray, level:int, ip:int, nvals:int):
    maxharmonic = Avar['sin'].shape[2]

    m = np.zeros((nvals, 1 + 2*maxharmonic))
    if nvals > 1:
        m[:,0] = Avar['constant'][level, ip:ip+nvals]
        m[:,1:1+maxharmonic] = Avar['sin'][level, ip:ip+nvals]
        m[:,1+maxharmonic:] = Avar['cos'][level, ip:ip+nvals] 
    else:
        m[:,0] = Avar['constant'][level, ip]
        m[:,1:1+maxharmonic] = Avar['sin'][level, ip]
        m[:,1+maxharmonic:] = Avar['cos'][level, ip] 

    return m

def evaluate(Aarray:np.ndarray, variable:str, level:int, time:np.ndarray, dist:np.ndarray):
    """ Evaluate the annual cycle

    Args:
        Aarray (np.ndarray): _description_
        variable (str): _description_
        level (int): _description_
        time (np.ndarray): Unix time, i.e. seconds since 1970-01-01
        dist (np.ndarray): Distance from the shore in km

    Returns:
        np.ndarray: evals
    """

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
        idx = np.where(ii)[0]
        m = prep_m(A[variable], level, 0, 1)
        evals[idx] = (G[idx,:] @ m.T).flatten()

    jj = dist >= np.max(A['xcenter'])
    if np.any(jj):
        idx = np.where(jj)[0]
        m = prep_m(A[variable], level, -1, 1)
        evals[idx] = (G[idx,:] @ m.T).flatten()

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

def calc_for_grid(grid:pandas.DataFrame, line:str, variable:str):

    # Load up
    anncyc_file = os.path.join(os.getenv('CUGN'), f'anncyc{line}.mat')
    A = loadmat(anncyc_file, variable_names=['A'])['A']

    # Distance
    dist, offset = cugn_utils.calc_dist_offset(line, grid.lon.values, grid.lat.values)

    # Times
    unix_time = (grid.time - pandas.Timestamp("1970-01-01")) / pandas.Timedelta('1s')

    # Evaluate
    uni_depth = np.unique(grid.depth.values)
    T_Annual = np.zeros(unix_time.size)

    # Loop on depth
    for level in uni_depth:
        in_depth = grid.depth == level
        # 
        T_Annual[in_depth] = evaluate(A, variable, level, 
                                      unix_time[in_depth], dist[in_depth])

    # Return
    return T_Annual