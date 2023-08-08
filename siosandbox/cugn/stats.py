""" Perform stats on this and that """

import os
import xarray
from glob import glob

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

import pandas

from IPython import embed

#def gaussian_cdf():

def chk_grid_gaussianity(values:np.ndarray, mean_grid:np.ndarray,
                         rms_grid:np.ndarray, indices:np.ndarray,
                         counts:np.ndarray, 
                         min_counts:int=10):

    p_values = np.ones_like(mean_grid)*np.nan
    # Cut on counts
    gd = counts > min_counts
    igd = np.where(gd)

    for ss in range(len(igd[0])):
        row, col = igd[0][ss], igd[1][ss]

        # Get indices
        in_cell = (indices[0] == row+1) & (indices[1] == col+1)
        idx_cell = np.where(in_cell)[0]

        # Prep
        vals = values[idx_cell]
        mean = mean_grid[row, col]
        rms = rms_grid[row, col]

        if row == 20 and col == 20:
            embed(header='41 chk_grid_gaussianity')

        # KS test
        r = stats.kstest(vals, 'norm', args=(mean, rms))
        p_values[row,col] = r.pvalue

    return p_values
