""" Module for oxygen data analysis. """
# imports
import os
import xarray
from glob import glob

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

import pandas

from gsw import conversions, density

from IPython import embed

def gen_map(ds:xarray.Dataset, axes:tuple=('SA', 'sigma0'),
            stat:str='median'):

    # Default bins -- Line 90
    bins = dict(SA=np.linspace(32.8, 34.8, 50),
                sigma0=np.linspace(23.0, 27.2, 50),
                CT=np.linspace(5, 22.5, 50))

    # Cut on good data
    xkey = axes[0]
    ykey = axes[1]
    
    gd = np.isfinite(ds[xkey]) & np.isfinite(ds[ykey]) & np.isfinite(ds.doxy)

    # Histogram
    med_oxy, xedges, yedges, indices =\
            stats.binned_statistic_2d(
                ds[xkey].data[gd], 
                ds[ykey].data[gd], 
                ds.doxy.data[gd],
                statistic=stat,
                bins=[bins[xkey], bins[ykey]],
                expand_binnumbers=True)

    # Counts
    counts, _, _ = np.histogram2d(
                ds[xkey].data[gd], 
                ds[ykey].data[gd], 
                bins=[bins[xkey], bins[ykey]])

    # Return
    return med_oxy, xedges, yedges, counts, indices