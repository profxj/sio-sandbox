""" Analysis for space and time """

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

def cut_on_dates(ds:xarray.Dataset, start:str, end:str):
    """ Cut on dates
    """

    # Cut on dates
    start_date = pandas.to_datetime(start)
    end_date = pandas.to_datetime(end)
    
    # Cut
    cut = (ds.time >= start_date) & (ds.time <= end_date)

    # Profile
    ds_cut = ds.isel(profile=cut)

    # Return
    return ds_cut