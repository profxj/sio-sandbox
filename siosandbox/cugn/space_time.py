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

def cut_on_months(ds:xarray.Dataset, start:int, end:int):
    """ Cut on months
    """

    #
    months = pandas.to_datetime(ds.time.data).month.values.astype(int)

    if start < end:
        cut = (months >= start) & (months <= end)
    else:
        cut = (months >= start) | (months <= end)

    # Profile
    ds_cut = ds.isel(profile=cut)

    # Return
    return ds_cut

def cut_on_lon(ds:xarray.Dataset, lon_min:float, lon_max:float):
    """ Cut on longitude
    """

    # Cut
    cut = (ds.lon >= lon_min) & (ds.lon <= lon_max)

    # Profile
    ds_cut = ds.isel(profile=cut)

    # Return
    return ds_cut