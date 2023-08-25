""" Module for grid utilities. """
# imports
import os
import xarray

import numpy as np
from scipy import stats

import pandas

from siosandbox import cat_utils
from siosandbox.cugn import io as cugn_io

from IPython import embed

default_bins = dict(SA=np.linspace(32.8, 34.8, 50),
                sigma0=np.linspace(23.0, 27.2, 50),
                CT=np.linspace(5, 22.5, 50))

def gen_grid(ds:xarray.Dataset, axes:tuple=('SA', 'sigma0'),
            stat:str='median', bins:dict=None):

    # Default bins -- Line 90
    if bins is None:
        bins = default_bins

    # Cut on good data
    xkey, ykey = axes
    
    gd = np.isfinite(ds[xkey]) & np.isfinite(ds[ykey]) & np.isfinite(ds.doxy)

    # Histogram
    med_oxy, xedges, yedges, grid_indices =\
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
    return med_oxy, xedges, yedges, counts, grid_indices, ds.doxy.data[gd], gd

def chk_grid_gaussianity(values:np.ndarray, mean_grid:np.ndarray,
                         rms_grid:np.ndarray, indices:np.ndarray,
                         counts:np.ndarray, 
                         min_counts:int=10):
    """ Evaluate the gaussianity of a grid of values

    Args:
        values (np.ndarray): Values to check
        mean_grid (np.ndarray): Average values
        rms_grid (np.ndarray): RMS of values
        indices (np.ndarray): _description_
        counts (np.ndarray): _description_
        min_counts (int, optional): _description_. Defaults to 10.

    Returns:
        np.ndarray: KS test p-values
    """

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

        #if row == 20 and col == 20:
        #    embed(header='41 chk_grid_gaussianity')

        # KS test
        r = stats.kstest(vals, 'norm', args=(mean, rms))
        p_values[row,col] = r.pvalue

    return p_values

def gen_outliers(line:str, pcut:float):
    
    # Load and unpack
    items = cugn_io.load_line(line)
    ds = items['ds']
    grid_tbl = items['grid_tbl']

    # Outliers
    if pcut > 50.:
        outliers = grid_tbl.doxy_p > pcut
    else:
        raise IOError("Need to implement lower percentile")

    grid_outliers = grid_tbl[outliers].copy()

    # Decorate items
    grid_outliers['times'] = pandas.to_datetime(ds.time[grid_outliers.profile.values].values)
    grid_outliers['lon'] = ds.lon[grid_outliers.profile.values].values
    grid_outliers['z'] = ds.depth[grid_outliers.depth.values].values

    # Physical quantities
    grid_outliers['SA'] = ds.SA.data[(grid_outliers.depth.values, 
                               grid_outliers.profile.values)]
    grid_outliers['sigma0'] = ds.sigma0.data[(grid_outliers.depth.values, 
                               grid_outliers.profile.values)]

    # Others                            
    grid_outliers['chla'] = ds.chlorophyll_a.data[(grid_outliers.depth.values, 
                               grid_outliers.profile.values)]

    # Return
    return grid_outliers, grid_tbl, ds

def old_find_outliers(values:np.ndarray, 
                  grid_indices:np.ndarray,
                  counts:np.ndarray, percentile:float,
                  da_gd:np.ndarray,
                  min_counts:int=50):
    """ Find outliers in a grid of values

    Args:
        values (np.ndarray): All of the values used to construct the grid
        grid_indices (np.ndarray): Indices of where the values are in the grid
        counts (np.ndarray): Counts per grid cell
        percentile (float): Percentile to use for outlier detection
        da_gd (np.ndarray): Used for indexing in ds space
        min_counts (int, optional): Minimum counts in the grid
          to perform analysis. Defaults to 50.

    Returns:
        tuple:
            np.ndarray: Outlier indices on ds grid, [depth, profile]
            np.ndarray: Row, col for each outlier
    """

    # upper or lower?
    high = True if percentile > 50 else False

    # Cut on counts
    gd = counts > min_counts
    igd = np.where(gd)

    ngd_grid_cells = len(igd[0])

    da_idx = np.where(da_gd)

    # Prep
    save_outliers = []
    save_rowcol = []
    save_cellidx = []

    # Loop on all the (good) grid cells
    for ss in range(ngd_grid_cells):
        # Unpack
        row, col = igd[0][ss], igd[1][ss]

        # Get indices
        in_cell = (grid_indices[0] == row+1) & (grid_indices[1] == col+1)
        idx_cell = np.where(in_cell)[0]

        # Percentile
        vals = values[idx_cell]
        pct = np.nanpercentile(vals, percentile)

        # Outliers
        if high:
            ioutliers = np.where(vals > pct)[0]
        else:
            ioutliers = np.where(vals < pct)[0]
        for ii in ioutliers:
            save_outliers.append((da_idx[0][idx_cell[ii]],
                             da_idx[1][idx_cell[ii]]))
            save_rowcol.append((row, col))
        save_cellidx += ioutliers.tolist()

    # Return
    return np.array(save_outliers), np.array(save_rowcol), np.array(save_cellidx)

def grab_control_values(outliers:pandas.DataFrame,
                        grid_tbl:pandas.DataFrame,
                        metric:str):

    comb_row_col = np.array([col*10000 + row for row,col in zip(outliers.row.values, 
                                                                outliers.col.values)])
    uni_rc = np.unique(comb_row_col)
    uni_col = uni_rc // 10000
    uni_row = uni_rc - uni_col*10000

    all_vals = []
    for row, col in zip(uni_row, uni_col):
        in_cell = (grid_tbl.row == row) & (grid_tbl.col == col)
        vals = grid_tbl[metric].values[in_cell]
        all_vals += vals.tolist()
    # Return
    return all_vals

def find_perc(grid_tbl:pandas.DataFrame, metric:str='doxy'):

    # Find unique to loop over
    comb_row_col = np.array([col*10000 + row for row,col in zip(grid_tbl.row.values, grid_tbl.col.values)])
    uni_rc = np.unique(comb_row_col)
    uni_col = uni_rc // 10000
    uni_row = uni_rc - uni_col*10000

    all_perc = np.zeros(len(grid_tbl))

    for row, col in zip(uni_row, uni_col):
        # Get indices
        in_cell = (grid_tbl.row == row) & (grid_tbl.col == col)

        # Values in the cell
        vals = grid_tbl[metric].values[in_cell]

        srt = np.argsort(vals)
        in_srt = cat_utils.match_ids(np.arange(len(vals)), srt)
        perc = np.arange(len(vals))/len(vals-1)*100.

        # Save
        all_perc[in_cell] = perc[in_srt]

    # Return
    grid_tbl[f'{metric}_p'] = all_perc
    return 

def old_find_perc(values:np.ndarray, 
              grid_indices:np.ndarray,
              row_cols, cell_idx):

    # Find unique to loop over
    comb_row_col = np.array([col*10000 + row for row,col in row_cols])
    uni_rc = np.unique(comb_row_col)
    uni_col = uni_rc // 10000
    uni_row = uni_rc - uni_col*10000

    all_perc = np.zeros(row_cols.shape[0])

    for row, col in zip(uni_row, uni_col):
        # Get indices
        in_cell = (grid_indices[0] == row+1) & (grid_indices[1] == col+1)
        idx_cell = np.where(in_cell)[0]

        # Values in the cell
        vals = values[idx_cell]

        srt = np.argsort(vals)
        perc = np.arange(len(vals))/len(vals-1)*100.

        # Items of interest
        idx = (row_cols[:,0] == row) & (row_cols[:,1] == col)
        cell_i = cell_idx[idx]
        in_srt = cat_utils.match_ids(cell_i, srt)

        # Save
        all_perc[idx] = perc[in_srt]

    # Return
    return all_perc