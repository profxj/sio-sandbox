# imports
from importlib import reload
import os
import xarray

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import seaborn as sns

import pandas

from siosandbox.cugn import grid_utils
from siosandbox.cugn import figures
from siosandbox.cugn import clusters
from siosandbox.cugn import io as cugn_io

from IPython import embed


def year_outliers(line:str, year:int=2017, pcut:float=95., show_depth:bool=False):

    # Load and build outliers
    grid_outliers, grid_tbl, ds = grid_utils.gen_outliers(line, pcut)

    # Time conveniences
    ptimes = pandas.to_datetime(grid_outliers.time.values)
    months = ptimes.month

    # Plot a year
    in_year = ptimes.year == year
    all_gd = in_year

    outfile = f'Figures/oxy_outliers_{year}_p{int(pcut)}.png' 
    if show_depth:
        yval = grid_outliers.z[all_gd]
        ylbl = 'Depth (m)'
    else:
        yval = ds.sigma0.data[(grid_outliers.depth.values, 
                               grid_outliers.profile.values)][all_gd],
        ylbl = 'Potential Density'

    # Do it
    figures.outlier_by_months(outfile, pcut, year,
        grid_outliers.lon.values[all_gd], yval, months[all_gd], ylbl=ylbl)

def outlier_montage(line:str, outl_dict:dict, outfile:str): 
    # Load and build outliers
    grid_outliers, grid_tbl, ds = grid_utils.gen_outliers(line, outl_dict['perc'])

    # Time conveniences
    ptimes = pandas.to_datetime(grid_outliers.time.values)
    months = ptimes.month

    # Cut
    idx_o = (ptimes.year==outl_dict['year']) & (
        grid_outliers.z >= outl_dict['z'][0]) & (
        grid_outliers.z <= outl_dict['z'][1]) & (
        grid_outliers.lon >= outl_dict['lons'][0]) & (
        grid_outliers.lon <= outl_dict['lons'][1])

    if 'months' in outl_dict.keys():
        idx_o = idx_o & (
            months>=outl_dict['months'][0]) & (
            months<=outl_dict['months'][1])

    # DOXY CUT BY JXP
    idx_o = idx_o & (grid_outliers.doxy >= 300.)

    sngl_outliers = grid_outliers[idx_o]

    # FIGURE
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    # Year plot
    ax0 = plt.subplot(gs[0])
    in_year = ptimes.year == outl_dict['year']

    figures.outlier_by_months(
        None, outl_dict['perc'], outl_dict['year'],
        grid_outliers.lon.values[in_year], 
        grid_outliers.sigma0.values[in_year], 
        months[in_year], ax=ax0)
    ax0.plot(sngl_outliers.lon, sngl_outliers.sigma0, 
        'o', color='grey', alpha=0.5, mfc='none', ms=3)

    # #################################
    # Histogram of DO
    all_DO = grid_utils.grab_control_values(sngl_outliers, grid_tbl, 'doxy')

    ax2 = plt.subplot(gs[2])
    sns.histplot(all_DO, ax=ax2, log_scale=(False,True))
    sns.histplot(sngl_outliers.doxy, ax=ax2)
    ax2.set_xlabel('Dissolved Oxygen')

    # #################################
    # Histogram of Chla
    grid_tbl['chla'] = ds.chlorophyll_a.data[(grid_tbl.depth.values, 
                               grid_tbl.profile.values)]
    all_Chla = grid_utils.grab_control_values(sngl_outliers, grid_tbl, 'chla')

    ax3 = plt.subplot(gs[3])
    sns.histplot(all_Chla, ax=ax3, log_scale=(False,True))
    sns.histplot(sngl_outliers.chla, ax=ax3)
    ax3.set_xlabel('ChlA')

    
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    
def cluster_stats(line:str, perc:float):

    # Load clusters of outliers
    grid_outliers = clusters.generate_clusters(line, perc)

    # Cluster stats
    cluster_stats = clusters.cluster_stats(grid_outliers)
    embed(header='cluster_stats')


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)


    # Build Grid
    if flg & (2**0):
        #line = '90'
        #line = '80'
        #line = '66'
        line = '56'
        line_files = cugn_io.line_files(line)

        build_grid(line_files['datafile'],
            line_files['gridtbl_file'], 
            line_files['edges_file'])

    # Year of outliers (testing)
    if flg & (2**1):
        line = '90'
        year_outliers(line, year=2017, pcut=95., show_depth=True)

    # Montage
    if flg & (2**2):
        line = '90'
        #outl_dict = dict(perc=98., year=2017, lons=[-118., -100.],
        #                z=[0., 50.], months=[6,8])
        outl_dict = dict(perc=98., year=2017, lons=[-119., -100.],
                        z=[0., 100.], months=[2,4])
        outlier_montage(line, outl_dict, 'Figures/tst.png')

    # Montage
    if flg & (2**3):
        cluster_stats('90', perc=98.)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Build grids
        #flg += 2 ** 1  # 2 -- Year plot
        #flg += 2 ** 2  # 4 -- Montage
        #flg += 2 ** 3  # 8 -- Generate clusters
    else:
        flg = sys.argv[1]

    main(flg)