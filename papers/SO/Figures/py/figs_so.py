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
from siosandbox import plot_utils

from IPython import embed

def fig_pdf_cdf(outfile:str, line, SO_cut:float=1.1):

    # Load
    items = cugn_io.load_line(line)
    grid_tbl = items['grid_tbl']
    ds = items['ds']

    # Fill
    grid_utils.fill_in_grid(grid_tbl, ds)

    # Cuts
    highSO = grid_tbl.SO > SO_cut
    highSO_tbl = grid_tbl[highSO]

    # FIGURE
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    axes = []
    metrics = ['N', 'z', 'chla', 'T']
    labels = ['N (cycles/hour)', 'z (m)', 'Chl-a', 'T (deg C)']
    for ss, metric, label in zip(np.arange(len(metrics)), metrics, labels):
        # Build control
        control = grid_utils.grab_control_values(
            highSO_tbl, grid_tbl, metric)
        # Nan
        control = control[np.isfinite(control)]

        # Plot
        ax = plt.subplot(gs[ss])

        if metric in ['chla']:
            log_scale = (True, False)
        else:
            log_scale = (False, False)

        sns.ecdfplot(x=highSO_tbl[metric], ax=ax, label=f'SO > {SO_cut}', log_scale=log_scale)
        sns.ecdfplot(x=control, ax=ax, label='Control', color='k', log_scale=log_scale)

        if ss == 0:
            ax.legend(fontsize=15.)

        # Label
        ax.set_xlabel(label)
        ax.set_ylabel('CDF')

        axes.append(ax)

    # Pretty up
    for ax in axes:
        plot_utils.set_fontsize(ax, 17)

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_varySO_pdf_cdf(outfile:str, line):
    # Load
    items = cugn_io.load_line(line)
    grid_tbl = items['grid_tbl']
    ds = items['ds']

    # Fill
    grid_utils.fill_in_grid(grid_tbl, ds)

    # Figure
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    ax = plt.gca()

    metric = 'N'
    for SO_cut in [1., 1.05, 1.1, 1.2, 1.3]:
        highSO = grid_tbl.SO > SO_cut
        highSO_tbl = grid_tbl[highSO]

        if SO_cut == 1.:
            label = 'Control'
            control = grid_utils.grab_control_values(
                highSO_tbl, grid_tbl, metric)
            control = control[np.isfinite(control)]
            sns.ecdfplot(x=control, ax=ax, label='Control', color='k')

        sns.ecdfplot(x=highSO_tbl[metric], ax=ax, label=f'SO > {SO_cut}')


    # Finish
    ax.legend(fontsize=15.)
    plot_utils.set_fontsize(ax, 17)

    ax.set_xlabel('N (cycles/hour)')
    ax.set_ylabel('CDF')

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_timeseries(outfile:str, line):

    # Load
    items = cugn_io.load_line(line)
    grid_tbl = items['grid_tbl']
    ds = items['ds']

    # Fill
    grid_utils.fill_in_grid(grid_tbl, ds)

    # Cluters 
    perc = 80.  # Low enough to grab them all
    grid_outliers, _, _ = grid_utils.gen_outliers(line, perc)

    extrem = grid_outliers.SO > 1.1
    grid_extrem = grid_outliers[extrem].copy()
    times = pandas.to_datetime(grid_extrem.time.values)

    # Cluster me
    clusters.generate_clusters(grid_extrem)
    cluster_stats = clusters.cluster_stats(grid_extrem)


    # Figure
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    ax = plt.gca()

    # Start with 2019
    year = 2020
    in_year = times.year == year
    grid_year = grid_extrem[in_year]


    # Finish
    ax.legend(fontsize=15.)
    plot_utils.set_fontsize(ax, 17)

    #ax.scatter(grid_year.lon, grid_year.time)
    ax.scatter(grid_extrem.lon, grid_extrem.time)

    ax.set_xlabel('lon')
    ax.set_ylabel('date')

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)


    # PDF CDFs
    if flg & (2**0):
        line = '90'
        #fig_pdf_cdf(f'fig_pdf_cdf_{line}.png', line)
        fig_pdf_cdf(f'fig_pdf_cdf_{line}_105.png', line, SO_cut=1.05)

    # PDF CDF vary SO_cut
    if flg & (2**1):
        line = '90'
        fig_varySO_pdf_cdf(f'fig_varySO_pdf_cdf_{line}.png', line)

    # Time-series
    if flg & (2**2):
        line = '90'
        fig_timeseries(f'fig_varySO_pdf_cdf_{line}.png', line)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- PDF CDF
        #flg += 2 ** 1  # 2 -- Vary SO cut
        #flg += 2 ** 2  # 4 -- time-series of outliers
    else:
        flg = sys.argv[1]

    main(flg)