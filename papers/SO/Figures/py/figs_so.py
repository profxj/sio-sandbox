# imports
from importlib import reload
import os
import xarray

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
mpl.rcParams['font.family'] = 'stixgeneral'


import seaborn as sns

import pandas

from siosandbox.cugn import grid_utils
from siosandbox.cugn import utils as cugn_utils
from siosandbox.cugn import figures
from siosandbox.cugn import clusters
from siosandbox.cugn import io as cugn_io
from siosandbox import plot_utils
from siosandbox import cat_utils

from IPython import embed

def load_up(line):
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

    # Fill in N_p
    grid_utils.find_perc(grid_tbl, 'N')
    dp_gt = grid_tbl.depth*100000 + grid_tbl.profile
    dp_ge = grid_extrem.depth*100000 + grid_extrem.profile
    ids = cat_utils.match_ids(dp_ge, dp_gt, require_in_match=True)
    assert len(np.unique(ids)) == len(ids)

    grid_extrem['N_p'] = grid_tbl.N_p.values[ids]

    # Add to df
    grid_extrem['year'] = times.year
    grid_extrem['doy'] = times.dayofyear

    # Add distance from shore
    dist, _ = cugn_utils.calc_dist_offset(
        line, grid_extrem.lon.values, grid_extrem.lat.values)
    grid_extrem['dist'] = dist

    # Cluster me
    clusters.generate_clusters(grid_extrem)
    cluster_stats = clusters.cluster_stats(grid_extrem)

    return grid_extrem, ds, times, grid_tbl

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


def fig_timeseries(outfile:str, line, vmax=1.3):

    # Do it
    items = load_up(line)
    grid_extrem = items[0]

    # Figure
    # https://stackoverflow.com/questions/57292022/day-of-year-format-x-axis-matplotlib
    fig = plt.figure(figsize=(12,12))
    _, axs = plt.subplots(1,3,sharey=True,gridspec_kw = {'wspace':0, 'hspace':0})


    # Loop in the years
    for ss, year in enumerate([2018, 2019, 2020]):
        in_year = grid_extrem.year == year
        grid_year = grid_extrem[in_year]

        ax = axs[ss]
        sc = ax.scatter(grid_year.dist, grid_year.doy,
                c=grid_year.SO, cmap='jet', s=1, vmin=1.1,
                vmax=vmax)

        if ss == 1:
            ax.set_xlabel('Distance from shore (km)')
        if ss == 0:
            #ax.set_ylabel('Date')
            ax.set_ylim(0., 366.)
            major_format = mdates.DateFormatter('%b')
            ax.yaxis.set_major_formatter(major_format)

        fsz = 13.
        ax.text(0.95, 0.9, f'{year}',
                transform=ax.transAxes,
                fontsize=fsz, ha='right', color='k')
        ax.set_xlim(-20., 399.)

        # Finish
        plot_utils.set_fontsize(ax, 13)

    cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    cbaxes = plt.colorbar(sc, cax=cax, **kw)
    cbaxes.set_label('Saturated Oxygen', fontsize=13.)
    cbaxes.ax.tick_params(labelsize=13)


    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_event(outfile:str, line:str, event:str, p_off:int=15,
    max_depth=10):

    items = load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]



    tevent = pandas.Timestamp(event)
    imin = np.argmin(np.abs(times-tevent))
    cidx = grid_extrem.iloc[imin].cluster

    # Range of dates
    in_cluster = grid_extrem.cluster == cidx

    p_min = grid_extrem.profile[in_cluster].min() - p_off
    p_max = grid_extrem.profile[in_cluster].max() + p_off

    # DOY
    otimes = pandas.to_datetime(ds.time.data)
    x_lims = mdates.date2num([otimes[p_min], otimes[p_max]])


    csz = 13.
    # Figure
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(2,2)

    # #########################################################3
    # DO
    ax_DO = plt.subplot(gs[0])
    # Contours from SO
    im = ax_DO.imshow(ds.doxy.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='Purples', vmin=200., aspect='auto')
    #im = ax.imshow(ds.SO.data[0:max_depth, p_min:p_max],
    #    extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
    #               cmap='jet', vmin=0.9, aspect='auto')
    cax,kw = mpl.colorbar.make_axes([ax_DO])
    cbaxes = plt.colorbar(im, cax=cax, **kw)
    cbaxes.set_label('Dissolved Oxygen', fontsize=csz)

    # #########################################################3
    # SO contour
    SOs = ds.SO.data[0:max_depth, p_min:p_max]
    Np = p_max-p_min
    X = np.outer(np.ones(max_depth), 
        np.linspace(x_lims[0], x_lims[1], Np))
    Y = np.outer(np.linspace(0., max_depth*10., max_depth), 
                 np.ones(Np))

    ax_DO.contour(X, Y, SOs, levels=[1., 1.1, 1.2],
               colors=['white', 'gray', 'black'], linewidths=1.5)

    '''
    # #########################################################3
    # DO percentile contour
    in_view = (grid_tbl.profile >= p_min) & (grid_tbl.profile <= p_max) & (
        grid_tbl.depth < max_depth)
    doxy_p_grid = np.zeros_like(ds.N.data)
    for _, row in grid_tbl[in_view].iterrows():
        doxy_p_grid[row.depth, row.profile] = row.doxy_p
    ax_DO.contour(X, Y, doxy_p_grid[0:max_depth, p_min:p_max], 
                  levels=[90., 95.],
               colors=['white', 'black'], linewidths=1.5)
    '''

    # #########################################################3
    # N
    ax_N = plt.subplot(gs[1])
    im_N = ax_N.imshow(ds.N.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='Blues', aspect='auto')

    # N percentile
    grid_utils.find_perc(grid_tbl, 'N')
    Np_grid = np.zeros_like(ds.N.data)

    in_view = (grid_tbl.profile >= p_min) & (grid_tbl.profile <= p_max) & (
        grid_tbl.depth < max_depth)
    # Painful loop, but do it
    #embed(header='214 of figs_so')
    for _, row in grid_tbl[in_view].iterrows():
        Np_grid[row.depth, row.profile] = row.N_p

    ax_N.contour(X, Y, Np_grid[0:max_depth, p_min:p_max], 
                  levels=[90., 95.],
               colors=['gray', 'black'], linewidths=1.5)
    

    cax,kw = mpl.colorbar.make_axes([ax_N])
    cbaxes = plt.colorbar(im_N, cax=cax, **kw)
    cbaxes.set_label('Buoyancy (cycles/hour)', fontsize=csz)

    # T
    ax_T = plt.subplot(gs[2])
    im_T = ax_T.imshow(ds.temperature.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='jet', aspect='auto')
    cax,kw = mpl.colorbar.make_axes([ax_T])
    cbaxes = plt.colorbar(im_T, cax=cax, **kw)
    cbaxes.set_label('Temperature (deg C)', fontsize=csz)

    # T
    ax_C = plt.subplot(gs[3])
    im_C = ax_C.imshow(ds.chlorophyll_a.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='Greens', aspect='auto')
    cax,kw = mpl.colorbar.make_axes([ax_C])
    cbaxes = plt.colorbar(im_C, cax=cax, **kw)
    cbaxes.set_label('Chla', fontsize=csz)


    # Finish
    for ss in range(4):
        ax = plt.subplot(gs[ss])
        #
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date')
        # Axes
        ax.xaxis_date()
        #major_format = mdates.DateFormatter('%b')
        #ax.xaxis.set_major_formatter(major_format)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))

        ax.tick_params(axis='x', rotation=10)
        #ax.autofmt_xdate()
        plot_utils.set_fontsize(ax, 14)

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_percentiles(outfile:str, line:str):

    # Load
    items = load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Figure
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    ax = plt.gca()

    #sns.histplot(data=grid_extrem, x='doxy_p', y='N_p', ax=ax)

    jg = sns.jointplot(data=grid_extrem, x='doxy_p', y='N_p',
                        kind='hex', bins='log', # gridsize=250, #xscale='log',
                       # mincnt=1,
                       cmap=plt.get_cmap('autumn'), 
                       marginal_kws=dict(fill=False, color='black', 
                                         bins=100)) 

    plt.colorbar()
    # Axes                                 
    jg.ax_joint.set_xlabel('Buoyancy Percentile')
    jg.ax_joint.set_ylabel('DO Percentile')
    plot_utils.set_fontsize(jg.ax_joint, 14)
    #jg.ax_joint.set_ylim(ymnx)
    
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
        fig_timeseries(f'fig_timeseries_{line}.png', line)

    # Events
    if flg & (2**3):
        line = '90'
        #event = '2020-09-01' # Sub-surface
        event = '2019-08-15'
        fig_event(f'fig_event_{line}_{event}.png', line, event)

    # Percentiles of DO and N
    if flg & (2**4):
        line = '90'
        fig_percentiles(f'fig_percentiles_{line}.png', line)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- PDF CDF
        #flg += 2 ** 1  # 2 -- Vary SO cut
        #flg += 2 ** 2  # 4 -- time-series of outliers
        #flg += 2 ** 3  # 8 -- Show individual events
        #flg += 2 ** 4  # 16 -- Percentiles
    else:
        flg = sys.argv[1]

    main(flg)