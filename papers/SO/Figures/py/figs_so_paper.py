""" Final figures for the paper 

Other figures are in figs_so.py

"""

# imports
from importlib import reload
import os
import xarray

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d 

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator 

mpl.rcParams['font.family'] = 'stixgeneral'


import seaborn as sns

import pandas

from siosandbox.cugn import grid_utils
from siosandbox.cugn import utils as cugn_utils
from siosandbox.cugn import figures
from siosandbox.cugn import clusters
from siosandbox.cugn import defs as cugn_defs
from siosandbox.cugn import io as cugn_io
from siosandbox import plot_utils
from siosandbox import cat_utils

from gsw import conversions, density
import gsw

from IPython import embed

lines = cugn_defs.lines
line_colors = cugn_defs.line_colors
line_cmaps = cugn_defs.line_cmaps


def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)


def fig_joint_pdfs(use_density:bool=False):


    if use_density:
        axes=('SA', 'sigma0')
        outfile = 'fig_paper_jointPDFs_density.png'
        lbl = r'$\log_{10} \, p(S_A,\sigma)$'
        ylbl = 'Potential Density (kg/m$^3$)'
        ypos = 0.9
    else:
        axes=('SA', 'CT')
        outfile = 'fig_paper_jointPDFs.png'
        lbl = r'$\log_{10} \, p(S_A,\theta)$'
        ylbl = 'Conservative Temperature (C)'
        ypos = 0.1


    fig = plt.figure(figsize=(12,10))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    all_ax = []
    for ss, cmap, line in zip(range(4), line_cmaps, lines):

        # Load
        items = cugn_io.load_line(line)
        ds = items['ds']

        # Oxygen
        mean_oxy, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
            ds, axes=axes, stat='mean', variable='doxy')

        # PDF
        dx = xedges[1] - xedges[0]
        dy = yedges[1] - yedges[0]

        p_norm = np.sum(counts) * (dx * dy)
        consv_pdf = counts / p_norm

        # #####################################################
        # PDF
        ax_pdf = plt.subplot(gs[ss])
        img = ax_pdf.pcolormesh(xedges, yedges, np.log10(consv_pdf.T),
                                cmap=cmap)
        gen_cb(img, lbl)
        all_ax.append(ax_pdf)

    for ss, line in enumerate(lines):
        ax = all_ax[ss]
        fsz = 17.
        ax.set_xlabel('Absolute Salinity (g/kg)')                    
        ax.set_ylabel(ylbl)
        # Set x-axis interval to 0.5
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        # 
        plot_utils.set_fontsize(ax, fsz)
        ax.text(0.05, ypos, f'Line={line}',
                transform=ax.transAxes,
                fontsize=fsz, ha='left', color='k')
        # Grid lines
        ax.grid()
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_SO_cdf(outfile:str):

    # Figure
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    for clr, line in zip(line_colors, lines):
        # Load
        items = cugn_io.load_up(line, skip_dist=True)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        grid_tbl = items[3]

        for ss, depth in enumerate([0,1]):
            ax = plt.subplot(gs[ss])
            # Cut
            cut_depth = grid_tbl.depth == depth
            grid_plt = grid_tbl[cut_depth]

            # Plot CDF
            if ss == 0:
                lbl = f'Line {line}'
            else:
                lbl = None
            sns.ecdfplot(x=grid_plt.SO, ax=ax, label=lbl, color=clr)

            # Stats
            srt = np.argsort(grid_plt.SO.values)
            cdf = np.arange(len(grid_plt.SO))/len(grid_plt.SO)
            idx = np.argmin(np.abs(cdf-0.95))
            print(f'95% for {line} {depth}m: {grid_plt.SO.values[srt][idx]}')

    # Finish
    for ss, depth in enumerate([0,1]):
        ax = plt.subplot(gs[ss])
        ax.axvline(1., color='black', linestyle='--')
        ax.axvline(1.1, color='black', linestyle=':')

        ax.set_xlim(0.5, 1.4)
        ax.set_xlabel('Saturated Oxygen')
        ax.set_ylabel('CDF')
                 #label=f'SO > {SO_cut}', log_scale=log_scale)
        ax.text(0.95, 0.05, f'depth={(depth+1)*10}m',
                transform=ax.transAxes,
                fontsize=15, ha='right', color='k')
        plot_utils.set_fontsize(ax, 15)

    ax = plt.subplot(gs[0])
    ax.legend(fontsize=15., loc='upper left')

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_dist_doy(outfile:str, line:str, color:str,
                 show_legend:bool=False):

    # Figure
    #sns.set()
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    #ax = plt.gca()

    # Load
    items = cugn_io.load_up(line)
    grid_extrem = items[0]
    #ds = items[1]
    #times = items[2]
    #grid_tbl = items[3]

    jg = sns.jointplot(data=grid_extrem, x='dist', 
                    y='doy', color=color,
                    marginal_kws=dict(fill=False, color='black', 
                                        bins=100)) 


    # Scatter plot time
    markers = ['o','x','v','s','*']
    jg.ax_joint.cla()

    for depth in range(5):
        at_depth = grid_extrem.depth == depth
        # Scatter plot
        if show_legend:
            label=f'z={(depth+1)*10}m'
        else:
            label = None
        if depth != 1:
            fc = 'none'
        else:
            fc = color
        jg.ax_joint.scatter(grid_extrem[at_depth].dist, grid_extrem[at_depth].doy, 
            marker=markers[depth], label=label, facecolors=fc,
            edgecolors=color)#, s=50.)
    
    # Axes                                 
    jg.ax_joint.set_ylabel('DOY')
    jg.ax_joint.set_xlabel('Distance from shore (km)')
    jg.ax_joint.set_ylim(0., 365.)

    xmin = -20. if line != '80' else -60.
    xmax = max(500., grid_extrem.dist.max())
    jg.ax_joint.set_xlim(xmin, xmax)

    fsz = 14.
    jg.ax_joint.text(0.95, 0.95, f'Line {line}',
                transform=jg.ax_joint.transAxes,
                fontsize=fsz, ha='right', color='k')
    plot_utils.set_fontsize(jg.ax_joint, 14)
    if show_legend:
        jg.ax_joint.legend(fontsize=13., loc='lower right')

    
    #gs.tight_layout(fig)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)


    # Figure 1 -- Joint PDFs
    if flg & (2**0):
        fig_joint_pdfs()
        fig_joint_pdfs(use_density=True)

    # Figure 2 -- SO CDFs
    if flg & (2**1):
        fig_SO_cdf('fig_SO_cdf.png')

    # Figure 3 -- DOY vs Offshore distance
    if flg & (2**2):
        for line, clr in zip(lines, line_colors):
            # Skip for now
            if line == '56':
                continue
            if line == '66':
                show_legend = True
            else:
                show_legend = False
            fig_dist_doy(f'fig_dist_doy_{line}.png', line, clr,
                         show_legend=show_legend)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Joint PDFs of all 4 lines
        #flg += 2 ** 1  # 2 -- SO CDF
        #flg += 2 ** 2  # 3 -- DOY vs Offshore, 1 by 1
    else:
        flg = sys.argv[1]

    main(flg)