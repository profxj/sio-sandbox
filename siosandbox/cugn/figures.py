
# imports
import os
import xarray

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

import pandas

def show_grid(xedges:np.ndarray, yedges:np.ndarray, 
              grid:np.ndarray, 
              axis_labels:tuple,
              cb_label:str,
              outfile:str=None,
              counts:np.ndarray=None,
              min_counts:int=None, 
              cmap:str='jet', vmnx:tuple=(None,None),
              title:str=None,
              afsz:float=14.,
              show:bool=True,
              ax=None):

    # Cut on counts
    if min_counts is not None:
        grid[counts < min_counts] = np.nan

    # Figure
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        plt.clf()
        ax = plt.gca()
    #
    img = ax.pcolormesh(xedges, yedges, grid.T, cmap=cmap,
                        vmin=vmnx[0], vmax=vmnx[1]) 

    # colorbar
    cb = plt.colorbar(img, pad=0., fraction=0.030)
    cb.set_label(cb_label, fontsize=14.)

    #
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    # Title?
    if title is not None:
        ax.set_title(title, fontsize=16.)

    set_fontsize(ax, afsz)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
        print(f'Saved: {outfile}')
    else:
        if show:
            plt.show()
    return ax

def set_fontsize(ax, fsz):
    """
    Set the fontsize throughout an Axis

    Args:
        ax (Matplotlib Axis):
        fsz (float): Font size

    Returns:

    """
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)


def outlier_by_months(outfile:str, pcut:float, year:int,
                      lons, sigma0, months,
                      show:bool=False, ax=None):

    
    if ax is None:
        plt.clf()
        ax = plt.gca()

    sc = ax.scatter( lons,
                    #ds.depth[outliers[:,-1]].values[all_gd],
        sigma0,
        c=months, cmap='tab20', s=0.8)

    cb = plt.colorbar(sc)
    cb.set_label('Month', fontsize=14.)
    #ax.set_ylim(499,0)

    ax.set_xlabel('Longitude (deg)')
    #ax.set_ylabel('Depth (m)')
    ax.set_ylabel('Potential Density')
    sign = '>' if pcut > 49. else '<'
    ax.set_title(f'{year} Outliers: {sign} {int(pcut)}th percentile')

    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        print("Saved: ", outfile)
    if show:
        plt.show()