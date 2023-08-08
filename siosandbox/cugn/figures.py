
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
              afsz:float=14.):

    # Cut on counts
    if min_counts is not None:
        grid[counts < min_counts] = np.nan

    # Figure
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
        plt.show()

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
