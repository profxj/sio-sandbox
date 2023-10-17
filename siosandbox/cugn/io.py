""" I/O for CUGN data and analysis """
import os
import numpy as np
import xarray
import pandas

from siosandbox.cugn import grid_utils
from siosandbox.cugn import clusters
from siosandbox.cugn import utils as cugn_utils
from siosandbox import cat_utils

from IPython import embed

data_path = os.getenv('CUGN')

def line_files(line:str):

    datafile = os.path.join(data_path, f'CUGN_potential_line_{line}.nc')
    gridtbl_file = os.path.join(data_path, f'doxy_grid_line{line}.parquet')
    edges_file = os.path.join(data_path, f'doxy_edges_line{line}.npz')

    # dict em
    lfiles = dict(datafile=datafile, 
                  gridtbl_file=gridtbl_file, 
                  edges_file=edges_file)
    # Return
    return lfiles
    
def load_line(line:str):
    # Files
    lfiles = line_files(line)

    grid_tbl = pandas.read_parquet(lfiles['gridtbl_file'])
    ds = xarray.load_dataset(lfiles['datafile'])
    edges = np.load(lfiles['edges_file'])


    # dict em
    items = dict(ds=ds, grid_tbl=grid_tbl, edges=edges)

    return items



def load_up(line:str, skip_dist:bool=False):
    # Load
    items = load_line(line)
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

    # Fill in N_p, chla_p
    grid_utils.find_perc(grid_tbl, 'N')
    grid_utils.find_perc(grid_tbl, 'chla')

    dp_gt = grid_tbl.depth*100000 + grid_tbl.profile
    dp_ge = grid_extrem.depth*100000 + grid_extrem.profile
    ids = cat_utils.match_ids(dp_ge, dp_gt, require_in_match=True)
    assert len(np.unique(ids)) == len(ids)

    grid_extrem['N_p'] = grid_tbl.N_p.values[ids]
    grid_extrem['chla_p'] = grid_tbl.chla_p.values[ids]

    # Add to df
    grid_extrem['year'] = times.year
    grid_extrem['doy'] = times.dayofyear

    # Add distance from shore
    if not skip_dist:
        dist, _ = cugn_utils.calc_dist_offset(
            line, grid_extrem.lon.values, grid_extrem.lat.values)
        grid_extrem['dist'] = dist

    # Cluster me
    clusters.generate_clusters(grid_extrem)
    cluster_stats = clusters.cluster_stats(grid_extrem)

    return grid_extrem, ds, times, grid_tbl