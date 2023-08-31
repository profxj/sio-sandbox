""" Methods for clustering the outliers 
and then analyzing these clusters """

import numpy as np
import pandas 

from sklearn.cluster import AgglomerativeClustering, DBSCAN

from siosandbox.cugn import grid_utils

from IPython import embed

def generate_clusters(line:str, perc:float,
                      time_scl:float=3.,
                      z_scl:float=5.):
    """ Generate clusters of outliers for a given line
    and percentage

    Args:
        line (str): Line
        perc (float): percentile for outliers
        time_scl (float, optional): _description_. Defaults to 3..
        z_scl (float, optional): _description_. Defaults to 5..

    Returns:
        pandas.DataFrame: table of outliers labeled by cluster
    """

    # Grab table of outliers
    grid_outliers, grid_tbl, ds = grid_utils.gen_outliers(line, perc)

    # ###########
    # Time
    ptimes = pandas.to_datetime(grid_outliers.time.values)
    mjd = ptimes.to_julian_date()

    # Offset
    t = mjd - mjd.min()
    # Scale
    tscl = t / time_scl

    # Longitdue
    dl = (grid_outliers.lon.max() - grid_outliers.lon) 
    lscl = dl.values * 100 / dl.max()

    # Depth
    zscl = grid_outliers.z.values / z_scl

    # Package for sklearn
    X = np.zeros((len(grid_outliers), 3))
    X[:,0] = tscl
    X[:,1] = lscl
    X[:,2] = zscl

    # Fit
    dbscan = DBSCAN(eps=3, min_samples=5)
    dbscan.fit(X)
    print(f"Found {len(np.unique(dbscan.labels_))} unique clusters")

    grid_outliers['cluster'] = dbscan.labels_

    # Save?

    # Return
    return grid_outliers

def cluster_stats(grid_outliers:pandas.DataFrame):

    # Stats
    cluster_IDs = np.unique(grid_outliers.cluster.values[
        grid_outliers.cluster.values >= 0])

    # Loop on clusters
    stats = {}
    mean_keys = ['z', 'lon','doxy', 'time', 'SA', 'CT', 
                 'sigma0', 'SO', 'chla']
    for key in mean_keys:
        stats[key] = []
    max_keys = ['doxy', 'SO', 'chla']
    for key in max_keys:
        stats['max_'+key] = []
    # A few others
    stats['N'] = []

    for cluster_ID in cluster_IDs:
        # Grab em
        in_cluster = grid_outliers.cluster.values == cluster_ID
        stats['N'].append(in_cluster.sum())

        # Means
        for key in mean_keys:
            stats[key].append(grid_outliers[in_cluster][key].mean())

        # Max
        for key in max_keys:
            stats['max_'+key].append(grid_outliers[in_cluster][key].max())

    # Package
    stats_tbl = pandas.DataFrame(stats)
    stats_tbl['cluster'] = cluster_IDs

    return stats_tbl