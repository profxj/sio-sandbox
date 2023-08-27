""" Methods for clustering the outliers 
and then analyzing these clusters """

import numpy as np
import pandas 

from sklearn.cluster import AgglomerativeClustering, DBSCAN

from siosandbox.cugn import grid_utils

def generate_clusters(line:str, perc:float,
                      time_scl:float=3.,
                      z_scl:float=5.):

    # Grab table of outliers
    grid_outliers, grid_tbl, ds = grid_utils.gen_outliers(line, perc)

    # ###########
    # Time
    ptimes = pandas.to_datetime(grid_outliers.times.values)
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