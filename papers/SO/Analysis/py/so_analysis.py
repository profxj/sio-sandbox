""" Analsis related to SO paper """

import numpy as np

import pandas

from siosandbox.cugn import io as cugn_io

from IPython import embed

lines =  ['56', '66', '80', '90']

def frac_within_x_days(line:str, dt_days:int=5):

    # Load
    items = cugn_io.load_up(line, skip_dist=True)
    grid_extrem = items[0]
    times = items[2]

    dt_pd = pandas.to_timedelta(dt_days, unit='d')

    # Loop through the profiles
    uni_prof = np.unique(grid_extrem.profile)
    n_prof = len(uni_prof)

    n_within = 0
    max_ddist = 0.
    for prof in uni_prof:
        itime = times[grid_extrem.profile == prof][0]
        # Other
        other_times = times[grid_extrem.profile != prof]
        min_time = np.min(np.abs(itime - other_times))
        #
        if min_time < dt_pd:
            n_within += 1

        # Minimum distance
        if 'dist' not in grid_extrem.keys():
            continue
        imin_time = np.argmin(np.abs(itime - other_times))
        ddist = np.abs(grid_extrem.dist[grid_extrem.profile != prof][imin_time] 
                       - grid_extrem.dist[grid_extrem.profile == prof][0])
        if ddist > max_ddist:
            max_ddist = ddist
        embed(header='45 of so_analysis.py')

    # Stats
    print("=====================================")
    print(f"Line {line}")
    print(f"Found {n_within} of {n_prof} profiles within {dt_days} days")
    print(f"Frac = {n_within/n_prof}")
    print(f"Max distance = {max_ddist}")

# Command line execution
if __name__ == '__main__':

    # Clustering
    for line in lines:
        frac_within_x_days(line, dt_days=1)