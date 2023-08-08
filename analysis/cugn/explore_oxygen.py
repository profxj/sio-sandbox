# imports
from importlib import reload
import os
import xarray

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

import pandas

from siosandbox.cugn import oxygen
from siosandbox.cugn import figures
from siosandbox.cugn import space_time


def inter_annual(ds):
    # Total map
    med_oxyT, xedges, yedges, countsT, indices = oxygen.gen_map(ds)

    # Early
    ds_early = space_time.cut_on_dates(ds, '2017-01-01', '2020-01-01')
    med_oxyE, xedges, yedges, countsE, indices = oxygen.gen_map(ds_early)

    diffET = med_oxyE - med_oxyT
    figures.show_grid(xedges, yedges, diffET/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='2017-2020', cmap='bwr', 
                 outfile='Figures/oxy_inter_annual_2017_2020.png',
                  vmnx=(-0.2,0.2))

    # Late
    ds_late = space_time.cut_on_dates(ds, '2020-01-01', '2029-01-01')
    med_oxyL, xedges, yedges, countsL, indices = oxygen.gen_map(ds_late)

    diffLT = med_oxyL - med_oxyT
    figures.show_grid(xedges, yedges, diffLT/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='2020-', cmap='bwr', 
                 outfile='Figures/oxy_inter_annual_2020+.png',
                  vmnx=(-0.2,0.2))

def seasonal(ds):
    # Total map
    med_oxyT, xedges, yedges, countsT, indices = oxygen.gen_map(ds)

    # Summer
    ds_summer = space_time.cut_on_months(ds, 6, 8)
    med_oxyS, xedges, yedges, countsS, indices = oxygen.gen_map(ds_summer)

    diffST = med_oxyS - med_oxyT
    figures.show_grid(xedges, yedges, diffST/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='Summer (June-August)', cmap='bwr', 
                 outfile='Figures/oxy_summer.png',
                  vmnx=(-0.2,0.2))

    # Winter
    ds_winter = space_time.cut_on_months(ds, 12, 2)
    med_oxyW, xedges, yedges, countsW, indices = oxygen.gen_map(ds_winter)

    diffWT = med_oxyW - med_oxyT
    figures.show_grid(xedges, yedges, diffWT/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='Winter (December-February)', cmap='bwr', 
                 outfile='Figures/oxy_winter.png',
                  vmnx=(-0.2,0.2))

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    data_path = '/home/xavier/Projects/Oceanography/Spray/CUGN'
    datafile = 'CUGN_potential_line_90.nc'
    ds = xarray.load_dataset(os.path.join(data_path, datafile))

    # Inter annual
    if flg & (2**0):
        inter_annual(ds)

    # Seasonal
    if flg & (2**1):
        seasonal(ds)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Inter annual
        #flg += 2 ** 1  # 2 -- Seasonal
    else:
        flg = sys.argv[1]

    main(flg)