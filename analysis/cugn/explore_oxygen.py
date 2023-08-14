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
from siosandbox.cugn import stats
from siosandbox.cugn import figures
from siosandbox.cugn import space_time

from IPython import embed

def total(ds):

    # Total maps
    med_oxyT, xedges, yedges, countsT, indices, _, _ = oxygen.gen_map(ds)

    figures.show_grid(xedges, yedges, med_oxyT,
              ('Absolute Salinity', 'Potential Density'),
              r'Median Dissolved Oxygen', 
                 title='All', cmap='jet', 
                 outfile='Figures/oxy_median_all.png')


    rms_oxyT, xedges, yedges, countsT, indices, _, _ \
        = oxygen.gen_map(ds, stat='std')

    figures.show_grid(xedges, yedges, rms_oxyT,
              ('Absolute Salinity', 'Potential Density'),
              r'RMS(Dissolved Oxygen)', 
                 title='All', cmap='jet', 
                 outfile='Figures/oxy_rms_all.png')


def inter_annual(ds):
    # Total map
    med_oxyT, xedges, yedges, countsT, indices, _, _ = oxygen.gen_map(ds)

    # Early
    ds_early = space_time.cut_on_dates(ds, '2017-01-01', '2020-01-01')
    med_oxyE, xedges, yedges, countsE, indices, _, _ = oxygen.gen_map(ds_early)

    diffET = med_oxyE - med_oxyT
    figures.show_grid(xedges, yedges, diffET/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='2017-2020', cmap='bwr', 
                 outfile='Figures/oxy_inter_annual_2017_2020.png',
                  vmnx=(-0.2,0.2))

    # Late
    ds_late = space_time.cut_on_dates(ds, '2020-01-01', '2029-01-01')
    med_oxyL, xedges, yedges, countsL, indices, _, _ = oxygen.gen_map(ds_late)

    diffLT = med_oxyL - med_oxyT
    figures.show_grid(xedges, yedges, diffLT/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='2020-', cmap='bwr', 
                 outfile='Figures/oxy_inter_annual_2020+.png',
                  vmnx=(-0.2,0.2))

def seasonal(ds):
    # Total map
    med_oxyT, xedges, yedges, countsT, indices, _, _ = oxygen.gen_map(ds)

    # Summer
    ds_summer = space_time.cut_on_months(ds, 6, 8)
    med_oxyS, xedges, yedges, countsS, indices, _, _ = oxygen.gen_map(ds_summer)

    diffST = med_oxyS - med_oxyT
    figures.show_grid(xedges, yedges, diffST/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='Summer (June-August)', cmap='bwr', 
                 outfile='Figures/oxy_summer.png',
                  vmnx=(-0.2,0.2))

    # Winter
    ds_winter = space_time.cut_on_months(ds, 12, 2)
    med_oxyW, xedges, yedges, countsW, indices, _, _ = oxygen.gen_map(ds_winter)

    diffWT = med_oxyW - med_oxyT
    figures.show_grid(xedges, yedges, diffWT/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='Winter (December-February)', cmap='bwr', 
                 outfile='Figures/oxy_winter.png',
                  vmnx=(-0.2,0.2))

def longitude(ds):
    
    # Total map
    med_oxyT, xedges, yedges, countsT, indices, gd_oxy, _ = oxygen.gen_map(ds)

    # Inshore
    ds_inshore = space_time.cut_on_lon(ds, -119., -115.)
    med_oxyI, xedges, yedges, countsI, indices, _, _ = oxygen.gen_map(ds_inshore)

    diffIT = med_oxyI - med_oxyT
    figures.show_grid(xedges, yedges, diffIT/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='Inshore (lon > -119deg)', cmap='bwr', 
                 outfile='Figures/oxy_inshore.png',
                  vmnx=(-0.2,0.2))

    # Offshore
    ds_offshore = space_time.cut_on_lon(ds, -130., -119.5)
    med_oxyO, xedges, yedges, _, _, _, _ = oxygen.gen_map(ds_offshore)

    diffOT = med_oxyO - med_oxyT
    figures.show_grid(xedges, yedges, diffOT/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='Offshore (lon < -119.5 deg)', cmap='bwr', 
                 outfile='Figures/oxy_offshore.png',
                  vmnx=(-0.2,0.2))

def gaussianity(ds):
    # Generate the grids
    mean_oxyT, xedges, yedges, countsT, \
        indices, gd_oxy, _ = oxygen.gen_map(ds, stat='mean')

    # RMS
    rms_oxyT, xedges, yedges, countsT, \
        indices, _, _ = oxygen.gen_map(ds, stat='std')

    # Check gaussianity
    p_values = stats.chk_grid_gaussianity(gd_oxy, mean_oxyT, rms_oxyT, indices,
                               countsT)

    figures.show_grid(xedges, yedges, np.log10(p_values),
              ('Absolute Salinity', 'Potential Density'),
              r'log10 KS p-value for Normality',
                 cmap='bone', 
                 outfile='Figures/oxy_gaussianity.png',
                 vmnx=(-3,0.))

def outliers(ds):

    # Generate the grid
    mean_oxyT, xedges, yedges, countsT, \
        grid_indices, gd_oxy, da_gd = oxygen.gen_map(ds, stat='mean')

    # Outliers
    outliers = stats.find_outliers(gd_oxy, grid_indices, countsT,
        95., da_gd, ds=ds)

    times = pandas.to_datetime(ds.time[outliers[:,1]])
    months = times.month.values.astype(int)

    lons = ds.lon[outliers[:,1]].values

    # Plot a year
    embed(header='164 of oxy.py')

    year = 2017
    in_year = times.year == year

    all_gd = in_year
    
    plt.clf()
    ax = plt.gca()

    sc = ax.scatter(lons[all_gd], 
                    #ds.depth[outliers[:,0]].values[all_gd],
                    ds.sigma0.data[(outliers[:,0], outliers[:,1])][all_gd],
               c=months[all_gd], cmap='tab20', s=0.8)

    cb = plt.colorbar(sc)
    cb.set_label('Month', fontsize=15.)
    #ax.set_ylim(500,0)

    ax.set_xlabel('Longitude (deg)')
    #ax.set_ylabel('Depth (m)')
    ax.set_ylabel('Potential Density')
    ax.set_title(f'{year} Outliers')

    plt.savefig(f'Figures/oxy_outliers_{year}.png', dpi=300)
    plt.show()

    embed(header='157 of oxy.py')


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

    # Longitude
    if flg & (2**2):
        longitude(ds)

    # Gaussianity
    if flg & (2**3):
        gaussianity(ds)

    # Total
    if flg & (2**4):
        total(ds)

    # Outliers in sigma, AS 
    if flg & (2**5):
        outliers(ds)




# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Inter annual
        #flg += 2 ** 1  # 2 -- Seasonal
        #flg += 2 ** 2  # 4 -- In/off shore
        #flg += 2 ** 3  # 8 -- Gaussiantiy
        #flg += 2 ** 4  # 16 -- Total
        #flg += 2 ** 5  # 32 -- Outliers
    else:
        flg = sys.argv[1]

    main(flg)