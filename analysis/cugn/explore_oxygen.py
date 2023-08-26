""" This was the first sandbox.  Things were refactored
such that this code is largely deprecated """
# imports
from importlib import reload
import os
import xarray

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import seaborn as sns

import pandas

from siosandbox.cugn import grid_utils
from siosandbox.cugn import figures
from siosandbox.cugn import space_time

from IPython import embed

def total(ds):

    # Total maps
    med_oxyT, xedges, yedges, countsT, indices, _, _ = grid_utils.gen_grid(ds)

    figures.show_grid(xedges, yedges, med_oxyT,
              ('Absolute Salinity', 'Potential Density'),
              r'Median Dissolved Oxygen', 
                 title='All', cmap='jet', 
                 outfile='Figures/oxy_median_all.png')


    rms_oxyT, xedges, yedges, countsT, indices, _, _ \
        = grid_utils.gen_grid(ds, stat='std')

    figures.show_grid(xedges, yedges, rms_oxyT,
              ('Absolute Salinity', 'Potential Density'),
              r'RMS(Dissolved Oxygen)', 
                 title='All', cmap='jet', 
                 outfile='Figures/oxy_rms_all.png')


def inter_annual(ds):
    # Total map
    med_oxyT, xedges, yedges, countsT, indices, _, _ = grid_utils.gen_grid(ds)

    # Early
    ds_early = space_time.cut_on_dates(ds, '2017-01-01', '2020-01-01')
    med_oxyE, xedges, yedges, countsE, indices, _, _ = grid_utils.gen_grid(ds_early)

    diffET = med_oxyE - med_oxyT
    figures.show_grid(xedges, yedges, diffET/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='2017-2020', cmap='bwr', 
                 outfile='Figures/oxy_inter_annual_2017_2020.png',
                  vmnx=(-0.2,0.2))

    # Late
    ds_late = space_time.cut_on_dates(ds, '2020-01-01', '2029-01-01')
    med_oxyL, xedges, yedges, countsL, indices, _, _ = grid_utils.gen_grid(ds_late)

    diffLT = med_oxyL - med_oxyT
    figures.show_grid(xedges, yedges, diffLT/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='2020-', cmap='bwr', 
                 outfile='Figures/oxy_inter_annual_2020+.png',
                  vmnx=(-0.2,0.2))

def seasonal(ds):
    # Total map
    med_oxyT, xedges, yedges, countsT, indices, _, _ = grid_utils.gen_grid(ds)

    # Summer
    ds_summer = space_time.cut_on_months(ds, 6, 8)
    med_oxyS, xedges, yedges, countsS, indices, _, _ = grid_utils.gen_grid(ds_summer)

    diffST = med_oxyS - med_oxyT
    figures.show_grid(xedges, yedges, diffST/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='Summer (June-August)', cmap='bwr', 
                 outfile='Figures/oxy_summer.png',
                  vmnx=(-0.2,0.2))

    # Winter
    ds_winter = space_time.cut_on_months(ds, 12, 2)
    med_oxyW, xedges, yedges, countsW, indices, _, _ = grid_utils.gen_grid(ds_winter)

    diffWT = med_oxyW - med_oxyT
    figures.show_grid(xedges, yedges, diffWT/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='Winter (December-February)', cmap='bwr', 
                 outfile='Figures/oxy_winter.png',
                  vmnx=(-0.2,0.2))

def longitude(ds):
    
    # Total map
    med_oxyT, xedges, yedges, countsT, indices, gd_oxy, _ = grid_utils.gen_grid(ds)

    # Inshore
    ds_inshore = space_time.cut_on_lon(ds, -119., -115.)
    med_oxyI, xedges, yedges, countsI, indices, _, _ = grid_utils.gen_grid(ds_inshore)

    diffIT = med_oxyI - med_oxyT
    figures.show_grid(xedges, yedges, diffIT/med_oxyT, 
              ('Absolute Salinity', 'Potential Density'),
              r'$\Delta($Dissolved Oxygen) Fractional', 
                 title='Inshore (lon > -119deg)', cmap='bwr', 
                 outfile='Figures/oxy_inshore.png',
                  vmnx=(-0.2,0.2))

    # Offshore
    ds_offshore = space_time.cut_on_lon(ds, -130., -119.5)
    med_oxyO, xedges, yedges, _, _, _, _ = grid_utils.gen_grid(ds_offshore)

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
        indices, gd_oxy, _ = grid_utils.gen_grid(ds, stat='mean')

    # RMS
    rms_oxyT, xedges, yedges, countsT, \
        indices, _, _ = grid_utils.gen_grid(ds, stat='std')

    # Check gaussianity
    p_values = grid_utils.chk_grid_gaussianity(gd_oxy, mean_oxyT, rms_oxyT, indices,
                               countsT)

    figures.show_grid(xedges, yedges, np.log10(p_values),
              ('Absolute Salinity', 'Potential Density'),
              r'log10 KS p-value for Normality',
                 cmap='bone', 
                 outfile='Figures/oxy_gaussianity.png',
                 vmnx=(-3,0.))

def outliers(ds, year:int=2017, pcut:float=95.):

    # Generate the grid
    mean_oxyT, xedges, yedges, countsT, \
        grid_indices, gd_oxy, da_gd = grid_utils.gen_grid(ds, stat='mean')

    # Outliers
    outliers, _ = grid_utils.find_outliers(gd_oxy, grid_indices, countsT, pcut, da_gd)

    times = pandas.to_datetime(ds.time[outliers[:,1]])
    months = times.month.values.astype(int)

    lons = ds.lon[outliers[:,1]].values

    # Plot a year
    in_year = times.year == year

    all_gd = in_year

    outfile = f'Figures/oxy_outliers_{year}_p{int(pcut)}.png' 
    figures.outlier_by_months(outfile, pcut, year,
        lons[all_gd], 
        ds.sigma0.data[(outliers[:,0], outliers[:,1])][all_gd],
        months[all_gd])


def outlier_montage(ds, out_dict:dict, outfile:str):

    # Generate the grid
    mean_oxyT, xedges, yedges, countsT, \
        grid_indices, gd_oxy, da_gd = grid_utils.gen_grid(ds, stat='mean')

    # Outliers
    outliers, out_rowcol, out_cellidx = grid_utils.find_outliers(
        gd_oxy, grid_indices, countsT, out_dict['perc'], da_gd)

    # For the subset
    times = pandas.to_datetime(ds.time[outliers[:,1]])
    months = times.month.values.astype(int)
    lons = ds.lon[outliers[:,1]].values

    # Density, AS
    sigma_o = ds.sigma0.data[(outliers[:,0], outliers[:,1])]
    SA_o = ds.SA.data[(outliers[:,0], outliers[:,1])]
    DO_o = ds.doxy.data[(outliers[:,0], outliers[:,1])]

    # Subset
    idx_o = (times.year==out_dict['year']) & (
        lons >= out_dict['lons'][0]) & (
        lons <= out_dict['lons'][1]) & (
        sigma_o >= out_dict['sigma'][0]) & (
        sigma_o <= out_dict['sigma'][1])

    if 'month' in out_dict.keys():
        idx_o = idx_o & (
            months>=out_dict['month'][0]) & (
            months<=out_dict['month'][1])

    # Percentile of DO
    DO_perc = grid_utils.find_perc(
        gd_oxy, grid_indices, 
        out_rowcol[idx_o], out_cellidx[idx_o]) 

    # Chla
    gd_chla = ds.chlorophyll_a.data[da_gd]
    Chla_perc = grid_utils.find_perc(
        gd_chla, grid_indices, 
        out_rowcol[idx_o], out_cellidx[idx_o]) 

    # Back scatter
    gd_bs = ds.acoustic_backscatter.data[da_gd]
    bs_perc = grid_utils.find_perc(
        gd_bs, grid_indices, 
        out_rowcol[idx_o], out_cellidx[idx_o]) 

    fig = plt.figure(figsize=(12,12))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    # Year plot
    ax0 = plt.subplot(gs[0])
    in_year = times.year == out_dict['year']

    figures.outlier_by_months(
        None, out_dict['perc'], out_dict['year'],
        lons[in_year], 
        ds.sigma0.data[(outliers[:,0], outliers[:,1])][in_year],
        months[in_year], ax=ax0)
    ax0.plot(lons[idx_o],
        ds.sigma0.data[(outliers[:,0], outliers[:,1])][idx_o],
        'o', color='grey', alpha=0.5, mfc='none', ms=3)
    

    # ####################################
    # sigma/S distribution
    ax1 = plt.subplot(gs[1])
    figures.show_grid(xedges, yedges, np.log10(countsT),
              ('Absolute Salinity', 'Potential Density'),
              r'log10 Counts',
                 cmap='bone', show=False, ax=ax1) 
                 #vmnx=(-3,0.))
    # Add on outliers
    ax1.plot(SA_o[idx_o], sigma_o[idx_o], 'rx')

    # #################################
    # Percetnils of DO and Chlaa 
    ax2 = plt.subplot(gs[2])
    ax2.plot(DO_perc, Chla_perc, 'o')
    ax2.set_xlabel('DO Percentile')
    ax2.set_ylabel('Chla Percentile')

    # #################################
    # Percetnils of DO and BS
    ax3 = plt.subplot(gs[3])
    ax3.plot(DO_perc, bs_perc, 'og')
    ax3.set_xlabel('DO Percentile')
    ax3.set_ylabel('Backscatter Percentile')


    # Font sizes
    for ax in [ax0, ax2, ax3]:
        figures.set_fontsize(ax, 13.)
    
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    #plt.show()

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    data_path = os.getenv('CUGN')
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
        #outliers(ds)
        outliers(ds, pcut=2.)

    # Outlier montage
    if flg & (2**6):
        '''
        # Low DO in 2017
        outfile = 'Figures/omontage_2017_p5_A.png'
        out_dict = dict(perc=5., year=2017,
                        lons=[-130., -121.],
                        sigma=[20., 25.5])
        outlier_montage(ds, out_dict, outfile)
        '''

        '''
        # High DO in 2017, in-shorte
        outfile = 'Figures/omontage_2017_p95_A.png'
        out_dict = dict(perc=95., year=2017,
                        lons=[-118., -100.],
                        sigma=[20., 25.5],
                        months=[6,8])
        outlier_montage(ds, out_dict, outfile)
        '''

        # High DO in 2017, in-shorte
        outfile = 'Figures/omontage_2017_p95_B.png'
        out_dict = dict(perc=95., year=2017,
                        lons=[-122., -120.],
                        sigma=[25.3, 26.3],
                        months=[12,12])
        outlier_montage(ds, out_dict, outfile)


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
        #flg += 2 ** 6  # 64 -- Outlier montage
    else:
        flg = sys.argv[1]

    main(flg)