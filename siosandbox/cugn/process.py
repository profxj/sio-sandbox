""" Routines to process the Spray CUGN data """

# imports
import os
import xarray
from glob import glob

import numpy as np

import pandas

from gsw import conversions, density
import gsw

from siosandbox.cugn import utils as cugn_utils
from siosandbox.cugn import grid_utils
from siosandbox.cugn import io as cugn_io
from siosandbox.cugn import defs as cugn_defs


from IPython import embed

def add_gsw():
    """ Add physical quantities to the Spray CUGN data
    using the TEOS-10 GSW package
    """
    data_path = os.getenv('CUGN') 

    # Spray files
    spray_files = glob(os.path.join(data_path, 'CUGN_line_*.nc'))

    for spray_file in spray_files:

        print(f"Working on: {spray_file}")
        # Load the data
        ds = xarray.load_dataset(spray_file)
        lat = np.nanmedian(ds.lat.data)
        lon = np.nanmedian(ds.lon.data)

        # Prep for new variables
        CT = np.ones_like(ds.temperature.data) * np.nan
        SA = np.ones_like(ds.temperature.data) * np.nan
        OC = np.ones_like(ds.temperature.data) * np.nan
        SO = np.ones_like(ds.temperature.data) * np.nan

        # Loop on depths
        for zz, z in enumerate(ds.depth.data):
            # Pressure
            p = conversions.p_from_z(-z, lat)

            # SA
            iSA = conversions.SA_from_SP(ds.salinity.data[zz,:], 
                                        p, lon, lat)
            SA[zz,:] = iSA

            # CT
            iCT = conversions.CT_from_t(iSA, ds.temperature.data[zz,:], 
                                        p)
            CT[zz,:] = iCT

            # Oxygen
            OC[zz,:] = gsw.O2sol(iSA, iCT, p, lon, lat)
            SO[zz,:] = ds.doxy[zz,:] / OC[zz,:] 

        # sigma0 
        sigma0 = density.sigma0(SA, CT)

        # Add to ds
        ds['CT'] = (('depth', 'profile'), CT)
        ds.CT.attrs = dict(units='Celsius', long_name='Conservative Temperature')
        ds['sigma0'] = (('depth', 'profile'), sigma0)
        ds.sigma0.attrs = dict(units='kg/m^3', long_name='potential density anomaly')
        ds['SA'] = (('depth', 'profile'), SA)
        ds.SA.attrs = dict(units='g/kg', long_name='Absolute Salinity')
        ds['SO'] = (('depth', 'profile'), SO)
        ds.SO.attrs = dict(long_name='Oxygen Saturation')

        # Buoyancy
        dsigmadz, _ = np.gradient(ds.sigma0.data, 
                                  float(ds.depth[1]-ds.depth[0]))
        dsigmadz[dsigmadz < 0.] = 0.
        buoyfreq = np.sqrt(9.8/1025*dsigmadz)/(2*np.pi)*3600
        ds['N'] = (('depth', 'profile'), buoyfreq)
        ds.N.attrs = dict(long_name='Buoyancy Frequency', units='cycles/hour')

        # Wipe out the 2020 trip to San Diego
        dskeys =  ['temperature', 'salinity', 'chlorophyll_a', 'u', 'v', 
                   'acoustic_backscatter', 'doxy', 'CT', 'sigma0', 'SA', 'SO', 'N']
        if 'line_80' in spray_file:
            dist, _ = cugn_utils.calc_dist_offset(
                '80', ds.lon.values, ds.lat.values)
            bad = dist < -50.
            # Zero em!!
            for key in dskeys:
                ds[key].data[:,bad] = np.nan

        # Mexican trip on Line 90
        if 'line_90' in spray_file:
            bad = ds.mission == 63
            for key in dskeys:
                ds[key].data[:,bad] = np.nan

        # Write
        new_spray_file = spray_file.replace('CUGN_', 'CUGN_potential_')
        ds.to_netcdf(new_spray_file)
        print(f"Wrote: {new_spray_file}")

def build_ds_grid(line_file:str, gridtbl_outfile:str, edges_outfile:str,
               min_counts:int=50):
    """ Grid up density and salinity for a line
    to generate a table of grid indices and values

    Args:
        line_file (str): Identify the line ['90', '67']
        gridtbl_outfile (str): name of the output table
        edges_outfile (str): name of the output grid edges
        min_counts (int, optional): Minimum counts on the
            grid to be included in the analysis. Defaults to 50.
    """
    # Dataset
    ds = xarray.load_dataset(line_file)

    # Generate the grid
    mean_oxyT, SA_edges, sigma_edges, countsT, \
        grid_indices, gd_oxy, da_gd = grid_utils.gen_grid(ds, stat='mean')

    # Table me
    depth, profile = np.where(da_gd)
    grid_tbl = pandas.DataFrame()

    grid_tbl['depth'] = depth
    grid_tbl['profile'] = profile
    grid_tbl['row'] = grid_indices[0,:] - 1
    grid_tbl['col'] = grid_indices[1,:] - 1
    grid_tbl['doxy'] = gd_oxy

    # Cut on counts
    gd_rows, gd_cols = np.where(countsT > min_counts)

    # Keep only the good ones
    keep = np.zeros(len(grid_tbl), dtype=bool)
    for gd_row, gd_col in zip(gd_rows, gd_cols):
        keep |= (grid_tbl.row == gd_row) & (grid_tbl.col == gd_col)

    grid_tbl = grid_tbl[keep].copy()
    grid_tbl.reset_index(inplace=True, drop=True)

    # Generate DO percentile
    grid_utils.find_perc(grid_tbl)

    # Test
    #in_cell = (grid_tbl.row == 41) & (grid_tbl.col == 45)
    #vals = grid_tbl.doxy.values[in_cell]
    #pct = np.nanpercentile(vals, 5)

    # Save
    grid_tbl.to_parquet(gridtbl_outfile)
    np.savez(edges_outfile, SA_edges=SA_edges, 
             sigma_edges=sigma_edges,
             counts=countsT)
    print(f"Wrote: \n {gridtbl_outfile} \n {edges_outfile}")



if __name__ == '__main__':
    # Add potential density and salinity to the CUGN files
    add_gsw()

    # Grids
    for line in cugn_defs.lines:
        line_files = cugn_io.line_files(line)

        build_ds_grid(line_files['datafile'],
            line_files['gridtbl_file'], 
            line_files['edges_file'])