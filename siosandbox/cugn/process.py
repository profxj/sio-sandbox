""" Routines to process the Spray CUGN data """

# imports
import os
import xarray
from glob import glob

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

import pandas

from gsw import conversions, density

from IPython import embed

def add_potential():
    data_path = os.getenv('CUGN') 

    # Spray files
    spray_files = glob(os.path.join(data_path, 'CUGN_*.nc'))

    for spray_file in spray_files:

        print(f"Working on: {spray_file}")
        # Load the data
        ds = xarray.load_dataset(spray_file)
        lat = np.nanmedian(ds.lat.data)
        lon = np.nanmedian(ds.lon.data)

        # Prep for new variables
        CT = np.ones_like(ds.temperature.data) * np.nan
        SA = np.ones_like(ds.temperature.data) * np.nan

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

        # sigma0 
        sigma0 = density.sigma0(SA, CT)

        # Add to ds
        ds['CT'] = (('depth', 'profile'), CT)
        ds.CT.attrs = dict(units='Celsius', long_name='Conservative Temperature')
        ds['sigma0'] = (('depth', 'profile'), sigma0)
        ds.sigma0.attrs = dict(units='kg/m^3', long_name='potential density anomaly')
        ds['SA'] = (('depth', 'profile'), SA)
        ds.SA.attrs = dict(units='g/kg', long_name='Absolute Salinity')

        # Write
        new_spray_file = spray_file.replace('CUGN_', 'CUGN_potential_')
        ds.to_netcdf(new_spray_file)
        print(f"Wrote: {new_spray_file}")


if __name__ == '__main__':
    add_potential()