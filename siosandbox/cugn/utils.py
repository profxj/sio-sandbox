""" Basic utilities for the CUGN project """

import numpy as np

def line_endpoints(line:str):
    
    if line == '66':
        lonendpts = [-121.8371, -124.2000]
        latendpts = [36.8907, 35.7900]
    elif line == '80':
        lonendpts = [-120.4773,-123.9100]
        latendpts = [34.4703, 32.8200]
    elif line == '90':
        lonendpts = [-117.7475, -124.0000]
        latendpts = [33.5009, 30.4200]
    elif line == 'al':
        lonendpts = [-119.9593, -121.1500]
        latendpts = [32.4179, 34.1500]
    else:
        raise ValueError('line not recognized')

    return lonendpts, latendpts

def calc_dist_offset(line:str, lons:np.ndarray, lats:np.ndarray):
    """ Calculate the distnace from shore and offset from a line
      for a given line 

    Args:
        line (str): line name
        lons (np.ndarray): longitudes
        lats (np.ndarray): latitudes

    Returns:
        tuple: dist, offset
    """

    # Endpoints
    lonendpts, latendpts = line_endpoints(line)
    lon0, lon1 = lonendpts
    lat0, lat1 = latendpts

    # Constants
    nm2km = 1.852
    deg2min = 60.
    deg2rad = np.pi/180
    deg2km=deg2min*nm2km

    # Calculate angle of new coordinate system relative to east
    dyy = (lat1-lat0)
    dxx = np.cos(1/2*(lat1+lat0)*deg2rad)*(lon1-lon0)
    theta = np.arctan2(dyy,dxx)

    # Calculate x, y of lon, lat relative to start of line
    dy = (lats-lat0)*deg2km;
    dx = np.cos(1/2*(lat1+lat0)*deg2rad)*(lons-lon0)*deg2km

    # Calculate dist, offset in new coordinate system by rotating
    z=dx+1j*dy
    zhat=z*np.exp(-1j*theta)

    # Finish
    dist=np.real(zhat)
    offset=np.imag(zhat)

    # Return
    return dist, offset
