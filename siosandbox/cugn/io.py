""" I/O for CUGN data and analysis """
import os
import numpy as np
import xarray
import pandas

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