# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:13:59 2021

@author: marie

Utility Functions.

"""
import os
import os.path
import pandas as pd

#%% Load observations

def load_data(dataset, simulated, 
              data_dir = "OneDrive\Dokumente\Sc_Master\Internship\Intern_MPI_Jena\data"):
    
    path_in = os.path.join(data_dir, f"{dataset}_clim")
    
    if (simulated==True):
        path_out = os.path.join(data_dir, f"{dataset}_preles_gpp")
    else:
        path_out = os.path.join(data_dir, f"{dataset}_gpp")
        
    X = pd.read_csv(path_in, sep=";")
    Y = pd.read_csv(path_out, sep=";")
    
    # Remove nows with na values
    #rows_with_nan = pd.isnull(X).any(1).to_numpy().nonzero()[0]
    #X = X.drop(rows_with_nan)
    #Y = Y.drop(rows_with_nan)
    
    return X, Y

X, Y = load_data("soro", False)
#%% Run Preles (R-Script)
    