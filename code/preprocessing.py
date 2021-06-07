#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:28:57 2021

@author: Marieke_Wesselkamp
"""

import pandas as pd
import utils

#%% Load observations
def load_data():
    
    X = pd.read_csv(r"data/soro_clim", sep=";")
    Y = pd.read_csv(r"data/soro_gpp", sep=";")
    Y_Preles = pd.read_csv(r"data/soro_preles_gpp", sep=";")
    
    years = X.year.unique()
    print(f"Loading {len(years)} years of soro stand data.")
    print(f"Years: {years}")
    
    # Remove nows with na values
    rows_with_nan = pd.isnull(X).any(1).to_numpy().nonzero()[0]
    X = X.drop(rows_with_nan)
    Y = Y.drop(rows_with_nan)
    
    Y_Preles = Y_Preles.drop(rows_with_nan)
    Y_Preles = Y_Preles.drop(columns=["ET", "SW"])
    
    return X, Y, Y_Preles

#%% Process 
def modify_data(X):
    
    X["date"] = X["date"].str[:4].astype(int) # get years as integers
    X["DOY_sin"], X["DOY_cos"] = utils.encode_doy(X["DOY"]) # encode day of year as sinus and cosinus
        
    return X
        
#%%
def subset_data(X, colnames = ["PAR", "TAir", "VPD", "Precip", "fAPAR", "DOY_sin", "DOY_cos", "year"]):
    
    try:
        X = X[colnames]
    except:
        print("Columns are missing!")
    
    return X

#%% 
def standardize_data(X):
    
    X_scaled = utils.minmax_scaler(X.drop(["year"], axis=1))
    X_scaled["year"] = X["year"]
    
    return X_scaled
        
#%%
def preprocessing(standardize=True):
    
    X, Y, Y_Preles = load_data()
    X = modify_data(X)
    X = subset_data(X)
    if standardize:
        X = standardize_data(X)
    
    return X, Y, Y_Preles

#%%
def split_by_year(X, Y, Y_Preles, 
               years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
               drop_year = True):
    
    row_ind = X["year"].isin(years)
    #print(f"Returns valid years from {years} in \n", X["year"].unique())
    if drop_year:
        X = X.drop(["year"], axis=1)
    X, Y, Y_Preles = X[row_ind], Y[row_ind], Y_Preles[row_ind]
        
    return X, Y, Y_Preles

#%%
def split_by_sequence(X, Y, Y_Preles,start, stop, other = None):
    
    try:
        X = X.drop(["year"], axis=1)
    except:
        pass
    X, Y, Y_Preles = X[start:stop], Y[start:stop], Y_Preles[start:stop]
        
    if other is not None:
        other = other[:,start:stop]
        return X, Y, Y_Preles, other
    else:
        return X, Y, Y_Preles