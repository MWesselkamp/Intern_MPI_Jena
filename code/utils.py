# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:13:59 2021

@author: marie

Utility Functions.

"""
import torch
import pandas as pd
import numpy as np

import random

#%%
def minmax_scaler(data, scaling = None):
    """
    This function scales all features in an array between mean and standard deviation. 
    
    Args:
        data(np.array): two dimensional array containing model features.
        
    Returns:
        data_norm(np.array): two dimensional array of scaled model features.
    """
    #scaler = MinMaxScaler(feature_range = (-1,1))
    if (scaling is None):
        
        if (isinstance(data, pd.DataFrame)):
            data_norm = (data - data.mean())/ data.std()
        elif (torch.is_tensor(data)):
            data_norm = (data - torch.mean(data)) / torch.std(data)
        else:
            data_norm = (data - np.mean(data, axis=0))/np.std(data, axis=0)
    else: 
        data_norm = (data - scaling[0])/scaling[1]
        
    return data_norm

#%%
def encode_doy(doy):
    """Encode the day of the year on a circle.
    
    Thanks to: Philipp Jund.
    
    """
    doy_norm = doy / 365 * 2 * np.pi
    
    return np.sin(doy_norm), np.cos(doy_norm)

#%% 
def create_batches(X, Y, batchsize, history):
    
    """
    Creates Mini-batches from training data set.
    Used in: dev_mlp.train_model_CV
    """
    
    subset = [j for j in random.sample(range(X.shape[0]), batchsize) if j > history]
    subset_h = [item for sublist in [list(range(j-history,j)) for j in subset] for item in sublist]
    x = np.concatenate((X[subset], X[subset_h]), axis=0)
    y = np.concatenate((Y[subset], Y[subset_h]), axis=0)
    
    return x, y

#%%
def rmse(targets, predictions):
    
    """
    Computes the Root Mean Squared Error.
    
    Args:
        targets (torch.tensor)
        predictions (torch.tensor)
    """
    if torch.is_tensor(targets):
        rmse = np.sqrt(np.mean(np.square(targets-predictions).numpy()))
    else:
        rmse = np.sqrt(np.mean(np.square(targets-predictions)))
    
    return rmse