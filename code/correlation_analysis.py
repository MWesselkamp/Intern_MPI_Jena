#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:28:51 2021

@author: Marieke_Wesselkamp
"""

#%% Set working directory
import os
print("Current Working Directory " , os.getcwd())
#%%
os.chdir("/Users/Marieke_Wesselkamp/Documents/Projects/Intern_MPI_Jena") 
print("Current Working Directory " , os.getcwd())

#%% Set system path
import sys
print(sys.path)
sys.path.append('/Users/Marieke_Wesselkamp/Documents/Projects/Intern_MPI_Jena/code')

#%% Import packages
import preprocessing
import visualizations
import utils
import training
import prediction
import random_search
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr

#%%

#%% Yearly correlations with GPP
def yearly_correlations():
    
    GPP = X.groupby('year')['GPP'].sum()
    Clim = X.groupby('year')['TAir', 'PAR', 'VPD', 'Precip', 'fAPAR'].mean()
    
    var = list(Clim.columns)
    corrs = []
    for i in range(len(Clim.columns)):
        corrs.append(pearsonr(GPP, Clim.iloc[:,i])[0])

    corrs = dict(zip(var, corrs))
    
#%%
def myfunc(data, agg = "sum"):
    
    s = []
    for i in range(365):
        if agg == "sum":
            s.append(data[i:i+30].sum())
        else:
            s.append(data[i:i+30].mean())
    return s

#%%
def correlation_ts(var, data):
    
    X, Y, Y_Preles = preprocessing.preprocessing()

    if data == "observed":
        X['GPP'] = Y
    else:
        X['GPP'] = Y_Preles

    years = list(X['year'].unique())
    corrs = []
    for y in years: 
        
        ys = [x for x in years if x != y]
        
        x , y, y_preles = preprocessing.split_by_year(X, Y, Y_Preles,
                                                          years = ys,
                                                          drop_year=False)
        GPP = x.groupby('year')['GPP']
        TAir = x.groupby('year')[var]
        
        gpp_series = GPP.apply(myfunc, agg=("sum"))
        tair_series = TAir.apply(myfunc, agg=("mean"))
        
        gpp_m = []
        for i in range(len(gpp_series)):
            gpp_m.append(gpp_series.to_numpy()[i])
            
        gpp_m = np.array(gpp_m)
            
        tair_m = []
        for i in range(len(tair_series)):
            tair_m.append(tair_series.to_numpy()[i])
        tair_m = np.array(tair_m)
        
        c = []
        for i in range(365):
            c.append(pearsonr(tair_m[:,i], gpp_m[:,i])[0])
        corrs.append(c)
        
    return np.array(corrs)

#%%

font = {'size'   : 18,
        'weight' : 'normal'}

plt.rc('font', **font)

#%%
def plot_correlations_by_data(var):
    fig, ax = plt.subplots()    
    ax.fill_between(np.arange(365), np.quantile(corrs_obs, q=0.05, axis=0),np.quantile(corrs_obs, q=0.95, axis=0), color="salmon", alpha=0.5)
    ax.plot(np.quantile(corrs_obs, q=0.5, axis=0), color="red", label = "GPP$_{obs}$ vs. "+var, linewidth=1.0)
    ax.fill_between(np.arange(365), np.quantile(corrs_sim, q=0.05, axis=0),np.quantile(corrs_sim, q=0.95, axis=0), color="lightblue", alpha=0.5)
    ax.plot(np.quantile(corrs_sim, q=0.5, axis=0), color="blue", label = "GPP$_{sim}$ vs. "+var, linewidth=1.0)
    ax.hlines(0, 0, 365, color="black", linestyle='dashed')
    ax.set_ylabel("Correlation coefficent (Pearson's r)")
    ax.set_xlabel("Day of year")
    ax.legend(loc="lower right")
    ax.set_ylim(-1,1)
    
#%%
corrs_obs = correlation_ts('TAir', 'observed')
corrs_sim = correlation_ts('TAir', 'simulated')  
#%% 
plot_correlations_by_data("TAir")
#%%
corrs_obs = correlation_ts('PAR', 'observed')
corrs_sim = correlation_ts('PAR', 'simulated')   
plot_correlations_by_data("PAR")
#%%
corrs_obs = correlation_ts('Precip', 'observed')
corrs_sim = correlation_ts('Precip', 'simulated')   
plot_correlations_by_data("Precip")

#%%
corrs_obs = correlation_ts('VPD', 'observed')
corrs_sim = correlation_ts('VPD', 'simulated')   
plot_correlations_by_data("VPD")

#%%
corrs_PAR = correlation_ts('PAR')
corrs_VPD = correlation_ts('VPD')
corrs_Precip = correlation_ts('Precip')
corrs_fAPAR = correlation_ts('fAPAR')
#%%
fig, ax = plt.subplots()
ax.fill_between(np.arange(365), np.quantile(corrs_Tair, q=0.05, axis=0),np.quantile(corrs_Tair, q=0.95, axis=0), color="salmon", alpha=0.5)
ax.plot(np.quantile(corrs_Tair, q=0.5, axis=0), color="red", label = "GPP vs. TAir", linewidth=1.0)
ax.fill_between(np.arange(365), np.quantile(corrs_PAR, q=0.05, axis=0),np.quantile(corrs_PAR, q=0.95, axis=0), color="lightblue", alpha=0.5)
ax.plot(np.quantile(corrs_PAR, q=0.5, axis=0), color="blue", label = "GPP vs. PAR", linewidth=1.0)
ax.fill_between(np.arange(365), np.quantile(corrs_VPD, q=0.05, axis=0),np.quantile(corrs_VPD, q=0.95, axis=0), color="lightgreen", alpha=0.5)
ax.plot(np.mean(corrs_VPD, axis=0), color="green", label = "GPP vs. VPD", linewidth=1.0)
ax.fill_between(np.arange(365), np.quantile(corrs_Precip, q=0.05, axis=0),np.quantile(corrs_Precip, q=0.95, axis=0), color="moccasin", alpha=0.5)
ax.plot(np.quantile(corrs_Precip, q=0.5, axis=0), color="orange", label = "GPP vs. Precip", linewidth=1.0)
ax.hlines(0, 0, 365, color="black", linestyle='dashed')
ax.set_ylabel("Correlation coefficent (Pearson's r)")
ax.set_xlabel("Day of year")
ax.legend(loc="lower right")
ax.set_ylim(-1,1)
