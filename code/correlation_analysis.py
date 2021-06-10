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
# /Users/marie/OneDrive/Dokumente/Sc_Master/Internship/Intern_MPI_Jena
# /Users/Marieke_Wesselkamp/Documents/Projects/Intern_MPI_Jena
os.chdir("/Users/marie/OneDrive/Dokumente/Sc_Master/Internship/Intern_MPI_Jena") 
print("Current Working Directory " , os.getcwd())

#%% Set system path
import sys
print(sys.path)
sys.path.append('/Users/marie/OneDrive/Dokumente/Sc_Master/Internship/Intern_MPI_Jena/code')

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
def mlp_predictions(randomsearch = False):
    
    if randomsearch:
        layersizes = random_search.architecture_search() # 7,32,32,16,1
        hparams = random_search.hparams_search(layersizes) # 0.005, 16
    else:
        layersizes = [7,32,32,16,1]
        hparams = [0.005, 16]
        
    hparams_setting = {"epochs":1000,
                       "batchsize":hparams[1],
                       "learningrate":hparams[0],
                       "history":1}
    model_design = {"layer_sizes":layersizes}
    
    X, Y, Y_Preles = preprocessing.preprocessing()
    X = X.drop(["year"], axis=1)
    
    running_losses = training.train(hparams_setting, model_design, X.to_numpy(), Y.to_numpy(), "D1")
    
    preds, mae, nse = prediction.predict(hparams_setting, model_design, X.to_numpy(), Y.to_numpy(),"D1")

    return preds
#%%
def correlation_ts(var, data):
    
    X, Y, Y_Preles = preprocessing.preprocessing()
    preds  = mlp_predictions()
    Y_NN = np.mean(preds, axis=0)
    
    if data == "observed":
        X['GPP'] = Y
    elif data == "preles":
        X['GPP'] = Y_Preles
    elif data == "nn":
        X['GPP'] = Y_NN

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
    
    fig, ax = plt.subplots(figsize=(9.5,7))    
    ax.fill_between(np.arange(365), np.quantile(corrs_obs, q=0.05, axis=0),np.quantile(corrs_obs, q=0.95, axis=0), color="salmon", alpha=0.5)
    ax.plot(np.quantile(corrs_obs, q=0.5, axis=0), color="red", label = "GPP$_{obs}$ vs. "+var, linewidth=1.0)
    ax.fill_between(np.arange(365), np.quantile(corrs_preles, q=0.05, axis=0),np.quantile(corrs_preles, q=0.95, axis=0), color="lightblue", alpha=0.5)
    ax.plot(np.quantile(corrs_preles, q=0.5, axis=0), color="blue", label = "$\widehat{GPP}_{preles}$ vs. "+var, linewidth=1.0)
    ax.fill_between(np.arange(365), np.quantile(corrs_nn, q=0.05, axis=0),np.quantile(corrs_nn, q=0.95, axis=0), color="lightgreen", alpha=0.5)
    ax.plot(np.quantile(corrs_nn, q=0.5, axis=0), color="green", label = "$\widehat{GPP}_{nn}$ vs. "+var, linewidth=1.0)

    ax.hlines(0, 0, 365, color="black", linestyle='dashed')
    ax.set_ylabel("Correlation coefficent (Pearson's r)")
    ax.set_xlabel("Day of year")
    ax.legend(loc="lower right")
    ax.set_ylim(-1,1)
    
#%%
corrs_obs = correlation_ts('TAir', 'observed')
corrs_preles = correlation_ts('TAir', 'preles')  
corrs_nn = correlation_ts('TAir', 'nn')  
plot_correlations_by_data("TAir")
#%%
corrs_obs = correlation_ts('PAR', 'observed')
corrs_preles = correlation_ts('PAR', 'preles')  
corrs_nn = correlation_ts('PAR', 'nn')  
plot_correlations_by_data("PAR")
#%%
corrs_obs = correlation_ts('Precip', 'observed')
corrs_preles = correlation_ts('Precip', 'preles')  
corrs_nn = correlation_ts('Precip', 'nn')  
#%% 
plot_correlations_by_data("Precip")

#%%
corrs_obs = correlation_ts('VPD', 'observed')
corrs_preles = correlation_ts('VPD', 'preles')  
corrs_nn = correlation_ts('VPD', 'nn')  
plot_correlations_by_data("VPD")
