#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:36:11 2021

@author: Marieke_Wesselkamp
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

#%%
def plot_data(Obs, Sim):
    
    plt.plot(Obs, label="Observed")
    plt.plot(Sim, label="Simulated")
    plt.legend()
    plt.xlabel("Day")
    plt.ylabel("GPP [g C m$^{-2}$ day$^{-1}$]")
    
#%%
def plot_prediction(observed, predicted, error=""):
    
    """
    Plot Model Prediction Error (root mean squared error).
    
    """
    
    fig, ax = plt.subplots(figsize=(7,7))
    
    ax.plot(observed, color="lightgrey", label="Ground truth", marker = "o", linewidth=0.8, alpha=0.9, markerfacecolor='lightgrey', markersize=4)

    ci_preds = np.quantile(np.array(predicted), (0.05,0.95), axis=0)
    m_preds = np.mean(np.array(predicted), axis=0)
    
    ax.fill_between(np.arange(len(ci_preds[0])), ci_preds[0],ci_preds[1], color="lightgreen", alpha=0.9)
    ax.plot(m_preds, color="green", label="Predictions", marker = "", alpha=0.5)

        
    errors = np.subtract(np.transpose(np.array(predicted)), observed)
    ci_preds = np.transpose(np.quantile(errors, (0.05,0.95), axis=1))
    m_errors = np.mean(errors, axis=1)
    ax.fill_between(np.arange(errors.shape[0]), ci_preds[:,0],ci_preds[:,1], color="lightsalmon", alpha=0.9)
    ax.plot(m_errors, color="red", label="Error", marker = "", alpha=0.5)
        
    ax.set_xlabel("Day of Year", size=20, family='Palatino Linotype')
    ax.set_ylabel("Gross primary produdction [g C m$^{-2}$ day$^{-1}$]", size=20, family='Palatino Linotype')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
        tick.label.set_fontfamily('Palatino Linotype') 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
        tick.label.set_fontfamily('Palatino Linotype') 
    
    ax.legend(loc="upper left", prop={'size':20, 'family':'Palatino Linotype'})
    ax.set_ylim((-4,14.2))
    
    #plt.text(250, 13.0, f"MAE = {mae: .4f}", family='Palatino Linotype', size=20)