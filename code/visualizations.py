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
def plot_predictions(Y, preds1, preds2, mae1, mae2):
    
    plt.plot(Y.to_numpy(), color="black", label="P2")
    #for i in range(5):
    plt.plot(np.mean(preds1, axis=0), color = "lightgreen", alpha = 1, linewidth=0.8, label="$\widehat{P2}_{m1}$")
    #plt.plot(Y_P2.to_numpy(), color="black")
    #for i in range(5):
    plt.plot(np.mean(preds2, axis=0), color = "lightblue", alpha = 1, linewidth=0.8, label="$\widehat{P2}_{m2}$")
    plt.ylabel("Gross primary produdction [g C m$^{-2}$ day$^{-1}$]")
    plt.xlabel("Day")
    plt.legend(loc = "upper left")
    plt.text(250, 16, f"Diff. in MAE: {np.round(abs(np.mean(mae1)-np.mean(mae2)), 4)}")
    #plt.text(250, 15, f"MAE_m2: {np.round(np.mean(mae_m2), 4)}")
    print("Extrapolation error:", np.mean(mae1))
    print("Training error:", np.mean(mae2))
    print("Absolute difference:", abs(np.mean(mae1)-np.mean(mae2)))