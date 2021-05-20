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
def plot_data(Obs, Sim, first = True,
              label=True):
    
    if first:
        colors = ["black", "green"]
    else:
        colors = ["gray", "lightgreen"]
        
    plt.plot(Obs, label="Observed", color="black")
    plt.plot(Sim, label="Simulated", color="lightgreen")
    if label:
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
    plt.ylabel("GPP [g C m$^{-2}$ day$^{-1}$]")
    plt.xlabel("Day")
    plt.legend(loc = "upper left")
    plt.text(250, 16, f"Diff. in MAE: {np.round(abs(np.mean(mae1)-np.mean(mae2)), 4)}")
    #plt.text(250, 15, f"MAE_m2: {np.round(np.mean(mae_m2), 4)}")
    print("Extrapolation error:", np.mean(mae1))
    print("Training error:", np.mean(mae2))
    print("Absolute difference:", abs(np.mean(mae1)-np.mean(mae2)))
    
    #%%
def plot_running_losses(train_loss, val_loss, 
                        plot_train_loss =True,
                        legend=True,
                        colors=["blue", "lightblue"],
                        colors_test_loss = ["green","lightgreen"]):

    #if model=="mlp":
    #    colors = ["blue","lightblue"]
    #elif model=="cnn":
    #    colors = ["darkgreen", "palegreen"]
    #elif model=="lstm":
    #    colors = ["blueviolet", "thistle"]
    #else:

    
    fig, ax = plt.subplots(figsize=(7,7))

    if train_loss.shape[0] > 1:
        ci_train = np.quantile(train_loss, (0.05,0.95), axis=0)
        ci_val = np.quantile(val_loss, (0.05,0.95), axis=0)
        train_loss = np.mean(train_loss, axis=0)
        val_loss = np.mean(val_loss, axis=0)
        
        if plot_train_loss:
            ax.fill_between(np.arange(len(train_loss)), ci_train[0],ci_train[1], color=colors[1], alpha=0.3)
        ax.fill_between(np.arange(len(train_loss)), ci_val[0],ci_val[1], color=colors_test_loss[1], alpha=0.3)
    
    else: 
        train_loss = train_loss.reshape(-1,1)
        val_loss = val_loss.reshape(-1,1)
    
    if plot_train_loss:
        ax.plot(train_loss, color=colors[0], label="Training loss", linewidth=1.2)
        ax.plot(val_loss, color="green", label = "Test loss", linewidth=1.2)
    else:
        ax.plot(val_loss, color=colors_test_loss[0], label = "Test loss\nfull re-training", linewidth=1.2)
    #ax[1].plot(train_loss, color="green", linewidth=0.8)
    #ax[1].plot(val_loss, color="blue", linewidth=0.8)
    ax.set_ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]", size=20)
    ax.set_xlabel("Epochs", size=20)
    #plt.ylim(bottom = 0.0)
    for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 
    for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 
    plt.rcParams.update({'font.size': 20})
    if legend:
        fig.legend(loc="upper right")