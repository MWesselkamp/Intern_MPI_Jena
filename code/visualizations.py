#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:36:11 2021

@author: Marieke_Wesselkamp
"""
import matplotlib.pyplot as plt
#%%
def plot_data(Obs, Sim):
    
    plt.plot(Obs, label="Observed")
    plt.plot(Sim, label="Simulated")
    plt.legend()
    plt.xlabel("Day")
    plt.ylabel("GPP [g C m$^{-2}$ day$^{-1}$]")