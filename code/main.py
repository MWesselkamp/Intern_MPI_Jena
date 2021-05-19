# -*- coding: utf-8 -*-
"""
Created on Wed May  5 08:58:11 2021

@author: marie

Inter-annual variability of NN-predicted GPP.



1. IMPORT OBSERVATIONS (PROFOUND DATA): X1

2. CALIBRATE AND RUN PRELES WITH OBSERVATIONS: X2

3. FIT AND PREDICT WITH NETWORK TO OBSERVATIONS: X3

4. BI-VARIATE CORRELATION ANALYSIS FOR X1, X2, X3.

"""
#%% Set working directory
import os
print("Current Working Directory " , os.getcwd())
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
#%%
X, Y, Y_Preles = preprocessing.preprocessing()
#%% Split data by years
X_P1 , Y_P1, Y_Preles_P1 = preprocessing.split_by_year(X, Y, Y_Preles,
                                                    years = [2001])
X_P2 , Y_P2, Y_Preles_P2 = preprocessing.split_by_year(X, Y, Y_Preles,
                                                    years = [2002])

#%% Plot Data
visualizations.plot_data(Y_P1, Y_Preles_P1)
visualizations.plot_data(Y_P2, Y_Preles_P2)

#%% Specify model structure and hyperparameter settings.
randomsearch = False
if randomsearch:
    layersizes = random_search.architecture_search() # 7,32,32,16,1
    hparams = random_search.hparams_search(layersizes) # 0.005, 16
else:
    layersizes = [7,32,32,16,1]
    hparams = [0.005, 16]
#%%
hparams_setting = {"epochs":1000,
           "batchsize":hparams[1],
           "learningrate":hparams[0],
           "history":1}
model_design = {"layer_sizes":layersizes}

#%% Train m1 and m2 on D1P1 and D1P2 and on D2P1 and D2P2
running_losses_d1p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "D1P1")
running_losses_d1p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(), "D1P2")

running_losses_d2p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_Preles_P1.to_numpy(), "D2P1")
running_losses_d2p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P2")
#%%
plt.plot(np.transpose(running_losses_d1p1["mae_val"]))
#%% Predict with fitted models to P2.
preds_d1m1, mae_d1m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P1")
preds_d1m2, mae_d1m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P2")

visualizations.plot_predictions(Y_P2, preds_d1m1, preds_d1m2, mae_d1m1, mae_d1m2)
#%% 
plt.plot(np.transpose(running_losses_d2p1["mae_val"]))
#%% Predict with fitted models to P2.
preds_d2m1, mae_d2m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P1")
preds_d2m2, mae_d2m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P2")

visualizations.plot_predictions(Y_Preles_P2, preds_d2m1, preds_d2m2, mae_d2m1, mae_d2m2)

#%%
def fit_with_moving_window(windowsize, seq_len):
    
    hparams_setting = {"epochs":300,
                       "batchsize":hparams[1],
                       "learningrate":hparams[0],
                       "history":1}
    model_design = {"layer_sizes":layersizes}
    mae_diff_d1 = []
    mae_diff_d2 = []
    
    df = pd.DataFrame(columns = ["mae_d1m1", "mae_d1m2", "mae_d2m1", "mae_d2m2"])

    for i in range(seq_len):
    
        X_P1 , Y_P1, Y_Preles_P1 = preprocessing.split_by_sequence(X, Y, Y_Preles,
                                                    start = i, stop=i+windowsize)
        X_P2 , Y_P2, Y_Preles_P2 = preprocessing.split_by_sequence(X, Y, Y_Preles,
                                                    start = i+windowsize, stop=i+windowsize+windowsize)

        running_losses_d1p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "D1P1")
        running_losses_d1p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(), "D1P2")
        
        running_losses_d2p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_Preles_P1.to_numpy(), "D2P1")
        running_losses_d2p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P2")
        
        preds_d1m1, mae_d1m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P1")
        preds_d1m2, mae_d1m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P2")
        
        preds_d2m1, mae_d2m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P1")
        preds_d2m2, mae_d2m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P2")
        
        #visualizations.plot_predictions(Y_P2, preds_d1m1, preds_d1m2, mae_d1m1, mae_d1m2)
        #visualizations.plot_predictions(Y_Preles_P2, preds_d2m1, preds_d2m2, mae_d2m1, mae_d2m2)
        df = df.append({"mae_d1m1":np.mean(mae_d1m1),
                        "mae_d1m2":np.mean(mae_d1m2),
                        "mae_d2m1":np.mean(mae_d2m1),
                        "mae_d2m2":np.mean(mae_d2m2)}, True)
        
        mae_diff_d1.append(abs(np.mean(mae_d1m2)-np.mean(mae_d1m1)))
        mae_diff_d2.append(abs(np.mean(mae_d2m2)-np.mean(mae_d2m1)))
    
    df.to_excel(r"results/fit_with_moving_window.xlsx")
    df.to_csv(r"results/fit_with_moving_window.csv")
    
    return df



#%%
def fit_with_increasing_windowsize(windowsize, max_len):

    hparams_setting = {"epochs":300,
                       "batchsize":hparams[1],
                       "learningrate":hparams[0],
                       "history":1}
    model_design = {"layer_sizes":layersizes}
    
    if max_len is None:    
        max_len = int(np.floor(len(Y)/2))
        
    df = pd.DataFrame(columns = ["mae_d1m1", "mae_d1m2", "mae_d2m1", "mae_d2m2"])
    
    for i in range(max_len):
    
        X_P1 , Y_P1, Y_Preles_P1 = preprocessing.split_by_sequence(X, Y, Y_Preles,
                                                                   start = 0, stop=windowsize)
        X_P2 , Y_P2, Y_Preles_P2 = preprocessing.split_by_sequence(X, Y, Y_Preles,
                                                                   start = windowsize+1, stop=1+windowsize+windowsize)
        windowsize += 1
    
        running_losses_d1p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "D1P1")
        running_losses_d1p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(), "D1P2")
        
        running_losses_d2p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_Preles_P1.to_numpy(), "D2P1")
        running_losses_d2p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P2")
        
        preds_d1m1, mae_d1m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P1")
        preds_d1m2, mae_d1m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P2")
        
        preds_d2m1, mae_d2m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P1")
        preds_d2m2, mae_d2m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P2")

        df = df.append({"mae_d1m1":np.mean(mae_d1m1),
                        "mae_d1m2":np.mean(mae_d1m2),
                        "mae_d2m1":np.mean(mae_d2m1),
                        "mae_d2m2":np.mean(mae_d2m2)}, True)
    
    df.to_excel(r"results/fit_with_increasing_windowsize.xlsx")
    df.to_csv(r"results/fit_with_increasing_windowsize.csv")
    
    return df

#%%
df = fit_with_moving_window(365, 50)
#df = fit_with_increasing_windowsize(365, max_len = None)
#%%
df = pd.read_csv(r"results/fit_with_moving_window.csv")
df["mae_diff_d1"] = (df["mae_d1m1"]-df["mae_d1m2"])/(df.max()["mae_d1m1"]-df.min()["mae_d1m2"])
df["mae_diff_d2"] = (df["mae_d2m1"]-df["mae_d2m2"])/(df.max()["mae_d2m1"]-df.min()["mae_d2m2"])
#%%
plt.plot(df["mae_diff_d1"], color="red", label = "Observed GPP")
plt.plot(df["mae_diff_d2"], color="blue", label = "Simulated GPP")
plt.legend()
#%%
df = pd.read_csv(r"results/fit_with_increasing_windowsize.csv")
df["mae_diff_d1"] = (df["mae_d1m1"]-df["mae_d1m2"])/(df.max()["mae_d1m1"]-df.min()["mae_d1m2"])
df["mae_diff_d2"] = (df["mae_d2m1"]-df["mae_d2m2"])/(df.max()["mae_d2m1"]-df.min()["mae_d2m2"])
#%%
plt.plot(df["mae_diff_d1"], color="red", label = "Observed GPP")
plt.plot(df["mae_diff_d2"], color="blue", label = "Simulated GPP")
plt.legend()
