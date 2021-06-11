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
X, Y, Y_Preles = preprocessing.preprocessing()

X_P1s , Y_P1s, Y_Preles_P1s = preprocessing.split_by_year(X, Y, Y_Preles,
                                                    years = [2001])
X_P2s , Y_P2s, Y_Preles_P2s = preprocessing.split_by_year(X, Y, Y_Preles,
                                                    years = [2002])


X_P1 , Y_P1, Y_Preles_P1 = preprocessing.split_by_year(X, Y, Y_Preles,
                                                    years = [2001, 2002])
X_P2 , Y_P2, Y_Preles_P2 = preprocessing.split_by_year(X, Y, Y_Preles,
                                                    years = [2003, 2004])

#%% Plot Data
visualizations.plot_data(Y_P1, Y_Preles_P1, True, True)
visualizations.plot_data(Y_P2, Y_Preles_P2, False, False)

#%% Specify model structure and hyperparameter settings.
randomsearch = False
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
#%%
font = {'size'   : 18,
        'weight' : 'normal'}

plt.rc('font', **font)
#%% Train m1 and m2 on D1P1 and D1P2 and on D2P1 and D2P2
running_losses_d1p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "D1P1")
running_losses_d1p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(), "D1P2")

running_losses_d2p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_Preles_P1.to_numpy(), "D2P1")
running_losses_d2p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P2")

#running_losses_d1p1s = training.train(hparams_setting, model_design, X_P1s.to_numpy(), Y_P1s.to_numpy(), "D1P1s")
#running_losses_d1p2s = training.train(hparams_setting, model_design, X_P2s.to_numpy(), Y_P2s.to_numpy(), "D1P2s")

#running_losses_d2p1s = training.train(hparams_setting, model_design, X_P1s.to_numpy(), Y_Preles_P1s.to_numpy(), "D2P1s")
#running_losses_d2p2s = training.train(hparams_setting, model_design, X_P2s.to_numpy(), Y_Preles_P2s.to_numpy(), "D2P2s")
#%%
visualizations.plot_running_losses(running_losses_d1p1["mae_train"], running_losses_d1p1["mae_val"])
#%%
#visualizations.plot_running_losses(running_losses_d1p1s["mae_val"], running_losses_d1p1["mae_val"], labels = ["365 data points", "730 data points"])
#visualizations.plot_running_losses(running_losses_d2p1s["mae_val"], running_losses_d2p1["mae_val"], labels = ["365 data points", "730 data points"])

#%% Predict with fitted models to D1P2.
preds_d1m1, mae_d1m1, nse_d1m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P1")
preds_d1m2, mae_d1m2, nse_d1m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P2")
print(np.mean(mae_d1m1))
print(np.mean(mae_d1m2))
print(np.mean(mae_d1m2)-np.mean(mae_d1m1))
visualizations.plot_running_losses(running_losses_d2p1["mae_train"], running_losses_d2p1["mae_val"])
#%%
visualizations.plot_predictions(Y_P2, preds_d1m1, preds_d1m2, mae_d1m1, mae_d1m2) 

visualizations.plot_prediction_differences(preds_d1m2, preds_d1m1)

#%% Predict with fitted models to D2P2.
preds_d2m1, mae_d2m1, nse_d2m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P1")
preds_d2m2, mae_d2m2, nse_d2m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P2")
#%%
visualizations.plot_predictions(Y_Preles_P2, preds_d2m1, preds_d2m2, mae_d2m1, mae_d2m2)
visualizations.plot_prediction_differences(preds_d2m1, preds_d2m2)
#%%
import torch
import models 

model_d1p1 = models.MLP(layersizes)
model_d1p1.load_state_dict(torch.load(r"models/mlp/D1P1_model0.pth"))

weights = []
for name, param in model_d1p1.named_parameters():
    weights.append(param)
    print(param.shape)
    print(name)


#%%
def fit_with_moving_window(windowsize, seq_len):
    
    hparams_setting = {"epochs":500,
                       "batchsize":hparams[1],
                       "learningrate":hparams[0],
                       "history":1}
    model_design = {"layer_sizes":layersizes}
    mae_diff_d1 = []
    mae_diff_d2 = []
    
    df = pd.DataFrame(columns = ["mae_d1m1", "mae_d1m2", "mae_d2m1", "mae_d2m2",
                                 "nse_d1m1","nse_d1m2","nse_d2m1","nse_d2m2"])

    for i in range(seq_len):
    
        X_P1 , Y_P1, Y_Preles_P1 = preprocessing.split_by_sequence(X, Y, Y_Preles,
                                                    start = i, stop=i+windowsize)
        X_P2 , Y_P2, Y_Preles_P2 = preprocessing.split_by_sequence(X, Y, Y_Preles,
                                                    start = i+windowsize, stop=i+windowsize+windowsize)

        running_losses_d1p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "D1P1")
        running_losses_d1p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(), "D1P2")
        
        running_losses_d2p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_Preles_P1.to_numpy(), "D2P1")
        running_losses_d2p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P2")
        
        preds_d1m1, mae_d1m1, nse_d1m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P1")
        preds_d1m2, mae_d1m2, nse_d1m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P2")
        
        preds_d2m1, mae_d2m1, nse_d2m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P1")
        preds_d2m2, mae_d2m2, nse_d2m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P2")
        
        #visualizations.plot_predictions(Y_P2, preds_d1m1, preds_d1m2, mae_d1m1, mae_d1m2)
        #visualizations.plot_predictions(Y_Preles_P2, preds_d2m1, preds_d2m2, mae_d2m1, mae_d2m2)
        df = df.append({"mae_d1m1":np.mean(mae_d1m1),
                        "mae_d1m2":np.mean(mae_d1m2),
                        "mae_d2m1":np.mean(mae_d2m1),
                        "mae_d2m2":np.mean(mae_d2m2),
                        "nse_d1m1":np.mean(nse_d1m1),
                        "nse_d1m2":np.mean(nse_d1m2),
                        "nse_d2m1":np.mean(nse_d2m1),
                        "nse_d2m2":np.mean(nse_d2m2)}, True)
        
        mae_diff_d1.append(abs(np.mean(mae_d1m2)-np.mean(mae_d1m1)))
        mae_diff_d2.append(abs(np.mean(mae_d2m2)-np.mean(mae_d2m1)))
    
    df.to_excel(r"results/fit_with_moving_window.xlsx")
    df.to_csv(r"results/fit_with_moving_window.csv")
    
    return df



#%%
def fit_with_increasing_windowsize(windowsize, max_len):

    hparams_setting = {"epochs":400,
                       "batchsize":hparams[1],
                       "learningrate":hparams[0],
                       "history":1}
    model_design = {"layer_sizes":layersizes}
    
    if max_len is None:    
        max_len = int(np.floor(len(Y)/2))
        
    df = pd.DataFrame(columns = ["windowsize", "mae_d1m1", "mae_d1m2", "mae_d2m1", "mae_d2m2",
                                 "nse_d1m1","nse_d1m2","nse_d2m1","nse_d2m2"])
    
    for i in range(max_len):
    
        X_P1 , Y_P1, Y_Preles_P1 = preprocessing.split_by_sequence(X, Y, Y_Preles,
                                                                   start = 0, stop=windowsize)
        X_P2 , Y_P2, Y_Preles_P2 = preprocessing.split_by_sequence(X, Y, Y_Preles,
                                                                   start = windowsize+1, stop=1+windowsize+windowsize)
        windowsize += 20
    
        running_losses_d1p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "D1P1")
        running_losses_d1p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(), "D1P2")
        
        running_losses_d2p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_Preles_P1.to_numpy(), "D2P1")
        running_losses_d2p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P2")
        
        preds_d1m1, mae_d1m1, nse_d1m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P1")
        preds_d1m2, mae_d1m2, nse_d1m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P2")
        
        preds_d2m1, mae_d2m1, nse_d2m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P1")
        preds_d2m2, mae_d2m2, nse_d2m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P2")

        df = df.append({"windowsize":windowsize,
                        "mae_d1m1":np.mean(mae_d1m1),
                        "mae_d1m2":np.mean(mae_d1m2),
                        "mae_d2m1":np.mean(mae_d2m1),
                        "mae_d2m2":np.mean(mae_d2m2),
                        "nse_d1m1":np.mean(nse_d1m1),
                        "nse_d1m2":np.mean(nse_d1m2),
                        "nse_d2m1":np.mean(nse_d2m1),
                        "nse_d2m2":np.mean(nse_d2m2)}, True)
    
    df.to_excel(r"results/fit_with_increasing_windowsize.xlsx")
    df.to_csv(r"results/fit_with_increasing_windowsize.csv")
    
    return df

#%%
def fit_by_year():

    hparams_setting = {"epochs":500,
                       "batchsize":hparams[1],
                       "learningrate":hparams[0],
                       "history":1}
    model_design = {"layer_sizes":layersizes}
    
    years = [2000, 2001 ,2002 ,2003, 2004, 2005 ,2006, 2007, 2008 ,2009 ,2010, 2011, 2012, 2000, 2001 ,2002 ]
        
    df = pd.DataFrame(columns = ["eval_year", "mae_d1m1", "mae_d1m2", "mae_d2m1", "mae_d2m2",
                                 "nse_d1m1","nse_d1m2","nse_d2m1","nse_d2m2"])
    
    for i in range(len(years)-3):
    
        X_P1 , Y_P1, Y_Preles_P1 = preprocessing.split_by_year(X, Y, Y_Preles,
                                                    years = [years[i], years[i+1]])
        X_P2 , Y_P2, Y_Preles_P2 = preprocessing.split_by_year(X, Y, Y_Preles,
                                                    years = [years[i+2], years[i+3]])
    
        running_losses_d1p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "D1P1")
        running_losses_d1p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(), "D1P2")
        
        running_losses_d2p1 = training.train(hparams_setting, model_design, X_P1.to_numpy(), Y_Preles_P1.to_numpy(), "D2P1")
        running_losses_d2p2 = training.train(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(), "D2P2")
        
        preds_d1m1, mae_d1m1, nse_d1m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P1")
        preds_d1m2, mae_d1m2, nse_d1m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_P2.to_numpy(),"D1P2")
        
        preds_d2m1, mae_d2m1, nse_d2m1 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P1")
        preds_d2m2, mae_d2m2, nse_d2m2 = prediction.predict(hparams_setting, model_design, X_P2.to_numpy(), Y_Preles_P2.to_numpy(),"D2P2")

        df = df.append({"eval_year":years[i+1],
                        "mae_d1m1":np.mean(mae_d1m1),
                        "mae_d1m2":np.mean(mae_d1m2),
                        "mae_d2m1":np.mean(mae_d2m1),
                        "mae_d2m2":np.mean(mae_d2m2),
                        "nse_d1m1":np.mean(nse_d1m1),
                        "nse_d1m2":np.mean(nse_d1m2),
                        "nse_d2m1":np.mean(nse_d2m1),
                        "nse_d2m2":np.mean(nse_d2m2)}, True)
    
    df.to_excel(r"results/fit_by_year.xlsx")
    df.to_csv(r"results/fit_by_year.csv")
    
    return df

#%%
df3 = fit_by_year()
#%%
df1 = fit_with_moving_window(730, 365)
#%%
df2 = fit_with_increasing_windowsize(90, max_len = 100)
#%%
df3 = pd.read_csv(r"results/fit_by_year.csv")
df3["mae_diff_d1"] = abs(df3["mae_d1m1"]-df3["mae_d1m2"])#/(df3.max()["mae_d1m1"]-df3.min()["mae_d1m2"])
df3["mae_diff_d2"] = abs(df3["mae_d2m1"]-df3["mae_d2m2"])#/(df3.max()["mae_d2m1"]-df3.min()["mae_d2m2"])
df3["mae_diff_d1_scaled"] = (df3["mae_diff_d1"]-df3.min()["mae_diff_d1"])/(df3.max()["mae_diff_d1"]-df3.min()["mae_diff_d1"])
df3["mae_diff_d2_scaled"] = (df3["mae_diff_d2"]-df3.min()["mae_diff_d2"])/(df3.max()["mae_diff_d2"]-df3.min()["mae_diff_d2"])
#%%
df3 = df3[10:]
plt.plot(df3["windowsize"], df3["mae_diff_d1"], color="forestgreen", label = "$D1$")
plt.plot(df3["windowsize"], df3["mae_diff_d2"], color="blue", label = "$D2$")
plt.ylabel("MAE$_{P2_{diff}}$")
plt.xlabel("Data points in P1,P2")
plt.legend()
#%%
df_bp1 = df3[["mae_diff_d1", "mae_diff_d2"]]
df_bp2 = df3[["mae_diff_d1_scaled", "mae_diff_d2_scaled"]]
#%%
plt.plot(figsize=(3,5))
plt.boxplot(df_bp1, labels=["D$_{observed}$", "D$_{preles}$"], showmeans = True, )
#plt.ylim(-0.1,1.1)
plt.ylabel("Biotic effect size ($\epsilon$)")
#%%
plt.subplot(1, 2, 2)
plt.boxplot(df_bp2, labels=["D$_{obs}$", "D$_{pre}$"])
plt.ylim(-0.1,1.1)
plt.ylabel("Scaled difference in MAEs")
plt.tight_layout(pad=1.0)
#%%
df3 = pd.read_csv(r"results/fit_with_increasing_windowsize.csv")

plt.scatter(df3["mae_d1m1"], df3["mae_d1m2"], c=df3["windowsize"], label="$D1$")#, color="red")
plt.xlabel("$\epsilon_{m1}$")
plt.ylabel("$\epsilon_{m2}$")
plt.xlim(0. ,1.5)
plt.ylim(0. ,1.5)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
#plt.legend()
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = 0 + 1 * x_vals
plt.plot(x_vals, y_vals, '--', c="black")
#%%
plt.scatter(df3["mae_d2m1"], df3["mae_d2m2"], c=df3["windowsize"], label="$D2$")#, color="blue")
plt.xlabel("$\epsilon_{m1}$")
plt.ylabel("$\epsilon_{m2}$")
plt.xlim(0. ,1.5)
plt.ylim(0. ,1.5)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
#plt.legend()
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = 0 + 1 * x_vals
plt.plot(x_vals, y_vals, '--', c="black")

#%%
df = pd.read_csv(r"results/fit_with_increasing_windowsize.csv")
df["mae_diff_d1"] = (df["mae_d1m1"]-df["mae_d1m2"])/(df.max()["mae_d1m1"]-df.min()["mae_d1m2"])
df["mae_diff_d2"] = (df["mae_d2m1"]-df["mae_d2m2"])/(df.max()["mae_d2m1"]-df.min()["mae_d2m2"])
#%%
plt.plot(df["mae_diff_d1"], color="red", label = "Observed GPP")
plt.plot(df["mae_diff_d2"], color="blue", label = "Simulated GPP")
plt.legend()

#%%
se = utils.nash_sutcliffe(Y_P1.to_numpy().squeeze(1), preds_d1m1)