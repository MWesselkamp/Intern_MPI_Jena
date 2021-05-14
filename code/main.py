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
import matplotlib.pyplot as plt
import numpy as np
#%%
X, Y, Y_Preles = preprocessing.preprocessing()
#%% Split data 
X_P1 , Y_P1, Y_Preles_P1 = preprocessing.split_data(X, Y, Y_Preles,
                                                    years = [2000])
X_P2 , Y_P2, Y_Preles_P2 = preprocessing.split_data(X, Y, Y_Preles,
                                                    years = [2001])

#%% Plot Data
visualizations.plot_data(Y_P1, Y_Preles_P1)
visualizations.plot_data(Y_P2, Y_Preles_P2)

#%% Specify model structure and hyperparameter settings.
hparams = {"epochs":300,
           "batchsize":16,
           "learningrate":0.01,
           "history":1}
model_design = {"layer_sizes":[7,8,16,1]}
#%% Train m1 and m2 on P1 and P2
running_losses_d1p1 = training.train(hparams, model_design, X_P1.to_numpy(), Y_P1.to_numpy(), "D1P1")

running_losses_d1p2 = training.train(hparams, model_design, X_P2.to_numpy(), Y_P2.to_numpy(), "D1P2")

#%% Predict with fitted models to P2.
preds_m1, rmse_m1, mae_m1 = prediction.predict(hparams, model_design, 
                                             X_P2.to_numpy(), Y_P2.to_numpy(),
                                             "D1P1")

preds_m2, rmse_m2, mae_m2 = prediction.predict(hparams, model_design, 
                                             X_P2.to_numpy(), Y_P2.to_numpy(),
                                             "D1P2")
#%%
for i in range(5):
    plt.plot(preds_m1[i,:])
print(np.mean(mae_m1))
#%%
for i in range(5):
    plt.plot(preds_m2[i,:])
print(np.mean(mae_m2))
#%% 
print(abs(np.mean(mae_m1)-np.mean(mae_m2)))