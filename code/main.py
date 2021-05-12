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

#%%
X, Y, Y_Preles = preprocessing.load_data()
X = preprocessing.modify_data(X)
X = preprocessing.subset_data(X)
X = preprocessing.standardize_data(X)
#%% Split data 
X_P1 , Y_P1, Y_Preles_P1 = preprocessing.split_data(X, Y, Y_Preles,
                                                    years = [2000])
X_P2 , Y_P2, Y_Preles_P2 = preprocessing.split_data(X, Y, Y_Preles,
                                                    years = [2006, 2007, 2008, 2009, 2010, 2011])

#%% Plot Data
visualizations.plot_data(Y_P1, Y_Preles_P1)
visualizations.plot_data(Y_P2, Y_Preles_P2)

#%% Initialize Model
import models
import torch
net = models.MLP(3, [7,8,16,32,1])
print(net)
x = net(torch.tensor(X_P1.to_numpy()).type(dtype=torch.float))
#%%
hparams = {"epochs":10000,
           "batchsize":16,
           "learningrate":0.01,
           "history":1}
model_design = {"hidden_layers":3,
                "layer_sizes":[7,8,16,32,1]}

running_losses, performance, y_tests, y_preds = training.mlp_training(hparams, model_design, 
                                                                      X_P1.to_numpy(), Y_P1.to_numpy())

#%%+
import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.transpose(running_losses["mae_train"]))
print(running_losses["mae_train"][:,-1])
#%%
plt.plot(y_preds[0])
plt.plot(y_tests[0])