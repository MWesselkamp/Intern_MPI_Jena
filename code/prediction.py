#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 07:48:37 2021

@author: Marieke_Wesselkamp
"""
import numpy as np
import torch
from sklearn import metrics

import models 
import utils

import os.path
#%%
def predict(hparams, model_design, X, Y,data,
                   data_dir="models/mlp", splits=5):
    
    x = torch.tensor(X).type(dtype=torch.float)
    y = torch.tensor(Y).type(dtype=torch.float)
    
    mae = np.zeros(splits)
    nse = np.zeros(splits)
    preds = np.zeros((splits, len(Y)))
    
    for i in range(splits):
        
        model = models.MLP(model_design["layer_sizes"])
        model.load_state_dict(torch.load(os.path.join(data_dir, f"{data}_model{i}.pth")))
        model.eval()
        
        with torch.no_grad():
            preds[i,:] = model(x).squeeze(1)
            mae[i] = metrics.mean_absolute_error(y, preds[i,:])
            nse[i] = utils.nash_sutcliffe(y.numpy().squeeze(1), preds[i,:])
            
    return preds, mae, nse