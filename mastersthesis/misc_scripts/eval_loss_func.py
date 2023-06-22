# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 16:22:48 2023

@author: kisen
"""

# All necessary torch-imports
import torch
from torch import nn
torch.set_default_dtype(torch.float)


# All necessary classic imports
import numpy as np
import random
import math

# All matplotlib imports

from matplotlib import pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# Misc. imports
import os
from datetime import datetime
from misc_scripts.numSameDiff import numSameDiff
from misc_scripts.scatter_plot_with_hover import scatter_plot_with_hover
from misc_scripts.lossPlot import lossPlot
from misc_scripts.tripletMask import getSemihardTripletMask
import time

# Evaluation metric imports
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity as cosSim
from sklearn.metrics import pairwise_distances as distSim
from sklearn.metrics import auc


# Necessary scripts for synthesizing images is in this folder
codeDir = "C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts"
os.chdir(codeDir)

# Imports of PCA-scripts for visualization purposes
from visualize_scripts.PCAScripts import PCALearn, PCATransformData

# Import of PCA visualization script
from visualize_scripts.evaluationPlot import evaluationPlot


# Importing loss functions and other methods appropriate for the loss functions
from loss_module.lossFunctions import ContrastiveLoss, TripletLoss, distanceMetric

from setSeed import setSeed



def eval_loss_func(valLoader, model):
    
    
    miniBatchValLossVector = torch.tensor([])
    lossFunc = TripletLoss(m=0.2)
    
    loss_valLoader = valLoader
    # loss_valLoader.batch_size = 128
    
    # Setting the seed for reproducible val-scores.
    setSeed(456)
    
    for batchIdx, batch in enumerate(loss_valLoader):
        
        XAnchor, XPos, XNeg, labelsPos, labelsNeg = batch
        
        output1, output2, output3 = model(XAnchor,
                                          XPos,
                                          XNeg)
        
        # By the FaceNet-paper, we normalize the output embeddings
        # such that ||f(x)||_2 = 1. We further mask these embeddings
        # to select semihard-triplets.
        output1, output2, output3 = \
            torch.nn.functional.normalize(output1),\
            torch.nn.functional.normalize(output2),\
            torch.nn.functional.normalize(output3)
        semihardMask = getSemihardTripletMask(output1, output2, output3, margin=0.2)
        
        lossVal = lossFunc(output1[semihardMask], 
                           output2[semihardMask],
                           output3[semihardMask])
        
        miniBatchValLoss = lossVal.item()
        
        miniBatchValLossVector = torch.cat( (miniBatchValLossVector,
                                             torch.tensor(miniBatchValLoss).unsqueeze(0) ))
        
    
        
    return miniBatchValLossVector