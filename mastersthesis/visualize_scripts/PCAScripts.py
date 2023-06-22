# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:11:01 2023

@author: kisen
"""
from torch import matmul
import torch

#%% Helper functions to learn and plot data in PCA space. Useful for showcasing trained embeddings in multi-D
# This function sets up the PCA-space, and returns variables needed to rebuild it
def PCALearn(trainingData, device):
    
    
    # Expects shape: trainingData -> [numPoints, numDims]
    
    
    # Restructuring the data tensor
    # imgs = torch.permute(trainingData, (2,0,1))
    # imgsFlatten = imgs.flatten(1)
    imgsFlatten = trainingData.flatten(1)
    
    # Useful variables
    nPoints = imgsFlatten.shape[0]
    onesVector = torch.ones(nPoints, 1)
    
    # Preprocessing the dataset by removing the mean observation
    imgsFlattenAvg = torch.mean(imgsFlatten, dim=0).unsqueeze(0)
    dataBar = matmul(onesVector.to(device), imgsFlattenAvg)
    B = imgsFlatten - dataBar
    
    # Doing the Singular Value Decomposition on the dataset
    _, S, Vt = torch.linalg.svd(B, full_matrices=True)
    
    # Transposing the Vtranspose matrix
    # V = Vt.mH
    V = torch.transpose(Vt, 0, 1)
    
    principalComponents = matmul(B, V)
    
    return principalComponents, dataBar[0,:].unsqueeze(0), V, S

# To predict on validation data, we must first transform the validation data to PCA space
# To do that, we need to transform the data in the same way 
def PCATransformData(evaluationData, dataBarTrained, V, device):
    
    # Restructuring the data tensor
    # imgs = torch.permute(evaluationData, (2,0,1))
    # imgsFlatten = imgs.flatten(1)
    
    # Useful variables
    nPoints = evaluationData.shape[0]
    onesVector = torch.ones(nPoints, 1)
    
    # Broadcasting the mean row
    dataBarTrained = matmul(onesVector.to(device), dataBarTrained)
    
    # Preprocessing the dataset by removing the mean observation from the 
    # training data of the PCA transformation.
    
    B = evaluationData - dataBarTrained
    
    principalComponents = matmul(B, V)
    
    # The rows are the samples, and the columns are the principal component features
    return principalComponents