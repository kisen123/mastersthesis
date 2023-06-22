# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:43:09 2023

@author: kisen
"""

import torch

# This function aims to calculate how many same-different pairs there are in 
# a dataset

def numSameDiff(datasetDictionary):
    
    
    IDs = list(datasetDictionary.keys())
    
    labels = []
    for ID in IDs:
        
        numUniqueIDs = datasetDictionary[ID].shape[0]
        
        for uniquePic in range(0, numUniqueIDs):
            labels.append(ID)
    
    Y = torch.zeros((len(labels), len(labels)))
    for i, label in enumerate(labels):
            
        Y[i, :] = torch.tensor([int(labelIteration
                                    != label) for
                                labelIteration in
                                labels])
        
    triuIndices = torch.triu_indices(Y.shape[0],
                                     Y.shape[1], offset=1)
    
    # We flatten the upper triangular matrix
    YFlat = Y[triuIndices[0], triuIndices[1]]
    
    numSame = torch.sum(YFlat == 0)
    numDiff = torch.sum(YFlat == 1)
    return int(numSame), int(numDiff)