# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:21:58 2023

@author: kisen
"""

import numpy as np

def confusion(predictedLabels, actualLabels):
    
    # This helper function is used to create the confusion matrix of the 
    # siamese network approach, using ContrastiveLoss and using the convention:
    # flag=0 means same class, flag=1 means different.
    # NOTE THAT IT IS A HARD CODED APPROACH.
    
    # This subsection localizes all the predicted labels. HARDCODED FOR BINARY
    predSame = predictedLabels == 0
    predSameInt = np.array([int(item) for item in predSame])
    predDiff = predictedLabels == 1
    predDiffInt = np.array([int(item) for item in predDiff])
    
    # This subsection localizes all the true labels. HARDCODED FOR BINARY
    trueSame = actualLabels == 0
    trueSameInt = np.array([int(item) for item in trueSame])
    trueDiff = actualLabels == 1
    trueDiffInt = np.array([int(item) for item in trueDiff])
    
    
    
    # Where two instances of a true or false (2) occurs, that determines where
    # in the confusion matrix the point will be placed.
    
    # This subsection covers all the correct classifications (trues) in the matrix
    truePositives = np.sum(predSameInt + trueSameInt == 2)
    trueNegatives = np.sum(predDiffInt + trueDiffInt == 2)
    
    # This subsection covers all the incorrect classifications (falses) in the matrix
    falseNegatives = np.sum(predDiffInt + trueSameInt == 2)
    falsePositives = np.sum(predSameInt + trueDiffInt == 2)
    
    # Constructing the 2D confusion matrix, and returning it.
    confMatrix = np.array([[truePositives, falsePositives], 
                           [falseNegatives, trueNegatives]])
    
    return confMatrix