# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:44:46 2023

@author: kisen
"""

import itertools

from similarity_models.model_classes import Siamese_LeNet5_var, \
    Siamese_MobileNetV3_var, Siamese_ResNet18_var


from torch import optim

MODEL_HP_DICT = {
    'epochs': [50],
    
    'learningRate': [0.0005],
    
    'batchSize': [128],
    
    'percentOfPairs': [1.0],
    
    'lossFunction': ["Triplet"],
    
    'similarityFlag': [False],
    
    
    
    
    'CylindricalModel': [True], 'model': [Siamese_LeNet5_var], 
    
    
    
    
    'optimizer': [optim.Adam]
}


# Create a list of all possible combinations of hyperparameters
MODEL_HP = list(itertools.product(*MODEL_HP_DICT.values()))

MODEL_HP_LIST = []

# Loop over each combination of hyperparameters
for hyperparams in MODEL_HP:
    
    # Converts the combination back to a dictionary, and stores them in a list
    MODEL_HP_LIST.append(dict(zip(MODEL_HP_DICT.keys(), hyperparams)))