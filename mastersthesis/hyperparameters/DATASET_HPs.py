# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 15:39:05 2023

@author: kisen
"""

import itertools

DATASET_HP_DICT = {
    'noiseType': ['gaussian'],
    
    'noiseAmount': [0.02],
    
    'noiseMeanLow': [0.45],
    
    'noiseMeanHigh': [0.55],
    
    'speckleSizeLow': [0.2],
    
    'speckleSizeHigh': [1.4],
    
    'speckleRotationLow': [0],
    
    'speckleRotationHigh': [360],
    
    # 'imageRotationTuple': [(-angle, angle) for angle in list(range(25, 91, 5))],
    
    'imageRotationTuple': [(-180, 180)],
    
    'speckleVaryBool': [False],
    
    'dx': [1],
    
    'dy': [1],
    
    'numSpecklesLow': [150],
    
    'numSpecklesHigh': [350],
    
    'blackLow': [0.2],
    
    'blackHigh': [0.8],
    
    'blacknessLow': [0.45],
    
    'blacknessHigh': [0.95],
    
    'digestiveBlackLow': [0.0], 
    
    'digestiveBlackHigh': [0.5],
    
    'numIDs': [75],
    
    'minNumSamples': [4],

    'maxNumSamples': [4],
    
    'polarBool': [True],
    
    'blurMaxSize': [9],
    
    'standardizeBool': [True],
    
    'trainValTestRatio': [(40/75, 15/75, 20/75)]
    
    # 'trainValTestRatio': [(80/100, 15/100, 5/100)]
}




# This commented-out dataset provides a hyperparameter reference that gives 
# good images.
# DATASET_HP_DICT = {
#     'noiseAmount': [0.02],
    
#     'noiseMeanLow': [0.4],
    
#     'noiseMeanHigh': [0.6],
    
#     'speckleSizeLow': [0],
    
#     'speckleSizeHigh': [0.5],
    
#     'rotationLow': [0],
    
#     'rotationHigh': [360],
    
#     'speckleVaryBool': [False],
    
#     'numSpecklesLow': [150],
    
#     'numSpecklesHigh': [300],
    
#     'blackLow': [0.2],
    
#     'blackHigh': [0.7],
    
#     'blacknessLow': [0.2],
    
#     'blacknessHigh': [0.8],
    
#     'digestiveBlackLow': [0], 
    
#     'digestiveBlackHigh': [0.7],
    
#     'numIDs': [5],
    
#     'minNumSamples': [5],

#     'maxNumSamples': [5],
# }



# THIS DICTIONARY IS TO BE USED FOR BOTORCH OR WANDB, CAN'T REMEMBER WHICH ONE.
# DATASET_HP_WANDB = {
    
#     'noiseAmount': {
#         'values': [0.05]
#         },
    
#     'noiseMeanLow': {
#         'values': [0.4]
#         },
    
#     'noiseMeanHigh': {
#         'values': [0.6]
#         },
    
#     'speckleSizeLow': {
#         'values': [0]
#         },
    
#     'speckleSizeHigh': {
#         'values': [0.5]
#         },
    
#     'rotationLow': {
#         'values': [0]
#         },
    
#     'rotationHigh': {
#         'values': [360]
#         },
    
#     'speckleVaryBool': {
#         'values': [False]
#         },
    
#     'numSpecklesLow': {
#         'values': [150]
#         },
    
#     'numSpecklesHigh': {
#         'values': [300]
#         },
    
#     'blackLow': {
#         'values': [0.2]
#         },
    
#     'blackHigh': {
#         'values': [0.7]
#         },
    
#     'blacknessLow': {
#         'values': [0.2]
#         },
    
#     'blacknessHigh': {
#         'values': [0.8]
#         },
    
#     'digestiveBlackLow': {
#         'values': [0]
#         },
    
#     'digestiveBlackHigh': {
#         'values': [0.7]
#         },
    
#     'numIDs': {
#         'values': [50]
#         },
    
#     'minNumSamples': {
#         'values': [5]
#         },

#     'maxNumSamples': {
#         'values': [5]
#         },



# }

# Create a list of all possible combinations of hyperparameters
DATASET_HP = list(itertools.product(*DATASET_HP_DICT.values()))

DATASET_HP_LIST = []

# Loop over each combination of hyperparameters
for hyperparams in DATASET_HP:
    
    # Converts the combination back to a dictionary, and stores them in a list
    DATASET_HP_LIST.append(dict(zip(DATASET_HP_DICT.keys(), hyperparams)))