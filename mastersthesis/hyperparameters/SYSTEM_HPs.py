# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:54:43 2023

@author: kisen
"""

import itertools

SYSTEM_HP_DICT = {
    'seed': [123]
    }


# Create a list of all possible combinations of hyperparameters
SYSTEM_HP = list(itertools.product(*SYSTEM_HP_DICT.values()))

SYSTEM_HP_LIST = []

# Loops over each combination of hyperparameters
for hyperparams in SYSTEM_HP:
    
    # Converts the combination back to a dictionary, and stores them in a list
    SYSTEM_HP_LIST.append(dict(zip(SYSTEM_HP_DICT.keys(), hyperparams)))