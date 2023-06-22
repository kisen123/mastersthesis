# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:57:18 2023

@author: kisen
"""

import torch

class distanceMetric():
    
    # Constructor to set the user-defined metric
    def __init__(self, distMetric, p=2.0):
        self.distMetric = distMetric
        
        if distMetric == "p-norm":
            self.distFunc = torch.nn.PairwiseDistance(p=p)
            
        elif distMetric == "cosine":
            self.distFunc = torch.nn.CosineSimilarity(dim=1)
            
        elif distMetric == "absolute":
            self.distFunc = torch.abs
            
    # Call-method to return the distance.
    def __call__(self, input1, input2):
        if self.distMetric == "absolute":
            return self.distFunc(input1 - input2)
        else:
            return self.distFunc(input1, input2)
