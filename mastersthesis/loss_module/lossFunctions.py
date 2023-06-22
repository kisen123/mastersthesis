# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:12:57 2023

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


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=1.0):
        # Inherit from the torch.nn.Module, and defining the margin variable
        super(ContrastiveLoss, self).__init__() 
        self.m = m

    def forward(self, y1, y2, flag):
        
        # flag = 0 means y1 and y2 are supposed to be same
        # flag = 1 means y1 and y2 are supposed to be different
        eucDist = torch.nn.functional.pairwise_distance(y1, y2)
        
        # torch.clamp is a clever way to set max()
        loss = torch.mean((1-flag) * torch.pow(eucDist, 2) +
          (flag) * torch.pow(torch.clamp(self.m - eucDist, min=0.0), 2))

        return loss
    

# The distanceMetric class must be in the same directory as TripletLoss

class TripletLoss(torch.nn.Module):
    def __init__(self, m=1.0):
        # Inherit the nn.Module attributes
        super(TripletLoss, self).__init__()
        self.m = m
        
    def forward(self, anchor, positive, negative):
        
        # Instantiate the distance metric class, 
        # and assign to it a metric measure
        distFunc = distanceMetric("p-norm", p=2.0)
        
        # Defining the distance instances for the loss function
        distAP = distFunc(anchor, positive)
        distAN = distFunc(anchor, negative)
        
        # torch.relu is a clever way to set the max{L, 0} condition.
        marginDistance = torch.clamp(distAP - distAN + self.m, min=0.0)
        loss = torch.mean(marginDistance)
        
        return loss