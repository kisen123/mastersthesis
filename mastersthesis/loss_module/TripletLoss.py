# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:56:44 2023

@author: kisen
"""

import torch
from loss_functions import distanceMetric
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
        loss = torch.mean(torch.clamp(distAP - distAN + self.m, min=0.0))
        
        return loss