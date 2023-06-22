# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:56:12 2023

@author: kisen
"""

import torch

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=0.2):
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