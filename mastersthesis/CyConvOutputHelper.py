# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:48:04 2023

@author: kisen
"""


import torch
import numpy as np

from models.cyconvlayer import CyConv2dFunction, CyConv2d
import CyConv2d_cuda

#%%

nL = (49,49)
fL = 4
pL = 0
s = 1

testInput = torch.ones([1, 1, nL[0], nL[1]]).cuda()
conv1Test = CyConv2d(in_channels=1, out_channels=1, kernel_size=fL, stride=s, padding=pL).cuda()
testOutput = conv1Test(testInput)


def figureOutDimensions(nL, fL, pL, s):
    if pL == 0:
        pLNew = 0
    else:
        pLNew: int = int(max(pL, np.floor(fL / 2 - 1)))
    
    nLBefore: int = nL
    
    nLAfter: int = int(np.floor( ((nLBefore + 2 * pLNew - fL) / s) + 1 ))
    
    return nLAfter


print("Predictions: \n")
print("nLBefore: " + str(nL))

print("nAfter: ({}, {})\n\n".format(figureOutDimensions(nL[0], fL, pL, s),
                                  figureOutDimensions(nL[1], fL, pL, s)))
print("Actually: \n")
print("Size before: ({}, {})".format(testInput.shape[2],
                                     testInput.shape[3]))
print("Size after: ({}, {})".format(testOutput.shape[2],
                                     testOutput.shape[3]))

print("\nOptions: kernel-size: {} \t padding: {} \t stride: {}".format(fL, pL, s))

