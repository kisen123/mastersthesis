# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:43:01 2023

@author: Kristian Lian
"""
import torch

def getSemihardTripletMask(output1, output2, output3, margin=0.2):
    
    """
    Inputs: Tensors -> Inputs output1, output2 and output3 are batched
    forward-propagations of an embedding model, with the dimensions
    [batchSize, embeddingDim]
    
    
    Returns: A masking vector to select the semihard triplets, with the
    dimensions [batchSize]. To be masked in the training loop as such:
        
    semihardOutput = outputX[semihardMask] 
    
    """
    
    
    # Triplet margin
    margin=0.2
    
    # Norm
    p = 2.0
    
    # For testing outputs
    # output1 = torch.randn((128,256))
    # output2 = torch.randn((128,256))
    # output3 = torch.randn((128,256))
    
    
    # Detaching the embedded outputs from the computational graph, we wouldn't
    # want the .backward() pass to pass through anything unwanted.
    output1 = output1.detach()
    output2 = output2.detach()
    output3 = output3.detach()


    # Calculating euclidean distance between Anchor-Positives and Anchor-Negatives
    pdist = torch.nn.PairwiseDistance(p=p)
    
    distAP = pdist(output1, output2)
    distAN = pdist(output1, output3)
    
    # Creating the semihard mask
    semihardConstraint = distAP < distAN
    violateConstraint  = distAN <= (distAP + margin)
    semihardMask = semihardConstraint * violateConstraint
        
    
    return semihardMask