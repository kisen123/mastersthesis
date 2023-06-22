# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:41:17 2023

@author: kisen
"""

import cv2
import numpy as np
import math
import torch


def rotationMatrix(dimX, dimY, randomAngle):
    
    # Getting the rotation matrix
    M = cv2.getRotationMatrix2D((dimX / 2, dimY / 2), randomAngle, 1)
    
    return M

# Helper function to rotate images
def rotateIm(image, angle):
    
    # Getting the dimensions
    dimX, dimY = image.shape[:2]
    
    # Getting the rotation matrix
    M = rotationMatrix(dimX, dimY, angle)
    
    # Rotating the image
    rotatedIm = cv2.warpAffine(image, M, (dimX, dimY))
    
    return rotatedIm

# Helper function to transform the images into polar coordinates.
def polarIm(image, transform_type='linearpolar'):
    """
    This function takes multiple images, and apply polar coordinate conversion to it.
    """
    
    if type(image) == torch.Tensor:
        image = np.array(image)
    # image = np.array(image.cpu())
    dimX, dimY = image.shape
    
            
    if transform_type == 'logpolar':
        image = cv2.logPolar(image, (dimX // 2, dimY // 2),
                             dimY / math.log(dimY / 2), cv2.INTER_LINEAR).reshape(dimX, dimY)
    elif transform_type == 'linearpolar':
        image = cv2.linearPolar(image, (dimX // 2, dimY // 2), dimY // 2, cv2.INTER_LINEAR)


    return image


def polarTransform(imagesDictionary, polarBlurMask=(0, 0), transform_type='linearpolar'):
    
    polarTransformedDictionary = {}
    
    for ID in list(imagesDictionary.keys()):
        
        sampleNums = len(imagesDictionary[ID])
        
        polarTransformedDictionary[ID] = torch.empty(imagesDictionary[ID].shape)
        
        # Because tuples are immutable, they must be reinstantiated instead of
        # rewritten
        
        
        for sampleNum in range(0, sampleNums):

            polarImage = polarIm(imagesDictionary[ID][sampleNum])
            
            polarTransformedDictionary[ID][sampleNum] = torch.tensor(polarImage)
        
    imagesDictionary = polarTransformedDictionary
    
    return imagesDictionary
    

def rotationTransform(imagesDictionary, rotationRange):
    
    rotationSeedGenerator = np.random.RandomState(0)
    rotatedImagesDictionary = {}


    # If we choose to rotate the dataset, the code will do just that.
    
        
    # We loop through the IDs
    for ID in list(imagesDictionary.keys()):

        sampleNums = len(imagesDictionary[ID])
        
        # Because tuples are immutable, they must be reinstantiated instead of
        # rewritten
        rotatedImagesDictionary[ID] = ()
        
        # Each sample is rotated randomly, by the RNG-variable.
        for sampleNum in range(0, sampleNums):
            
            # Setting a fixed, different seed for each for loop interaction
            randAngle = rotationSeedGenerator.randint(rotationRange[0], rotationRange[1])
            
            rotatedImage = rotateIm(imagesDictionary[ID][sampleNum], randAngle)
            
            # if polarBool:
            #     rotatedImage = np.expand_dims(rotatedImage, axis=(0,1))
            #     rotatedImage = polarTransform(rotatedImage)[0][0]
            
            rotatedImagesDictionary[ID] += (rotatedImage, )
    
    imagesDictionary = rotatedImagesDictionary
    
    return imagesDictionary