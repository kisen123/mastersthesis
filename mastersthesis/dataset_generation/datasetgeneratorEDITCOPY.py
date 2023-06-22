# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:37:56 2022

@author: kisen
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image
import cv2 as cv
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle, Circle

import os
import glob

import torch
import math
import random




"""

This script generates synthetic images of salmon lice, masked in a circular 
fashion to mimic the received dataset. 

It can be tuned with many options, and this docstring serves as an instruction
manual as to how they are generated

noiseTuple : tuple  ->  ('gaussian', noiseStd),  TODO

noiseStd : float  ->  > 0, determines how noisy the image gets

noiseMeanRange : tuple  ->  (low, high), determines the greyscale of the initial 
                                   background image. Sample from uniform dist.

numSamples : list  ->  [int, int, ...], each ID has a number of <int> samples

speckleGeometry : tuple  ->  (scale-width, scale-height, rotation, randomBool)    
#EXAMPLE: (0, 0.5) is a good range, and being between 0 and 1 is optimal.
rotation is a tuple that determines the rotation range, i.e. (0, 360)
randomBool sets the seed for each ID, such that an ID does/doesn't vary the 
speckle geometry'

blacknessInterval : tuple  ->  (black-opacity-low, black-opacity-high), 
sets the overall blackness of speckles, and is targeted to EVERY speckle 
instead of individual ones.

digestiveInterval : tuple -> (black-opacity-low, black-opacity-high),
sets the blackness of the synthetic digestive channel. The position and geometry
of the digestive system is a TODO project, but generally appear as rectangular
shapes.
"""



#%% Helper function to set seed
def setSeed(seed, cuda=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


# This little helper function determines the rotation matrix for the given
# image to be rotated, re-calculated for every single image......
def rotatePoints(angleRotation):
    
    
    radianRotation = angleRotation * np.pi / 180
    
    
    cos, sin = np.cos(radianRotation), np.sin(radianRotation)
    rotationMatrix = np.array([[cos, -sin], [sin, cos]])
    
    return rotationMatrix
    
#%%
# This script will, given the user's inputs, output a dataset for transfer 
# learning for the task of recognizing and classifying salmon lice.


def makeClosedLoops(dimX, dimY, numRands=10, curvy=False):
    
    t = np.linspace(0,2*np.pi,101);
    r = np.ones(t.shape);
    
    if curvy: 
        rho = np.random.rand(1,numRands) * np.logspace(-0.5, -2.5, numRands)
        phi = np.random.rand(1, numRands) * 2*np.pi
        
        
        for h in range(0,numRands):
          r = r + rho[0, h]*np.sin(h*t+phi[0, h])
    
    
        # Makes curvy circles
        x = np.floor(dimX/4 * r * np.cos(t) + dimX/2)
        y = np.floor(dimY/4 * r * np.sin(t) + dimY/2)
    else:
        x = np.floor((dimX/2 - 3) * np.cos(t) + dimX/2)
        y = np.floor((dimY/2 - 3) * np.sin(t) + dimY/2)
    return x, y

#%% CODE FROM https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
def circMask(h, w):

    center = ((w-1) / 2, (h-1) / 2)
    radius = w/2
    

    Y, X = np.ogrid[:h, :w]
    distFromCenter = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = distFromCenter <= radius
    return mask
#%%

# This has been proven to work fine.
# datasetGenerator.makeImage(codeDir, 
#                            noiseTuple=('gaussian', 0.01),
#                            noiseMeanRange=(0.4, 0.6),
#                            numSamples=numSamples,
#                            speckleGeometry=(0, 0.5, (0, 360), False),
#                            my_dpi=54.3, 
#                            numSpecklesInterval=(150, 300),
#                            blackInterval=(0.7, 0.7),
#                            blacknessInterval=(0.2, 0.8),
#                            digestiveInterval=(0, 0.7))

def makeImage(codeDir,
              
              noiseType='gaussian',
              
              noiseAmount = 0.05,
              
              dimX=64, dimY=64,
              
              noiseMeanRange=(0.4, 0.4), 
              
              numSamples=None, 
              
              speckleGeometry=(0, 1, (0, 360), False),
              
              specklePerturb=(0, 0),
              
              my_dpi=54.3,
              
              numSpecklesInterval=(150, 300), 
              
              blackInterval=(0.2, 0.7),
              
              blacknessInterval=(0.2, 0.8),
              
              digestiveInterval=(0, 0.5),
              
              imageRotationTuple=(0, 0)):
    
    # We initially start in the main folder
    os.chdir(codeDir)
    
    # Makes a directory to store the images.    
    if os.path.exists(os.path.join(os.getcwd(), "syntheticFolder")) == False:
        os.mkdir("C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts/syntheticFolder")
        
    if os.path.exists(os.path.join(os.getcwd(), "syntheticFolder/images")) == False:
        os.mkdir("C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts/syntheticFolder/images")
        
    syntheticDataPath = "C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts/syntheticFolder/images"
    os.chdir(syntheticDataPath)
        
    # If the variable numSpecklesInterval is a value range, we need to seed the
    # variable properly (done inside the double for loop)
    if numSpecklesInterval[0] != numSpecklesInterval[1]:
        numSpecklesVariation = True
    else:
        numSpecklesVariation = False
        
        
    fileRemove = True
    if fileRemove:
        os.chdir(syntheticDataPath)
        filesToRemove = os.listdir()
        for f in filesToRemove:
            os.remove(f)
            
    
    # We set the seed for reproducibility
    setSeed(13)
    
    # Setting a few standard variables.
    dimX = 64
    dimY = 64
    dimImages = (dimX, dimY)
    loopMask = circMask(dimX, dimY)
    
    center = [(dimX - 1) / 2, (dimY - 1) / 2]
    lowerAngle, higherAngle = imageRotationTuple[0], imageRotationTuple[1]
    
    # We set the randomizers fixed. I.e. we want the numbers to be random in 
    # nature, but if we want the randomness to be fixed, we much sample from
    # from the same RandomState. They are instantiated inside the double for loop.
    if speckleGeometry[3] == True:
        geometricRNG = np.random.RandomState(0)
        
    blackRNG = np.random.RandomState(2)
        
    datasetRotate = True
    if datasetRotate:
        rotateRNG = np.random.RandomState(5)
    
    blacknessRNG = np.random.RandomState(3)

    
    positionRNG = np.random.RandomState(4)
    
    dxMin, dyMin = 0, 0
    dxMax, dyMax = specklePerturb[0], specklePerturb[1]
    
    
    if dxMax != 0.0 and dyMax != 0.0:
        positionPerturb = True
    else:
        positionPerturb = False
    
    # Instantiating the figure to write images to.
    fig, ax = plt.subplots(figsize=(dimX/my_dpi, dimY/my_dpi))
    
    # The enumeration index is individual, and numSamples is the number of 
    # samples per individual.
    for ID, samples in enumerate(numSamples):
    

        # Individual-iterator variable
        IDNum = 0
        
        # Looping over each individual-sample.
        for sample in range(samples):
            
            # Setting rotation options
            angleRotation = rotateRNG.uniform(lowerAngle, higherAngle)
            rotationMatrix = rotatePoints(angleRotation)
            
            
            
            # Setting the image noise and brightness
            noiseMean = np.random.uniform(noiseMeanRange[0], noiseMeanRange[1])
            initGreyscale = noiseMean*np.ones(dimImages)
            
            if noiseType == 'gaussian':
                noiseAdd = np.random.normal(0, noiseAmount, dimImages)
                backgroundImage = initGreyscale + noiseAdd
                
                
            
        
            # We fix the randomization for each ID if the user does not want
            # these variables to vary.
            if speckleGeometry[3] == False:
                geometricRNG = np.random.RandomState(ID)
                
            # if blackVariation == False:
            #     blackRNG = np.random.RandomState(ID)
                
            if numSpecklesVariation == True:
                numSpecklesRNG = np.random.RandomState(ID)  
                numSpeckles = numSpecklesRNG.randint(numSpecklesInterval[0],
                                                     numSpecklesInterval[1])
                
            else:
                numSpeckles = numSpecklesInterval
            
            # Initial coordinates must be the same for the same ID.
            speckleCoordinatesRNG = np.random.RandomState(ID)
            speckleCoordinates = speckleCoordinatesRNG.randint(0, dimX, size=(numSpeckles, 2))
            
            # Blackness variation describes how black the majority of the dots
            # should be
            # if blacknessVariation == False:
            #     blacknessRNG = np.random.RandomState(ID)
                
            blacknessGround = blacknessRNG.uniform(blacknessInterval[0],
                                                   blacknessInterval[1])
            
            
            newSpeckleCoordinates = np.empty(speckleCoordinates.shape)
            
            # Makes a perturbation for each point and coordinate
            if positionPerturb:
                dx = positionRNG.uniform(dxMin, dxMax, size=(numSpeckles, 1))
                dy = positionRNG.uniform(dyMin, dyMax, size=(numSpeckles, 1))
            else:
                dx, dy = np.zeros((numSpeckles, 1)), np.zeros((numSpeckles, 1))
            
            specklePerturbation = np.concatenate((dx, dy), axis=1)
            speckleCoordinates = speckleCoordinates + specklePerturbation
            
            
            
            # The rectangle specs are set. This rotates the rectangular speckles,
            # ultimately making the dataset more unique.
            randomAngles = geometricRNG.randint(speckleGeometry[2][0],
                                               speckleGeometry[2][1], size=(numSpeckles))
            
            randomWidthsHeights = geometricRNG.uniform(speckleGeometry[0],
                                                       speckleGeometry[1],
                                                       size=(numSpeckles, 2))
            
            if datasetRotate:
                speckleCoordinates = np.dot(rotationMatrix, 
                                                   (speckleCoordinates - center).T).T + center
            else:
                speckleCoordinates = speckleCoordinates
            
            
            # Create a random blackness for individual speckles, and thresholding
            # it at 1
            blackness = blackRNG.uniform(blackInterval[0],
                                             blackInterval[1] +\
                                             blacknessGround, size=(numSpeckles))
            blackness[blackness >= 1] = 1
            
            
                
            for i, (x, y) in enumerate(speckleCoordinates):
                
                
                rectangle = Rectangle((x, y), randomWidthsHeights[i, 0], 
                                      randomWidthsHeights[i, 1], randomAngles[i], color='black',
                                      alpha=blackness[i])
                ax.add_patch(rectangle)
    

            
        
            digestiveBlack = blacknessRNG.uniform(digestiveInterval[0],
                                                  digestiveInterval[1])
            imgName = str(ID) + "-" + str(IDNum) + '.tif'
            
            
            # Adding a bigger black dot in the center
            ax.scatter(dimX/2, dimY/2, s=10, c='black', alpha=0.95)
            
            # Making a black line to set orientation
            middlePoint = (round(dimX/2), round(dimY/2))
            
            digestivePosition = np.array([dimX/2 - 1, 40])
            digestivePosition = np.dot(rotationMatrix, (digestivePosition - center).T).T + center
            
            # Adds the black line that imitates the digestive system
            digestiveRectangle = Rectangle(digestivePosition, 2, 40, angleRotation, color='black', alpha=digestiveBlack)
            
            ax.add_patch(digestiveRectangle)
            
            ax.imshow(loopMask*backgroundImage, cmap='gray', vmin=0, vmax=1,
                       interpolation="none")
            ax.axis("off")
            fig.savefig(fname=imgName, bbox_inches='tight', pad_inches=0, format='tif')
            
            plt.cla()
            
            IDNum += 1
        if samples % 10 == 0:
            print("A number of: " + str(ID+1) + "/" + str(len(numSamples)) + " duplicates created for each ID")
    plt.close()
    os.chdir(codeDir)
#%%
# Determines if we are to remove the images already created in the SyntheticData-folder


# makeImage1(numSamples=numSamples, numIDs=numIDs, geometricVariation=False,
#            noiseMean=0.4, noiseAmount=0.2, my_dpi=54.3, scaleVariation=True,
#            numSpecklesVariation=True, blacknessVariation=True)
    


#%%
# # Merging mask and outline
# bodyImage = cv.imread("testBilde2.png")[:,:,0]
# outlineImage = cv.imread("testBilde.png")[:,:,0]

# fig = plt.figure(figsize=(dimX/my_dpi, dimY/my_dpi))

# initGreyscale = 0.4*np.ones(dimImages)

# negativeImage = abs(bodyImage + outlineImage - 254)

# plt.imshow(negativeImage, cmap='gray', vmin=0, vmax=255, alpha=initGreyscale)
# plt.show()
# # plt.show()
# # plt.axis('off')
# # plt.savefig("kanskje.png", bbox_inches='tight', pad_inches=0)

# # # Create data
# # N = 500

# # scatterX = []
# # scatterY = []
# # for n in range(0,N):
# #     scatterX.append(random.randrange(0, 255))
# #     scatterY.append(random.randrange(0, 255))
# # colors = (0,0,0)
# # area = np.pi*3

# # Plot
# #plt.scatter(scatterX, scatterY, s=area, c=colors, alpha=0.5)

# #%%
# dimImages = (256,256)
# initGreyscale = 0.3*np.ones(dimImages)
# colVector = np.linspace(0, dimImages[0]-1, dimImages[0])
# rowVector = np.linspace(0, dimImages[1]-1, dimImages[1])

# colVector = colVector.astype(int)
# rowVector = rowVector.astype(int)
# colSpace, rowSpace = np.meshgrid(rowVector, colVector)

# x = x.astype(int)
# y = y.astype(int)

# for i in range(0, len(x)):
#     initGreyscale[x[i], y[i]] = 1
        
# #%%
# initGreyscale = 0.4*np.ones(dimImages)
# for r, rows in enumerate(range(0, initGreyscale.shape[1])):
#     for c, cols in enumerate(range(0, initGreyscale.shape[0])):
        
