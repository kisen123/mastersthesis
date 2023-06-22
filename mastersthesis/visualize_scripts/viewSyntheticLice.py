# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:51:51 2023

@author: kisen
"""
import os
codeDir = "C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts"
os.chdir(codeDir)
from setSeed import setSeed

from dataset_transform.rotationScripts import rotationTransform, polarTransform
from hyperparameters import DATASET_HPs, SYSTEM_HPs
from dataset_generation import datasetGenerator, datasetExtracter
import random
from matplotlib import pyplot as plt
from visualize_scripts.licePlots import gridPlotLice, samePlotLice


# Because this very specific script is only for visualizing example synthetic
# lice, we need to unpack the list of length 1.
SYSTEM_HP = SYSTEM_HPs.SYSTEM_HP_LIST[0]
DATASET_HP = DATASET_HPs.DATASET_HP_LIST[0]

setSeed(SYSTEM_HP["seed"])

numIDs = 16
minNumSamples = 3
maxNumSamples = 3

numSamples = [random.randint(minNumSamples, maxNumSamples) for i in range(numIDs)]

print("\n...Please wait, dataset generation in progress...\n")

# This line makes synthetic images, and stores them in
# codeDir + "/syntheticFolder/images" as .tif-images
makeImages = True
if makeImages:
    
    datasetGenerator.makeImage(codeDir,
                               noiseType=DATASET_HP['noiseType'],
                               noiseAmount=DATASET_HP['noiseAmount'],
                               noiseMeanRange=(DATASET_HP['noiseMeanLow'],
                                               DATASET_HP['noiseMeanHigh']),
                               numSamples=numSamples,
                               speckleGeometry=(DATASET_HP['speckleSizeLow'],
                                                DATASET_HP['speckleSizeHigh'],
                                                (DATASET_HP['speckleRotationLow'],
                                                 DATASET_HP['speckleRotationHigh']), 
                                                 DATASET_HP['speckleVaryBool']),
                               specklePerturb=(DATASET_HP['dx'],
                                               DATASET_HP['dy']),
                               my_dpi=54.3, 
                               numSpecklesInterval=(DATASET_HP['numSpecklesLow'], 
                                                    DATASET_HP['numSpecklesHigh']),
                               blackInterval=(DATASET_HP['blackLow'], 
                                              DATASET_HP['blackHigh']),
                               blacknessInterval=(DATASET_HP['blacknessLow'], 
                                                  DATASET_HP['blacknessHigh']),
                               digestiveInterval=(DATASET_HP['digestiveBlackLow'],
                                                  DATASET_HP['digestiveBlackHigh']))
    
# This line reads all the .tif-files, and writes the images to a .pkl-file.
# Additionally, the function can return the dictionary that is written to the
# .pkl-file.
imagesDictionary = datasetExtracter.makePKLFile(realDataFlag=False, resizeTuple=(64,64))
print("\nImage generation done, they now exist in the variable: imagesDictionary\n")
        


# Here we decide on the proper dataset transformation
if DATASET_HP['imageRotationTuple'][0] != 0 and DATASET_HP['imageRotationTuple'][1] != 0:
    imagesDictionary = rotationTransform(imagesDictionary, 
                                         rotationRange=(DATASET_HP['imageRotationTuple'][0],
                                                        DATASET_HP['imageRotationTuple'][1]))
    
    
# This line plots a grid of random chosen lice from the dataset
beforeTransformFig = gridPlotLice(numCols=4, numRows=4, 
                                   plotDictionary=imagesDictionary, 
                                   showPlot=True)

# This line plots images from the same louse ID
beforeTransformSameFig = samePlotLice(numSame=2, 
                                        plotDictionary=imagesDictionary,
                                        showPlot=True)

if DATASET_HP['polarBool']:
    imagesDictionary = polarTransform(imagesDictionary)



# This line plots a grid of random chosen lice from the dataset
afterTransformFig = gridPlotLice(numCols=4, numRows=4, 
                                   plotDictionary=imagesDictionary, 
                                   showPlot=True)

# This line plots images from the same louse ID
afterTransformSameFig = samePlotLice(numSame=2, 
                                     plotDictionary=imagesDictionary,
                                     showPlot=False)

closePrompt = print("\nplt.close('all') to close figures\n")