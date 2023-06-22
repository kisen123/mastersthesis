# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:11:01 2023

@author: kisen
"""

import random
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches





def gridPlotLice(numCols, numRows, plotDictionary=None, plotDataset=None, showPlot=False):
    
   
    # Defining the setup for the subplots
    numCols = numCols
    numRows = numRows

    if showPlot == False:
        plt.ioff()
    # Instantiating a plt figure
    fig, ax = plt.subplots(numCols, numRows)
    fig.set_size_inches(15, 7)
    fig.suptitle("Example lice")
    fig.tight_layout(pad=0.9)
    
    forIter = 0
    
    if plotDictionary != None:
        trainIDs = list(plotDictionary.keys())
    
    for col in range(0, numCols):
        for row in range(0, numRows):
            
            # Setting the seed for reproducible images
            random.seed(forIter)
            
            # We collect images, depending on if we are before or after 
            # dataset transformation.
            if plotDictionary != None:
                imageCollection = plotDictionary.get(trainIDs[forIter])
                
                # imgIdx sets which of the images in the collection of a louse to be 
                # shown.
                imgIdx = random.randrange(0, len(imageCollection))
                
                # Getting the image to plot
                imgToPlot = imageCollection[imgIdx]
                trainID = trainIDs[forIter]
                
            # The SiameseDataset-class does on-the-fly transformation, so
            # we call on the images with the __getitem__ method. We save both
            # memory and lines of code by directly collecting images from there.
            elif plotDataset != None:
                plotDataset.visualize = True
                tripletSample = plotDataset.__getitem__(forIter)
                
                imgToPlot, trainID = tripletSample[0][0], tripletSample[3]
                
            
            # Misc. plotting options.
            ax[row, col].imshow(imgToPlot.cpu(), cmap='gray', vmin=0, vmax=1)
            ax[row, col].axis("off")
            
            ax[row, col].set_title("Louse ID: "+str(trainID))
            forIter += 1
            
    if showPlot:
        fig.show()
    
    if plotDataset != None:
        plotDataset.visualize = False
    return fig
            

            
def samePlotLice(numSame=2, plotDictionary=None, plotDataset=None, showPlot=False):
    
    
    
   
    
    
    # CAVEAT! Since we are collecting processed images from the SiameseDataset-
    # class, we can ONLY plot 2 lice on the same plot. Hopefully this will not
    # be an issue.
    numSame = 2
    IDIteration = 0
    
    if plotDictionary != None:
        trainIDs = list(plotDictionary.keys())
        while IDIteration < len(trainIDs):
            
            # When the dataset encounters an ID with the required amount of images, 
            # we choose that louse to be the one shown.
            if len(plotDictionary.get(trainIDs[IDIteration])) >= numSame:
                break
            
            IDIteration += 1
    
    if showPlot == False:
        plt.ioff()
    # Instantiating a plt figure
    fig, ax = plt.subplots(1, numSame)
    fig.set_size_inches(15, 7)
    filterSize = (15,15)
    fig.suptitle(str(numSame) + " images of the same louse, \nshowcasing a kernel size: " + str(filterSize), size=20)
    fig.tight_layout(pad=3)
    
    forIter = 0
    
    if plotDictionary != None:
        louseID = list(plotDictionary.keys())[IDIteration]
        
    elif plotDataset != None:
        # From the __getitem__ method, this is anchor/positive.
        plotDataset.visualize = True
        tripletSample = plotDataset.__getitem__(0)
        louseIms, louseID = tripletSample[0:2], tripletSample[3]
        
    
    
    
    randomCenterX = np.random.randint(10, 50)
    randomCenterY = np.random.randint(10, 50)
    
    for col in range(0, numSame):
        
        # Getting the image to plot
        if plotDictionary != None:
            imgToPlot = plotDictionary[louseID][col]
            
        elif plotDataset != None:
            imgToPlot = louseIms[forIter][0]
        
        # Misc. plotting options.
        ax[col].imshow(imgToPlot.cpu(), cmap='gray', vmin=0, vmax=1)
        
        # Bigger blacker grid around the entire FOV for the filter
        ax[col].add_patch(patches.Rectangle((randomCenterX, randomCenterY), 
                                            filterSize[0], filterSize[1], 
                                            fill=None, alpha=1))
        
        # We show how the convolution filter of a given size will convolve with the
        # image.
        for rows in range(filterSize[0]):
            for cols in range(filterSize[1]):
                ax[col].add_patch(patches.Rectangle((randomCenterX + rows, randomCenterY + cols),
                                         1, 1, alpha=0.1, fill=None, edgecolor='k'))
        
        ax[col].axis("off")
        
        ax[col].set_title("Louse ID: "+str(louseID))
        
        
        
        forIter += 1
    
    if showPlot:
        fig.show()
    
    if plotDataset != None:
        plotDataset.visualize = False
    return fig
        
# DEPRECATED FUNCTION
# def singleLousePlot(plotDictionary, numPlots=3):
    
    
#     # This function is purely for plotting purposes, showing what 
#     # a random image from the dataset could look like. 
#     for _ in range(numPlots):
            
#         # Choosing a random image from the provided dictionary. 
#         randomID = random.choice(list(plotDictionary.keys()))
#         plotImg = random.choice(plotDictionary[randomID])
        
#         # Instantiating a plt figure
#         fig, ax = plt.subplots()
#         fig.set_size_inches(15, 7)
#         fig.suptitle("A random sample from the synthetic dataset", size=20)
        
#         ax.imshow(plotImg, cmap='gray', vmin=0, vmax=1)
#         ax.axis("off")