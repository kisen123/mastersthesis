# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:24:05 2022

@author: kisen
"""

import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import scipy.stats as stats
import pickle
import json

from matplotlib import pyplot as plt

#%% Section to change directory and set the dataset flag.
def makePKLFile(realDataFlag=False, resizeTuple=(64,64)):
    
    dataPath = os.getcwd()
    
    # Here we set the real data flag. If we deal with the real data, code has been
    # written to deal with the uneven dimensions and so on.
    paddingFlag = False
    
    
    
    if realDataFlag:

        realDataPath = os.path.join(dataPath, "realFolder")
        
        if os.path.exists(realDataPath) == False:
            os.mkdir(realDataPath)
        
        dataPath = realDataPath
    else:
        syntheticDataPath = os.path.join(os.getcwd(), "syntheticFolder")
        dataPath = syntheticDataPath
    
    
    
    # Get the current working directory
    cwd = os.getcwd()
    
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))
    
    # We change the directory to the image directory
    os.chdir(dataPath)
    
    # Print the current working directory
    print("Directory changed to: {0}".format(os.getcwd()))
    # This script will read in a directory
    
#   # %% This section sets up the data
    
    # Defining functions that are needed for PyTorch tensors.
    ConvertToTensor = transforms.ToTensor()
    resize = transforms.Resize(resizeTuple)
    
    # Making a dictionary to store the images and their pairs
    imagesDictionary = {}
    
    # Get the list of image directories
    dirList = os.listdir()
    dirList = [file for file in dirList if not file.endswith('.pkl')]
    
    for directory in dirList:
        
        print("Current directory: " + str(directory))
        
        # # Makes a new directory for padded images.
        if paddingFlag:
            padDirPath = os.path.join(dataPath, directory, 'padded').replace("\\", "/")
            if os.path.exists(padDirPath) == False:
                os.mkdir(padDirPath)
            
        # # Going through each folder of images
        os.chdir(os.path.join(dataPath, directory).replace("\\", "/"))
        
        # Getting the list of images in the cwd.
        # imagesList = os.listdir()
        
        
        entries = (os.scandir(os.getcwd()))
        entries = sorted(entries, key=lambda entry: (entry.stat().st_ctime, entry.name))
        entries = sorted(entries, key=lambda entry: entry.stat().st_ctime)
        
        imagesList = [entry.name for entry in entries]
        
        
        # Iterate over each image
        i = 1
        for image in imagesList:
            
            if realDataFlag:
                # If the first three digits are 0, then the louse is 
                # unidentified in a pair
                if image[0:3] == '000':
                    louseID = "UNKNOWN"
                    
                else:
                    louseID = image[0:9]
            
            else:
                louseID = image.split("-")[0]
                
            # For each newly encountered louseID, a tuple must be instantiated
            if imagesDictionary.get(louseID) == None:
                imagesDictionary[louseID] = ()
                
            # Opening and preprocessing the images
            liceImage = Image.open(os.path.join(dataPath, directory, image).replace("\\", "/"))
            liceImageGray = liceImage.convert("L")
            liceGray = ConvertToTensor(liceImageGray)
            
            # Checks if image dimensions coincide. 
            # If they do, we only need to resize.
            if liceGray.shape[1] == liceGray.shape[2]:
            
                processedImage = np.array(liceGray[0])
                
            else:
                
                
                
                # Extracting the image dimensions, and which dimension to pad in
                imDimMax, imDim = torch.max(torch.tensor([liceGray.shape[1],
                                                          liceGray.shape[2]]),
                                            dim=0)
                
                rowsColsToMatch = abs(liceGray.shape[1] - liceGray.shape[2])
                
                
                oneSide = int(rowsColsToMatch / 2)
                otherSide = oneSide
                
                # If-statement to take care of odd numbers
                if rowsColsToMatch%2 == 1:
                    oneSide = int(np.floor(rowsColsToMatch / 2))
                    otherSide = int(np.ceil(rowsColsToMatch / 2))
                    
                    
                    
                # imDim = 0 means that we need to pad in width
                if imDim == 0:
                    
                    processedImage = np.pad(liceGray[0], ((0, 0), (oneSide, otherSide)),
                                         'constant', constant_values=(0,0))
                
                else:
                    processedImage = np.pad(liceGray[0], ((oneSide, otherSide), (0,0)),
                                         'constant', constant_values=(0,0))
                
                
            processedImage = processedImage*255
            processedImage = processedImage.astype(np.uint8)
            
            # Converting and saving the array as a png-image
            im = Image.fromarray(processedImage)
            processedImage = resize(im)
            processedImage.convert("L")
            
            if paddingFlag:
                # Saving the padded image to a folder
                processedImage.save(padDirPath + '/' + image.rstrip('.tif') + '_padded.tif')
            
            
            # Saving the image with its ID to a tuple. UNKNOWNS is its own tuple,
            # and ID'd lice correspond to their own tuples.
            processedImage = np.asarray(processedImage).astype(np.float32) / 255
            imagesDictionary[louseID] += (processedImage, )
            
            # Unless the dataset is small, I'd advice this to be set False
            if False:
                plt.imshow(processedImage, cmap='gray')
                plt.show()
            
            
            i += 1
            
            
#    #%% Here we store the imagesDictionary to a file
    try:
        os.chdir(dataPath)
        
        # fileToSave = json.dumps(imagesDictionary)
        # with open("imagesDictionary.json", "w") as f:
        #     f.write(fileToSave)
        #     f.close()
        
        with open(directory + ".pkl", "wb") as f:
            pickle.dump(imagesDictionary, f)
            
        print("Done writing the .pkl file")
        return imagesDictionary
        
    except:
        print("Something went wrong during file dump")
    
            
#%% THE FOLLOWING CODE ATTEMPTS TO FILL IN THE LETTERBOXES WITH NOISE, NO LUCK YET.
# dataPath = "C:/Users/kisen/Documents/Masteroppgave/Bilder/Datasett/IR_pictures_Expt3_2020"

# # Get the current working directory
# cwd = os.getcwd()

# # Print the current working directory
# print("Current working directory: {0}".format(cwd))

# # We change the directory to the image directory
# os.chdir(dataPath)

# # Print the current working directory
# print("Directory changed to: {0}".format(os.getcwd()))
# # This script will read in a directory

# ConvertToTensor = transforms.ToTensor()

# directory = "Austevoll_2021.02.19_high"
# liceImage = Image.open(os.path.join(dataPath, directory, "K5F1.jpg").replace("\\", "/"))
# liceImageGray = liceImage.convert("L")

# liceGray = ConvertToTensor(liceImageGray)

# histogram, bin_edges = np.histogram(liceGray[0, 1600:2000, 100:1000], bins=256, range=(0, 1), density=True)

# # configure and draw the histogram figure
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("grayscale value")
# plt.ylabel("PDF-value")
# plt.xlim([0.0, 1.0])  # <- named arguments do not work here

# plt.plot(bin_edges[0:-1], histogram)  # <- or here



# # Here we calculate the Gaussian distribution model from the example noise image.

# gaussMean = bin_edges[np.argmax(histogram)]
# gaussSTD = np.sqrt(np.sum( ((histogram - gaussMean) ** 2) ) / (256))


# gaussSTD = np.sqrt(torch.sum((liceGray[0, 1600:2000, 100:1000].flatten() - gaussMean) ** 2) / (360000)).item()



# #%%
# # SKAL VÆRE RIKTIG MEN SER FEIL UT
# plt.plot(bin_edges, stats.norm.pdf(bin_edges, gaussMean, gaussSTD))
# plt.show()
# #%%
# # SER RIKTIG UT MEN HAR BRUTE FORCET MEG FREM

# # configure and draw the histogram figure
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("grayscale value")
# plt.ylabel("PDF-value")
# plt.xlim([0.0, 1.0])  # <- named arguments do not work here

# plt.plot(bin_edges, stats.norm.pdf(bin_edges, gaussMean, 0.0285))
# plt.show()

# #%%
# gaussMean = 0.69140625
# gaussSTD = 0.0285



# #%%

# a = [[1, 2], [3, 4]]

# print(np.pad(a, ((2, 5), (0, 0)), 'constant', constant_values=(4,6)))