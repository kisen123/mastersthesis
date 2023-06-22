# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:52:08 2022

@author: kisen


THIS PIPELINE USES SIAMESE NETWORKS FOR COMPARING DATAPOINTS
"""

# We first define the paramaters/hyperparameters of the entire system.

runIterator = 1

#%%
import subprocess

import time


from matplotlib import pyplot as plt

import os
# Necessary scripts for synthesizing images is in this folder
codeDir = "C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts"
os.chdir(codeDir)


# All necessary torch-imports
import torch
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float32)


# All necessary classic imports
import random

# Importing interactive plotting tools
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# Misc. imports
from datetime import datetime
from misc_scripts.saveExperiment import parametersToCSV, modelToTXT, dictParametersToTXT

# Imports of the dataset-generator functions, and the SiameseDataset class
from dataset_generation import datasetGenerator, datasetExtracter
from misc_scripts.SiameseDataset import SiameseDataset

# Imports of visualization scripts for showing what the images look like
from visualize_scripts.licePlots import gridPlotLice, samePlotLice

# Imports of dataset transformations in terms of rotation and polar transformation
from dataset_transform.rotationScripts import polarTransform

# Imports the splitting function for the image dictionaries
from dictionarySplit import datasetSplit


# Importing the training and evaluation functions
from misc_scripts.train_eval import trainSiamese, evalSiamese


# Importing the dataset hyperparameters
from hyperparameters import DATASET_HPs, MODEL_HPs, SYSTEM_HPs

# Import of a setSeed function to properly seed backpropagation passes, and other
# modules dependent on random number generators.
from setSeed import setSeed


# Importing the CyConv-replace function. This replaces all nn.Conv2d layers
# with CyConv
from models.replaceConvWithCyConv import replaceConvWithCyConv

# %% This section makes a dataset after the user's own liking
os.chdir(codeDir)

# For each experiment we do, we save those experiments in one directory.
# The directories under this experiment directory contain each time-stamped
# experiment with its corresponding run parameters.
individualExperimentFolder = "Experiment_no._" + str(runIterator)
runIterator += 1

# Collects today's date
date = datetime.now().strftime("%d-%m-%Y")
datePath = os.path.join(codeDir, 
                        "experimentFolder", 
                        date).replace("\\", "/")

# Makes the day's directory if it does not exist.
if os.path.exists(datePath) == False:
    os.mkdir(datePath)

experimentPath = os.path.join(datePath, 
                              individualExperimentFolder).replace("\\", "/")
if os.path.exists(experimentPath) == False:
    os.mkdir(experimentPath)


# Saves the dictionaries to readable .txt files.
dictParametersToTXT(SYSTEM_HPs.SYSTEM_HP_DICT, experimentPath, "systemHyperparameters")
dictParametersToTXT(MODEL_HPs.MODEL_HP_DICT, experimentPath, "modelHyperparameters")
dictParametersToTXT(DATASET_HPs.DATASET_HP_DICT, experimentPath, "datasetHyperparameters")


systemParameters = {}

experimentExplain = str(input("Would you like to add comments to the experiment (yes/no)?: "))

# We open a notepad for user-explanation if wanted
if experimentExplain == 'yes':
    
    with open(experimentPath + "/experiment_explanation.txt", 'w') as file:
        file.close()
        pass

    # Opens up the experiment's parameters in notepad
    subprocess.Popen(["notepad.exe", experimentPath + "/systemHyperparameters.txt"])
    subprocess.Popen(["notepad.exe", experimentPath + "/modelHyperparameters.txt"])
    subprocess.Popen(["notepad.exe", experimentPath + "/datasetHyperparameters.txt"])
    
    # Lets the user write an explanation
    notepad = subprocess.Popen(["notepad.exe", experimentPath + "/experiment_explanation.txt"])
    notepad.wait()
    print("File saved!")

#%% 


realDataFlag = False
runSavedModels = False

modelsDir = "C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts/results_folder/SyntheticDatasetResultsCopy/models"


valPerformanceList = []
trainPerformanceList = []

for SYSTEM_HP in SYSTEM_HPs.SYSTEM_HP_LIST:
    
      
        
    for DATASET_HP in DATASET_HPs.DATASET_HP_LIST:
    
        # The seed is set to generate reproducible images
        setSeed(SYSTEM_HP['seed'])
        os.chdir(codeDir)
        
        numIDs = DATASET_HP['numIDs']
        minNumSamples = DATASET_HP['minNumSamples']
        maxNumSamples = DATASET_HP['maxNumSamples']
        
        numSamples = [random.randint(minNumSamples, maxNumSamples) for i in range(numIDs)]
        
        print("\n...Please wait, dataset generation in progress...\n")
        
        # This line makes synthetic images, and stores them in
        # codeDir + "/syntheticFolder/images" as .tif-images
        makeImages = False
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
                                       my_dpi=141, 
                                       numSpecklesInterval=(DATASET_HP['numSpecklesLow'], 
                                                            DATASET_HP['numSpecklesHigh']),
                                       blackInterval=(DATASET_HP['blackLow'], 
                                                      DATASET_HP['blackHigh']),
                                       blacknessInterval=(DATASET_HP['blacknessLow'], 
                                                          DATASET_HP['blacknessHigh']),
                                       digestiveInterval=(DATASET_HP['digestiveBlackLow'],
                                                          DATASET_HP['digestiveBlackHigh']))
                                       # imageRotationTuple=DATASET_HP['imageRotationTuple'])

        # This line reads all the .tif-files, and writes the images to a .pkl-file.
        # Additionally, the function can return the dictionary that is written to the
        # .pkl-file.
        imagesDictionary = datasetExtracter.makePKLFile(realDataFlag=realDataFlag, resizeTuple=(127, 127))
        print("\nImage generation done, they now exist in the variable: imagesDictionary\n")
        
        
        # sampleIDSplit determines if we should split the dataset according to the
        # trainValTestRatio in terms of a percentage of IDs or number of samples in an ID.
        
        
        # Training options and datasets are set in this section
        # Device is set
        device = (torch.device('cuda') if torch.cuda.is_available()
                  else torch.device('cpu'))
        print(f"Training on device {device}.")
        
        
        for MODEL_HP in MODEL_HPs.MODEL_HP_LIST:
        
            # Seed is set for reproducibility in the Siamese Dataset collection.
            setSeed(SYSTEM_HP['seed'])
            
            lossFunction = MODEL_HP['lossFunction']
            batchSize = MODEL_HP['batchSize']
            percentOfPairs = MODEL_HP['percentOfPairs']
            
            
            # The real dataset requires a custom ID splitting algorithm, and
            # is explained by the real train/val/test split ratio in the thesis.
            if realDataFlag:
                sampleIDSplit = "IDCustom"
            else:
                sampleIDSplit = "ID"
                
                
            # This section sets up the data in train/val/test/unknown datasets
            # We can also determine if we want to standardize them.
            trainDictionary, \
                valDictionary, \
                testDictionary, \
                unknownDictionary, \
                allDictionary = datasetSplit(imagesDictionary,
                                             trainValTestRatio=DATASET_HP['trainValTestRatio'],
                                             datasetPlot=False, 
                                             standardizeBool=DATASET_HP['standardizeBool'],
                                             sampleIDSplit="ID",
                                             lossFunction=lossFunction,
                                             device='cpu',
                                             realDataFlag=realDataFlag)
            
            # NOTE, device='cuda' here is notably slower when augmenting data.

            
            
            
            # This line plots a grid of random chosen lice from the dataset
            beforeTransformFig = gridPlotLice(numCols=4, numRows=4, 
                                               plotDictionary=trainDictionary,
                                               showPlot=False)
            
            # This line plots images from the same louse ID
            beforeTransformSameFig = samePlotLice(numSame=2, 
                                                    plotDictionary=trainDictionary,
                                                    showPlot=False)
            
            
            
            
            # The datasets are set, along with the DataLoaders.
            trainDataset = SiameseDataset(device, trainDictionary,
                                          "train", mode="training",
                                          lossFunction=lossFunction,
                                          percentOfPairs=percentOfPairs,
                                          imageRotationTuple=DATASET_HP['imageRotationTuple'],
                                          polarBool=DATASET_HP['polarBool'],
                                          blurMaxSize=DATASET_HP['blurMaxSize'],
                                          unknownDictionary=unknownDictionary)
            
            trainPerformDataset = SiameseDataset(device, trainDictionary,
                                                 "trainPerform", mode="evaluation",
                                                 lossFunction=lossFunction,
                                                 percentOfPairs=percentOfPairs,
                                                 imageRotationTuple=DATASET_HP['imageRotationTuple'],
                                                 polarBool=DATASET_HP['polarBool'],
                                                 blurMaxSize=DATASET_HP['blurMaxSize'])
            
            valDataset = SiameseDataset(device, valDictionary,
                                        "val", mode="evaluation",
                                        lossFunction=lossFunction,
                                        percentOfPairs=percentOfPairs,
                                        imageRotationTuple=DATASET_HP['imageRotationTuple'],
                                        polarBool=DATASET_HP['polarBool'],
                                        blurMaxSize=DATASET_HP['blurMaxSize'])
            
            allDataset = SiameseDataset(device, allDictionary,
                                        "val", mode="training",
                                        lossFunction="Triplet",
                                        imageRotationTuple=DATASET_HP['imageRotationTuple'],
                                        polarBool=DATASET_HP['polarBool'],
                                        blurMaxSize=DATASET_HP['blurMaxSize'])
            

            # This line plots a grid of random chosen lice from the dataset
            afterTransformFig = gridPlotLice(numCols=4, numRows=4, 
                                               plotDataset=trainDataset,
                                               showPlot=False)
            
            # This line plots images from the same louse ID
            afterTransformSameFig = samePlotLice(numSame=2, 
                                                 plotDataset=trainDataset,
                                                 showPlot=False)
            
            currTime = datetime.now().strftime("%H.%M.%S")
            timePath = os.path.join(experimentPath, currTime).replace("\\", "/")
            os.mkdir(timePath)
            
            # We also save how the lice look like before and after transformation
            beforeTransformFig.savefig(fname=timePath + "/beforeTransformLice.png")
            beforeTransformSameFig.savefig(fname=timePath + "/beforeTransformSameLice.png")
            afterTransformFig.savefig(fname=timePath + "/afterTransformLice.png")
            afterTransformSameFig.savefig(fname=timePath + "/afterTransformSameLice.png")
            plt.close('all')
            
            
            
            # A train performance loader is set to evaluate the model in between
            # parameter updates and epochs.
            trainLoader = DataLoader(trainDataset, 
                                     batch_size=batchSize, 
                                     shuffle=True)
            valLoader = DataLoader(valDataset, 
                                   batch_size=batchSize,
                                   shuffle=False)
            trainPerformLoader = DataLoader(trainPerformDataset,
                                            batch_size=1,
                                            shuffle=False)
            allLoader = DataLoader(allDataset,
                                   batch_size=1,
                                   shuffle=False)
            
            modelName = MODEL_HP['model'].__name__
            
            if MODEL_HP['CylindricalModel']:
                modelName = "Cy-" + modelName
                
            if DATASET_HP['polarBool']:
                modelName = modelName + "-p"
            
            if runSavedModels:
                
                
                # By choice, the model that gives the best F1-score is chosen 
                # to be the pretraining model. Often, the model giving the best
                # F1-score and TAR are the same, the only difference is their
                # thresholds.
                modelPath = modelsDir + "/" + modelName + "/best_F1-score_model.pt"
                model = torch.load(modelPath).to(device)
                
                
                
            else:
                # Allocating model to device
                model = MODEL_HP['model']().to(device)
                
                if MODEL_HP['CylindricalModel'] == True:
                    model = replaceConvWithCyConv(model).to(device)
            print(modelName)
            
            # The optimizer is defined
            optimizer = MODEL_HP['optimizer'](model.parameters(), lr=MODEL_HP['learningRate'])
            
            
            
            def saveExperimentInfo(modelTrainTuple):
                
                # We select the best model what the user wants. By design,
                # the author uses TAR.
                bestModels = modelTrainTuple[3]
                bestModelPerformances = modelTrainTuple[4]
                
                
                
                
                
                
                for performanceMeasure, bestModel in zip(list(bestModels.keys()), list(bestModels.values())):
                    torch.save(bestModel, timePath + "/best_" + str(performanceMeasure) + "_model.pt")
                
                
                
                thresholdsVal = modelTrainTuple[6]
                thresholdsTrain = modelTrainTuple[7]
                
                
                for i, (performanceMeasure, performanceStats) in enumerate(bestModelPerformances.items()):
                    numEpochsBest = performanceStats[0]
                    print("The model with the highest {} performed: ({:.4f}, {:.4f}), and required {} epochs".format(performanceMeasure,
                                                                                                           performanceStats[1],
                                                                                                           torch.std(modelTrainTuple[0], dim=2)[i, int(numEpochsBest)-1],
                                                                                                           numEpochsBest))
                    
                    simThresholdMean = torch.mean(thresholdsVal, dim=2)[i, int(numEpochsBest) - 1]
                    simThresholdSTD  = torch.std( thresholdsVal, dim=2)[i, int(numEpochsBest) - 1]
                    print("The mean threshold was {:.4f} with std {:.4f}".format(simThresholdMean,
                                                                                 simThresholdSTD))
                    
                    
                
                # Creating the plots for saving
                lossPerformanceFig = modelTrainTuple[5]
                lossPerformanceFig.savefig(fname=os.path.join(timePath + "/loss&performance.png"))
                
                PCATransform = True
                
                
                _, PCAInfoFigTrain, PCAScatterFigTrain, tSNEScatterFigTrain, AUCPlotFigTrain = \
                    evalSiamese(bestModels["F1-score"], trainPerformLoader, device,
                                "F1-score", embeddingVisualization=True,
                                dataFlag="Training data",
                                numClassesPlot=int(DATASET_HP['numIDs'] * \
                                                   DATASET_HP['trainValTestRatio'][0]),
                                lossFlag="BCELoss", distMetric="euclidean",
                                modelType="Veration", PCATransform=PCATransform)
                        
                        
                        
                
                _, PCAInfoFigVal, PCAScatterFigVal, tSNEScatterFigVal, AUCPlotFigVal = \
                    evalSiamese(bestModels["F1-score"], valLoader, device,
                                "F1-score", embeddingVisualization=True,
                                dataFlag="Validation data",
                                numClassesPlot=int(DATASET_HP['numIDs'] * \
                                                   DATASET_HP['trainValTestRatio'][1]),
                                    lossFlag="Triplet", distMetric="euclidean",
                                modelType="Verifation", PCATransform=PCATransform)
                
                
                # Saving the plots
                PCAInfoFigTrain.savefig(fname=timePath + "/PCAInfoTrain.png")
                PCAScatterFigTrain.savefig(fname=timePath + "/PCAScatterEmbeddingSpaceTrain.png")
                tSNEScatterFigTrain.savefig(fname=timePath + "/tSNEScatterEmbeddingSpaceTrain.png")
                AUCPlotFigTrain.savefig(fname=timePath + "/AUCPlotTrain.png")
                
                PCAInfoFigVal.savefig(fname=timePath + "/PCAInfoVal.png")
                PCAScatterFigVal.savefig(fname=timePath + "/PCAScatterEmbeddingSpaceVal.png")
                tSNEScatterFigVal.savefig(fname=timePath + "/tSNEScatterEmbeddingSpaceVal.png")
                AUCPlotFigVal.savefig(fname=timePath + "/AUCPlotVal.png")
                
                
                
                
                plt.close('all')
                
                
                
                
                valPerformances = {pMeasure: valPerformances for \
                                   pMeasure, valPerformances in  \
                                   zip(performanceMeasures, modelTrainTuple[0])}
                torch.save(valPerformances, timePath + "/validationPerformancesModelRun.pt")
                
                trainPerformances = {pMeasure: trainPerformances for \
                                     pMeasure, trainPerformances in  \
                                     zip(performanceMeasures, modelTrainTuple[1])}
                torch.save(trainPerformances, timePath + "/trainPerformancesModelRun.pt")
                
                
                torch.save(thresholdsVal, timePath + "/thresholdsVal.pt")
                torch.save(thresholdsTrain, timePath + "/thresholdsTrain.pt")
                
                
                # Because the hyperparameter "epochs" could be variable depending
                # on if saveInterEpoch=True, modelsEpochs[2] is the training-run's
                # hyperparameters.
                RUN_HP = modelTrainTuple[2]
                systemID = "Run ID: " + date + " " + currTime
                systemParameters[systemID] = [DATASET_HP, RUN_HP, SYSTEM_HP]
                
                
                # We save the HP dictionaries and model options for logging experiments
                parametersToCSV(systemParameters[systemID], timePath)
                modelToTXT(model, timePath, 127, 127)
                
                
                
                
            
            # Running and training the models
            performanceMeasures = ["F1-score"]
            modelTrainTuple = trainSiamese(model, lossFunction, MODEL_HP,
                                           trainLoader, trainPerformLoader, valLoader,
                                           optimizer, device, saveInterEpoch=True, plotInter=True,
                                           performanceShadow=False, learningVisualizationEpoch=False,
                                           learningVisualizationStep=False, timePath=timePath,
                                           performanceMeasures=performanceMeasures)
                
            # Saving valuable information of the pipeline pass.
            saveExperimentInfo(modelTrainTuple)
            
            
#%%


# Change the second-to-last folder extension to get the results you want.
resultsDir = "C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts/results_folder/SyntheticDatasetResultsCopy/models"

modelsNames = ['Cy-Siamese_LeNet5_var-p',        #0
               'Cy-Siamese_MobileNetV3_var-p',   #1
               'Cy-Siamese_ResNet18_var-p',      #2
               'Siamese_LeNet5_var',             #3
               'Siamese_LeNet5_var-p',           #4
               'Siamese_MobileNetV3_var',        #5
               'Siamese_MobileNetV3_var-p',      #6
               'Siamese_ResNet18_var',           #7
               'Siamese_ResNet18_var-p']         #8

modelName = modelsNames[7]




polarBool = False
if modelName[-2:] == '-p':
    polarBool = True
    
    
    
    

modelDirPath = os.path.join(resultsDir, modelName).replace('\\', '/')

performanceMeasuresList = ["Accuracy", "Precision", "Recall", 
                           "F1-score", "TAR", "AUC"]
performanceMeasures = ["F1-score", "TAR"]


# If we pick F1-score and TAR, this list should be [3, 4]. Made to select out 
# which rows of the output the collect
performanceIdx = [performanceMeasuresList.index(performanceMeasure) for performanceMeasure in performanceMeasures]



savedModelF1  = torch.load(os.path.join(modelDirPath, "best_F1-score_model.pt").replace('\\', '/'))
savedModelTAR = torch.load(os.path.join(modelDirPath, "best_TAR_model.pt").replace('\\', '/'))


evalRuns = 100
validationPerformances = torch.tensor([])




imagesDictionary = datasetExtracter.makePKLFile(realDataFlag=False, resizeTuple=(127, 127))
# This section sets up the data in train/val/test/unknown datasets
# We can also determine if we want to standardize them.
trainDictionary, \
    valDictionary, \
    testDictionary, \
    unknownDictionary, \
    allDictionary = datasetSplit(imagesDictionary,
                                 trainValTestRatio=DATASET_HP['trainValTestRatio'],
                                 datasetPlot=False, 
                                 standardizeBool=DATASET_HP['standardizeBool'],
                                 sampleIDSplit="ID",
                                 lossFunction="Triplet",
                                 device='cpu',
                                 realDataFlag=False)



# This loop gradually rotates the validation dataset. The point here is to
# show when rotation becomes a problem for unseen lice
for rotationRange in range(0, 45, 3):
    
    print("Evaluating images rotated between {} and {}".format(-rotationRange, rotationRange))
    setSeed(123)
    
    valDataset = SiameseDataset(device, valDictionary,
                                "val", mode="evaluation",
                                lossFunction="Triplet",
                                percentOfPairs=1.0,
                                imageRotationTuple=(-rotationRange, rotationRange),
                                polarBool=polarBool,
                                blurMaxSize=9)
    
    valLoader = DataLoader(valDataset,
                           batch_size=1,
                           shuffle=False)
    
    valOutput = evalSiamese(savedModelF1,
                valLoader, 
                device,
                performanceMeasures,
                lossFlag="Triplet",
                distMetric="euclidean",
                evalRuns=evalRuns)
    
    
    
    validationPerformances = torch.cat( (validationPerformances, 
                                         valOutput[0][performanceIdx]), dim=1)
    
    
    

meanValPerformancesRotationsF1 = torch.mean(validationPerformances, dim=2)[0]
meanValPerformancesRotationsTAR = torch.mean(validationPerformances, dim=2)[1]

stdValPerformancesRotationsF1 = torch.std(validationPerformances, dim=2)[0]
stdValPerformancesRotationsTAR = torch.std(validationPerformances, dim=2)[1]



    
    









# validationPerformancesModelRun = torch.load(os.path.join(modelDirPath, "validationPerformancesModelRun.pt").replace('\\', '/'))









#%%
saveExperimentInfo(modelTrainTuple)
#%% This subsection plots up the training/validation performances of a 

# valPerformancePlot = [performance.item() for performance in valPerformanceList]
# trainPerformancePlot = [performance.item() for performance in trainPerformanceList]

xTicks = [str((0, 0))] + [str((angle, angle)) for angle in list(range(5, 14, 4))]

xTicks = ["(-5, 5)", "(-30, 30)", "(-60, 60)"]
#xTicks = [str((-angle, angle)) for angle in list(range(0, 46, 5))]
xAxis = torch.linspace(1, len(xTicks), len(xTicks))

plt.plot(xAxis, valPerformanceList, color=(1, 0.6, 0), marker='.')
plt.plot(xAxis, trainPerformanceList, c='#007AFF', marker='.')
plt.xticks(xAxis, xTicks)
plt.grid(True)
plt.xlabel("Rotation interval")
plt.ylabel("F1-scores")
plt.ylim(0.47, 1)
plt.title("Images are randomly rotated, and trained by rotation augmentation")
plt.legend(["Validation F1-score", "Training F1-score"])


plt.show()
 # %% This section does evaluation tests
# modelTest = modelsEpochs["NumEpochs: 50"][2]
modelTest = model
# %%
PCATransform = True
# %% TRAINING PERFORMANCE
trainPerformance = evalSiamese(modelTest, trainPerformLoader, device,
                               "F1-score", embeddingVisualization=True,
                               dataFlag="Training data",
                               numClassesPlot=30, lossFlag="BCELoss", distMetric="euclidean",
                               modelType="Veration", PCATransform=PCATransform,
                               datasetTransform="tSNE")
print(trainPerformance)
# %% VALIDATION PERFORMANCE
valPerformance = evalSiamese(modelTest, valLoader, device,
                             "F1-score", embeddingVisualization=True,
                             dataFlag="Validation data",
                             numClassesPlot=20, lossFlag="Triplet", distMetric="euclidean",
                             modelType="Verifation", PCATransform=PCATransform,
                             datasetTransform="tSNE")
print(valPerformance)

# %%
allPerformance = evalSiamese(modelTest, allLoader, device,
                             "F1-score", embeddingVisualization=True,
                             dataFlag="Validation data",
                             numClassesPlot=100, lossFlag="Triplet", distMetric="euclidean",
                             modelType="Verifation", PCATransform=PCATransform)
print(allPerformance)



#%%
# This section sets up the data in train/val/test/unknown datasets
# We can also determine if we want to standardize them.
trainDictionary, \
    valDictionary, \
    testDictionary, \
    unknownDictionary, \
    allDictionary = datasetSplit(imagesDictionary,
                                 trainValTestRatio=DATASET_HP['trainValTestRatio'],
                                 datasetPlot=False, 
                                 standardizeBool=DATASET_HP['standardizeBool'],
                                 sampleIDSplit="ID",
                                 lossFunction=lossFunction)
    
#%%
trainDataset = SiameseDataset(device, trainDictionary,
                              "train", mode="training",
                              lossFunction=lossFunction,
                              percentOfPairs=percentOfPairs,
                              imageRotationTuple=DATASET_HP['imageRotationTuple'],
                              polarBool=DATASET_HP['polarBool'],
                              blurMaxSize=DATASET_HP['blurMaxSize'],
                              unknownDictionary=unknownDictionary)

#%%
trainDataset.__getitem__(45)
#%%

model = MODEL_HP['model'].to(device)
#### SAFE COPY OF ALL IMPORTS ####
# # All necessary torch-imports
# import torch
# from torch import nn, optim, matmul
# from torch.utils.data import DataLoader
# import torchvision
# from torchvision import transforms
# from torchinfo import summary
# import torch.nn.functional as F
# from torch import Tensor
# from torch.nn.functional import relu
# from torch import sigmoid
# import torchvision.models as PyTorchModels
# torch.set_default_dtype(torch.float)


# # All necessary classic imports
# import numpy as np
# import math
# import random

# # All matplotlib imports
# import matplotlib.patches as patches
# from matplotlib.patches import Patch
# import matplotlib.ticker as mticker
# import matplotlib.cm as cm
# import matplotlib.colors as mcol
# from matplotlib import pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

# # Misc. imports
# from typing import Type
# import pickle
# from PIL import Image
# import cv2
# from datetime import datetime
# from misc_scripts.numSameDiff import numSameDiff
# from misc_scripts.scatter_plot_with_hover import scatter_plot_with_hover
# from misc_scripts.saveExperiment import parametersToCSV, modelToTXT, dictParametersToTXT
# import itertools
# import pandas as pd

# # Evaluation metric imports
# from sklearn.metrics import confusion_matrix


# # Feature engineering imports
# from feature_engineer_scripts import columnCollapseFunc

# # Imports of the dataset-generator functions, and the SiameseDataset class
# from dataset_generation import datasetGenerator, datasetExtracter
# from misc_scripts.SiameseDataset import SiameseDataset

# # Imports of PCA-scripts for visualization purposes
# from visualize_scripts.PCAScripts import PCALearn, PCATransformData

# # Imports of visualization scripts for showing what the images look like
# from visualize_scripts.licePlots import gridPlotLice, samePlotLice

# # Imports of dataset transformations in terms of rotation and polar transformation
# from dataset_transform.rotationScripts import rotationTransform, polarTransform

# # Imports the splitting function for the image dictionaries
# from dictionarySplit import datasetSplit

# # Importing different NN-models to be tested
# from similarity_models.model_classes import SiameseNetResnet18Similarity, ATTNet, OwnNet, SiameseMobileNet2, CyConvOwnNet

# # Importing loss functions and other methods appropriate for the loss functions
# from loss_module.lossFunctions import ContrastiveLoss, TripletLoss, distanceMetric
#%%
imagesDictionary = datasetExtracter.makePKLFile(realDataFlag=True, resizeTuple=(129, 129))

# # Importing the training and evaluation functions
# from misc_scripts.train_eval import trainSiamese, evalSiamese
#%%
len(trainDataset)


#%%
trainDictionary, \
    valDictionary, \
    testDictionary, \
    unknownDictionary, \
    allDictionary = datasetSplit(imagesDictionary,
                                 trainValTestRatio=DATASET_HP['trainValTestRatio'],
                                 datasetPlot=False, 
                                 standardizeBool=DATASET_HP['standardizeBool'],
                                 sampleIDSplit="sample")


# # Importing the dataset hyperparameters
# from hyperparameters import DATASET_HPs, MODEL_HPs, SYSTEM_HPs

# # Import of a setSeed function to properly seed backpropagation passes, and other
# # modules dependent on random number generators.
# from setSeed import setSeed