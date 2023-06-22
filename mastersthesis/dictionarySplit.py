# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:15:40 2023

@author: kisen
"""

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import torch
import math
import random
import numpy as np
from torchvision import transforms

def datasetSplit(imagesDictionary, trainValTestRatio=(0.7, 0.3, 0.0), 
                 datasetPlot=True, standardizeBool=False, sampleIDSplit="sample",
                 lossFunction=None, device='cpu', realDataFlag=False):
    
    
    
    # We make a bar plot of the dataset, as explained by the figure in the Model
    # Evaluation part of the Master's thesis.
    imageCount = 0
    imageCountVector = []
    for ID in imagesDictionary:
        imageCount += len(imagesDictionary[ID])
        
        if ID == "UNKNOWN":
            # [imageCountVector.append(1) for _ in range(len(imagesDictionary[ID]))]
            imageCountVector.append(len(imagesDictionary[ID]))
        else:
            imageCountVector.append(len(imagesDictionary[ID]))
    
    
    
    
    if datasetPlot:
        
        bluePatch   = mpatches.Patch(color="royalblue", label="Training data")
        orangePatch = mpatches.Patch(color="orange", label="Validation data")
        blackPatch  = mpatches.Patch(color="black", label="Testing data")
        
        plt.figure(104)
        
        if realDataFlag:
            datasetTitle = "real"
            
            
            # TODO This way of defining yTicksList assumes UNKNOWNS to have the
            # most samples, which is not always true
            yTicksList = list(range(min(imageCountVector), max(sorted(imageCountVector)[:-1]) + 1))
            # The last element is cut out as it is the number of unknonws, 
            # which will make the plot look ugly.
            
            colorsList = ["orange" if imageCount==2 else "royalblue" for imageCount in sorted(imageCountVector)[:-1]]
            plt.bar(range(len(imageCountVector)-1), sorted(imageCountVector)[:-1], color=colorsList)
            
            
            
            
            
        else:
            datasetTitle = "synthetic"
            yTicksList = list(range(min(imageCountVector), max(imageCountVector) + 1))
            colorsList = ["royalblue"] * int(trainValTestRatio[0] * len(imageCountVector)) + \
                         ["orange"]    * int(trainValTestRatio[1] * len(imageCountVector)) + \
                         ["black"]     * int(trainValTestRatio[2] * len(imageCountVector))
                         
            plt.bar(range(len(imageCountVector)), sorted(imageCountVector), color=colorsList)
        
        plt.yticks(yTicksList)
        plt.xlabel('ID-number')
        plt.ylabel('Number of samples to an ID')
        plt.title('The ' + datasetTitle +  ' dataset distribution', fontweight='bold')
        plt.grid(True, axis='y')
        plt.show()
        plt.legend(handles=[bluePatch, orangePatch, blackPatch])
    
    
    # Making standard variables and checking image shapes.
    exampleImageKey = list(imagesDictionary.keys())[0]
    shapeImage = imagesDictionary[exampleImageKey][0].shape
    shapeDataTensor = shapeImage + (int(imageCount), )
    dataTensor = torch.zeros(shapeDataTensor)
    
    
    
    # Checks if the dataset contains lice images of an unknown ID.
    if imagesDictionary.get("UNKNOWN") == None:
        numUnknowns = 0
    else:
        numUnknowns = len(imagesDictionary["UNKNOWN"])
    
    
    
    # Instantiating variables that change in the for loop.
    imageCount = 0
    numCollections = 0
    
    
    # This little loop figures out how many collections there are of unique lice.
    # This way we can set up the training, validation and test sets.
    for ID in imagesDictionary:
        
        sampleImages = imagesDictionary[ID]
        
        if ID != "UNKNOWN" and len(sampleImages) > 1:
            numCollections += 1
    
    
    # Figuring out how many training, validation and test collections correspond to
    # the given percentages
    
    print("Requested train/validation/test ratio is: " + str(trainValTestRatio))
    
    fracTrain, numTrains = math.modf(trainValTestRatio[0]*numCollections)
    fracVal, numVals = math.modf(trainValTestRatio[1]*numCollections)
    fracTest, numTests = math.modf(trainValTestRatio[2]*numCollections)
    
    # Processing information
    fracSum = fracTrain + fracVal + fracTest
    numVals, numTests = int(numVals), int(numTests)
    
    # The uncertainty of decimals in the splitting ratio is ruled 
    # in favor of adding to the training set.
    if fracSum > 0:
        numTrains = int(numTrains + round(fracSum))
    else:
        numTrains = int(numTrains)
        
    
    allDictionary = {}
    datasetDictionary = {}
    trainDictionary = {}
    valDictionary = {}
    testDictionary = {}
    unknownDictionary = {}
        
    datasets = ["trainSet", "valSet", "testSet", "UNKNOWN"]
    labels = ["trainLabels", "valLabels", "testLabels", "UNKNOWN_Labels"]
    
    
    # Instantiating the dictionary tuples.
    for dataset in datasets:
    
        if datasetDictionary.get(dataset) == None:
            datasetDictionary[dataset] = ()
            
    for label in labels:
        
        if datasetDictionary.get(label) == None:
            datasetDictionary[label] = ()
    
    listKeys = list(imagesDictionary.keys())

    trainValTestCount = 1
    
    
    shuffledKeys = listKeys.copy()
    random.shuffle(shuffledKeys)
    indexVector = [listKeys.index(x) for x in shuffledKeys]
    
    
    
    
    # Sorting the image count vector to match it with the shuffledKeys
    sortedList = [imageCountVector[index] for index in indexVector]
    
    
    # Sorting the IDs to be l
    if sampleIDSplit == "IDCustom":
        
        
        # We split the dictionaries into dictionaries of labeled and un-labeled
        # images.
        labeledDictionary = {ID: samples for ID, samples in imagesDictionary.items() if ID != "UNKNOWN"}
        unknownDictionary["UNKNOWN"] = torch.tensor(np.asarray(imagesDictionary["UNKNOWN"]))
        
        # Get the potential number of samples for IDs.
        numSamplesLabeled = list(set([len(list(labeledDictionary.values())[i]) \
                                       for i in range(len(labeledDictionary.keys()))]))[::-1]
        
        # We want the order to be ....., 1, 2, as we save the 2 samples IDs
        # for evaluation.
        if numSamplesLabeled[-1] == 1:
            numSamplesLabeled[-1], numSamplesLabeled[-2] = numSamplesLabeled[-2], numSamplesLabeled[-1]
            
        sortingOrder = {length: order for length, order in zip(numSamplesLabeled, list(range(len(numSamplesLabeled))))}
            
        
        # Sorting the labeled dictionary by number of samples ....., 1, 2.
        labeledDictionary = dict(sorted(labeledDictionary.items(), key=lambda x: sortingOrder[len(x[1])]))
        numSamplesID = [len(labeledDictionary[key]) for key in labeledDictionary.keys()]
        
        labeledKeys = labeledDictionary.keys()
        
        
        
        # TODO Construct code to let the user decide on a train/val/test
        # ratio for later datasets. This custom if-statement to split data
        # was hard-coded for the provided dataset.
        numTrains = 55
        numVals = 20
        numTests = 0
        # numTrains = TODO
        # numVals = TODO
        # numTests = TODO
        
        
        # NOTE! We must NOT shuffle the keys here, as they are constructed
        # to be descending as the previous code explains.
        for ID in labeledKeys:
            
            # The images corresponding to one ID is iterated over.
            sampleImages = np.asarray(imagesDictionary[ID])
            allDictionary[ID] = torch.tensor([sampleImages][0])
            
            
            # Part of the TODO statement regarding train/val/test splits.
            if len(sampleImages) != 2:
                trainDictionary[ID] = torch.tensor([sampleImages][0])
            else:
            # Part of the TODO statement regarding train/val/test splits.
                
                
                if trainValTestCount > numTrains and trainValTestCount <= \
                    numTrains + numVals:
                        
                    valDictionary[ID] = torch.tensor([sampleImages][0])
                
                if trainValTestCount > numTrains + numVals and trainValTestCount <=\
                    numTrains + numVals + numTests:
                
                    testDictionary[ID] = torch.tensor([sampleImages][0])
                    
                
            trainValTestCount += 1
                
                
    
    elif sampleIDSplit == "ID":
        
        # CODE FOR ZERO-SHOT LEARNING AND EVALUATION DATASETS
        for ID in shuffledKeys:
            
            # The images corresponding to one ID is iterated over.
            sampleImages = np.asarray(imagesDictionary[ID])
        
            allDictionary[ID] = torch.tensor([sampleImages][0])
            # If-statement to single out collections of lice IDs
            if ID != "UNKNOWN" and len(sampleImages) > 1:
                
                
                # When the number of train/val/testpoints, the code moves on to fill the
                # other dictionaries.
                if trainValTestCount <= numTrains:
                    
                    trainDictionary[ID] = torch.tensor([sampleImages][0])
                    
                if trainValTestCount > numTrains and trainValTestCount <=\
                    numTrains + numVals:
                    
                    valDictionary[ID] = torch.tensor([sampleImages][0])
                    
                if trainValTestCount > numTrains + numVals and trainValTestCount <=\
                    numTrains + numVals + numTests:
                
                    testDictionary[ID] = torch.tensor([sampleImages][0])
                
                trainValTestCount += 1
                
            else:
                
                unknownDictionary[ID] = torch.tensor([sampleImages][0])
                
        print("The actual train/validation/test ratio is: " + \
              str((round(numTrains/numCollections, 4),
                   round(numVals/numCollections, 4),
                   round(numTests/numCollections, 4))))
                
        print("The number of IDs in the train-set:         ", str(len(trainDictionary)))
        print("The number of IDs in the validation-set:    ", str(len(valDictionary)))
        print("The number of IDs in the test-set:          ", str(len(testDictionary)))
        print("The number of IDs that are unknown:         ", str(len(unknownDictionary)))
        
        
    
    
    
    elif sampleIDSplit == "IDCustom2":
        
        
        
        imageCountVector = [2, 2, 1, 1, 3, 1, 3, 1, 3, 2, 1, 1, 1, 1, 1,
                            3, 3, 1, 1, 1, 2, 2, 3, 4, 3, 1, 3, 1, 3, 3,
                            3, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 3,
                            1, 2, 1, 1, 2, 2, 3, 2, 2, 2, 3, 1, 3, 1, 1,
                            1, 1, 1, 3, 2, 2, 1, 3, 2, 2, 3, 2, 1, 1, 1]
        
        
        # We split the dictionaries into dictionaries of labeled and un-labeled
        # images.
        labeledDictionary = {ID: samples[:imageCount] for (ID, samples), imageCount in zip(imagesDictionary.items(), imageCountVector) if ID != "UNKNOWN"}
        
        # Get the potential number of samples for IDs.
        numSamplesLabeled = list(set([len(list(labeledDictionary.values())[i]) \
                                       for i in range(len(labeledDictionary.keys()))]))[::-1]
        
        # We want the order to be ....., 1, 2, as we save the 2 samples IDs
        # for evaluation.
        if numSamplesLabeled[-1] == 1:
            numSamplesLabeled[-1], numSamplesLabeled[-2] = numSamplesLabeled[-2], numSamplesLabeled[-1]
            
        sortingOrder = {length: order for length, order in zip(numSamplesLabeled, list(range(len(numSamplesLabeled))))}
            
        
        # Sorting the labeled dictionary by number of samples ....., 1, 2.
        labeledDictionary = dict(sorted(labeledDictionary.items(), key=lambda x: sortingOrder[len(x[1])]))
        numSamplesID = [len(labeledDictionary[key]) for key in labeledDictionary.keys()]
        
        labeledKeys = labeledDictionary.keys()
        
        
        
        # TODO Construct code to let the user decide on a train/val/test
        # ratio for later datasets. This custom if-statement to split data
        # was hard-coded for the provided dataset.
        numTrains = 55
        numVals = 20
        numTests = 0
        # numTrains = TODO
        # numVals = TODO
        # numTests = TODO
        
        
        # NOTE! We must NOT shuffle the keys here, as they are constructed
        # to be descending as the previous code explains.
        for ID in labeledKeys:
            
            # The images corresponding to one ID is iterated over.
            sampleImages = np.asarray(labeledDictionary[ID])
            allDictionary[ID] = torch.tensor([sampleImages][0])
            
            
            # Part of the TODO statement regarding train/val/test splits.
            if len(sampleImages) != 2:
                trainDictionary[ID] = torch.tensor([sampleImages][0])
            else:
            # Part of the TODO statement regarding train/val/test splits.
                
                
                if trainValTestCount > numTrains and trainValTestCount <= \
                    numTrains + numVals:
                        
                    valDictionary[ID] = torch.tensor([sampleImages][0])
                
                if trainValTestCount > numTrains + numVals and trainValTestCount <=\
                    numTrains + numVals + numTests:
                
                    testDictionary[ID] = torch.tensor([sampleImages][0])
                    
                
            trainValTestCount += 1
    
    
    
    
    
    
    
    
    elif sampleIDSplit == "sample":
        # CODE FOR FEW-SHOT LEARNING WHERE EACH ID HAS THE SAME AMOUNT OF IMAGES
        for i, ID in enumerate(shuffledKeys):
            
            # The images corresponding to one ID is iterated over.
            sampleImages = np.asarray(imagesDictionary[ID])
        
            allDictionary[ID] = torch.tensor([sampleImages][0])
            # If-statement to single out collections of lice IDs
            if ID != "UNKNOWN" and len(sampleImages) > 1:
                
                # Because IDs can have varying amounts of samples, we use
                # the train/test split ratios to determine how many samples
                # per ID should be allocated to which dataset dictionary.
                fracTrain, numTrains = math.modf(trainValTestRatio[0]*sortedList[i])
                fracVal, numVals = math.modf(trainValTestRatio[1]*sortedList[i])
                fracTest, numTests = math.modf(trainValTestRatio[2]*sortedList[i])
                
                # Processing information
                fracSum = fracTrain + fracVal + fracTest
                numVals, numTests = int(numVals), int(numTests)
                
                # For the special case of triplet loss, we need to hard code
                # the instance where an ID has only 2 samples.
                if sortedList[i] == 2 and (lossFunction == "Triplet" or lossFunction == "TripletTorch"):
                    numTrains = 2
                    numVals = 0
                    numTests = 0
                else:
                    # The uncertainty of decimals in the splitting ratio is ruled 
                    # in favor of adding to the training set.
                    if fracSum > 0:
                        numTrains = int(numTrains + round(fracSum))
                    else:
                        numTrains = int(numTrains)
                    
                    
                # Defining how many samples per ID to allocate to each dataset
                # as per the trainValTestRatio.
                trainIdx = numTrains
                valIdx = numTrains + numVals
                testIdx = numTrains + numVals + numTests
                
                trainDictionary[ID] = torch.tensor([sampleImages][0])[:trainIdx, :, :]
                valDictionary[ID] = torch.tensor([sampleImages][0])[trainIdx:valIdx, :, :]
                testDictionary[ID] = torch.tensor([sampleImages][0])[valIdx:testIdx, :, :]
                
                
            else:
                
                unknownDictionary[ID] = torch.tensor([sampleImages][0])
    
    
    
    
    # for ID in listKeys:
        
    #     # The images corresponding to one ID is iterated over.
    #     sampleImages = np.asarray(imagesDictionary[ID])
    
            
    #     # If-statement to single out collections of lice IDs
        
            
    #     trainDictionary[ID] = torch.tensor([sampleImages][0])[:2, :, :]
    #     valDictionary[ID] = torch.tensor([sampleImages][0])[2:3, :, :]
    #     testDictionary[ID] = torch.tensor([sampleImages][0])[3:4, :, :]
    #         # # When the number of train/val/testpoints, the code moves on to fill the
    #         # # other dictionaries.
    #         # if trainValTestCount <= numTrains:
                
    #         #     trainDictionary[ID] = torch.tensor([sampleImages][0])
                
    #         # if trainValTestCount > numTrains and trainValTestCount <=\
    #         #     numTrains + numVals:
                
    #         #     valDictionary[ID] = torch.tensor([sampleImages][0])
                
    #         # if trainValTestCount > numTrains + numVals and trainValTestCount <=\
    #         #     numTrains + numVals + numTests:
            
    #         #     testDictionary[ID] = torch.tensor([sampleImages][0])
            
    #         # trainValTestCount += 1
            
            
    
    
    
    """
    Now the data exists like this in a training/val/test dictionary:
    trainingData = {
        'label_1': [image1, image2, image3],
        'label_2': [image4, image5],
        'label_3': [image6],
        ...
    }
    """
    
    # If we standardize the images, the standardized images will be the output
    # of the function.
    if standardizeBool:
        trainingData = torch.empty(0, shapeImage[0], shapeImage[1])
        for label, imageTuple in trainDictionary.items():
            trainingData = torch.cat( (trainingData, imageTuple), axis=0 )
            
        # Here we add the unknowns to the mean/std calculation, as they are part
        # of the training set.
        # TODO, NEEDS TO BE ADJUSTED FOR CONTRASTIVE LOSS.
        if realDataFlag:
            
            unknownsData = unknownDictionary["UNKNOWN"]
            trainingData = torch.cat( (trainingData, unknownsData), axis=0 )
                
        # Calculating necessary values for standardization
        mean = torch.mean(trainingData, dim=[0,1,2])

        sumOfDeviations = torch.sum((trainingData - mean) ** 2)
        std = np.sqrt( (sumOfDeviations) / (trainingData.size(0) * trainingData.size(1) * \
                                            trainingData.size(2)))

        # Instancing a transformation
        transform = transforms.Normalize(mean, std)

        
        # Instantiating the data dictionaries
        trainStandDictionary = {}
        valStandDictionary = {}
        testStandDictionary = {}
        unknownStandDictionary = {}
        allStandDictionary = {}

        standDictionaries = [trainStandDictionary, valStandDictionary,
                             testStandDictionary, unknownStandDictionary,
                             allStandDictionary]

        dataDictionaries = [trainDictionary, valDictionary,
                            testDictionary, unknownDictionary, allDictionary]
        
        # Filling the dictionaries with standardized images
        for i, dataDictionary in enumerate(dataDictionaries):
            for label, imageTuple in dataDictionary.items():
                standDictionaries[i][label] = transform(imageTuple)
                
                
        trainDictionary = trainStandDictionary
        valDictionary = valStandDictionary
        testDictionary = testStandDictionary
        unknownDictionary = unknownStandDictionary
        allDictionary = allStandDictionary
        
    # Writing the dictionaries to the proper device
    trainDictionary   = {key: value.to(device) for key, value in trainDictionary.items()}
    valDictionary     = {key: value.to(device) for key, value in valDictionary.items()}
    testDictionary    = {key: value.to(device) for key, value in testDictionary.items()}
    unknownDictionary = {key: value.to(device) for key, value in unknownDictionary.items()}
    allDictionary     = {key: value.to(device) for key, value in allDictionary.items()}
    
    return trainDictionary, valDictionary, testDictionary, unknownDictionary, allDictionary