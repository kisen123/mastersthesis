# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:00:23 2023

@author: kisen
"""
# All necessary torch-imports
import torch
from torch import nn
torch.set_default_dtype(torch.float)


# All necessary classic imports
import numpy as np
import random
import math

# All matplotlib imports

from matplotlib import pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# Misc. imports
import os
from datetime import datetime
from misc_scripts.numSameDiff import numSameDiff
from misc_scripts.scatter_plot_with_hover import scatter_plot_with_hover
from misc_scripts.lossPlot import lossPlot
from misc_scripts.tripletMask import getSemihardTripletMask
from misc_scripts.eval_loss_func import eval_loss_func
import time

# Evaluation metric imports
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity as cosSim
from sklearn.metrics import pairwise_distances as distSim
from sklearn.metrics import auc


# Necessary scripts for synthesizing images is in this folder
codeDir = "C:/Users/kisen/Documents/Masteroppgave/Kode/liceScripts"
os.chdir(codeDir)

# Imports of PCA-scripts for visualization purposes
from visualize_scripts.PCAScripts import PCALearn, PCATransformData

# Import of PCA visualization script
from visualize_scripts.evaluationPlot import evaluationPlot


# Importing loss functions and other methods appropriate for the loss functions
from loss_module.lossFunctions import ContrastiveLoss, TripletLoss, distanceMetric






def trainSiamese(model, lossFlag, hyperParameters, trainLoader,
                 trainPerformLoader, valLoader, optimizer, device,
                 saveInterEpoch=True, plotInter=False, performanceShadow=False,
                 learningVisualizationEpoch=False, learningVisualizationStep=False,
                 timePath=None, performanceMeasures=["F1-score", "TAR"], FARThreshold=10e-3):
    
    savePATH = timePath + "/savedRunEpoch"
    os.mkdir(savePATH)
    try:
        runParameters = hyperParameters.copy()
        
        # Instantiating the proper loss function.
        if lossFlag == "Contrastive":
            lossFunc = ContrastiveLoss()
    
        elif lossFlag == "Triplet":
            lossFunc = TripletLoss(m=0.2)
    
        elif lossFlag == "TripletTorch":
            lossFunc = nn.TripletMarginLoss(margin=0.2, p=2)
    
        elif lossFlag == "BCELoss":
            lossFunc = nn.BCELoss()
    
        # Defining a couple of standard variables and functions
        numBatches = len(trainLoader)
        batchSize = trainLoader.batch_size
        epochs = runParameters["epochs"]
        modelsDictionary = {}
    
    
        interThresholdsVal = torch.tensor([])
        interThresholdsTrain = torch.tensor([])
        



    
    
        # We make the epoch plotting list
        epochInterval = 1
    
        epochsXAxis = list(range(1, epochs, epochInterval)) + [epochs]
    
        # Letting the model know that it is in train-mode
        model.train()
    
        # Zeroing out potential previous gradients.
        optimizer.zero_grad(set_to_none=True)
    
    
        # The pipeline's performance measures
        performanceMeasuresList = ["Accuracy", "Precision", "Recall", "F1-score",
                                   "TAR", "AUC"]
        performanceIdx = [performanceMeasuresList.index(performanceMeasure) for performanceMeasure in performanceMeasures]
        bestPerformances = {key: None for key in performanceMeasures}
        
        # User prompt to decide performance measure.
    
        trainDataset = trainLoader.dataset.dataset
        valDataset = valLoader.dataset.dataset
    
        # Calculating the cardinality of the Siamese datasets.
        trainCardinality = numSameDiff(trainDataset)
        valCardinality = numSameDiff(valDataset)
        
        datasetCardinality = trainCardinality, valCardinality
        
    
        print("\nPlease wait... the model is now being trained...\n")
        epochLossHistory = torch.tensor([])
    
        # The max-min confidence intervals are stored in the loop.
        maxTrains = torch.tensor([])
        minTrains = torch.tensor([])
    
        maxVals = torch.tensor([])
        minVals = torch.tensor([])
    
        
        
        initialize = True
        firstPoint = True
        
        iterationMiniBatch = 0
        
        plotIterator = 0
        
        
        plotXVector = torch.tensor([])
        plotTrainLossVector = torch.tensor([])
        plotValLossVector   = torch.tensor([])
        performanceXVector = torch.tensor([])
        trainingLoss = torch.tensor([])
        interValPerformances = torch.tensor([])
        interTrainPerformances = torch.tensor([])
        
        plotTrainPerformance = torch.tensor([])
        plotValPerformance = torch.tensor([])
        
    
        lossPerformanceFig, ax1, ax2 = lossPlot(lossFlag=lossFlag, 
                                                batchSize=batchSize,
                                                performanceMeasures=performanceMeasures,
                                                datasetCardinality=datasetCardinality,
                                                initialize=True,
                                                plotInter=plotInter,
                                                FARThreshold=FARThreshold)
        
        checkStepsList = [0 for _ in range(epochs)]
        # This for loop loops over epochs.
        for epochIdx, epoch in enumerate(range(1, epochs + 1)):
            

            miniBatchTrainLossVector = torch.tensor([])
            # Instantiation of variables and lists/tensors.
            epochLoss = 0.0
            
            # This for loop loops over minibatches
            for batchIdx, batch in enumerate(trainLoader):
                # The forward pass is slightly different for different loss functions
    
    
                
                # If-statement for the contrastive loss approach
                if lossFlag == "Contrastive":
    
                    # Collect data and forward passing the input.
                    X1, y1, X2, y2, flags = batch
                    output1, output2 = model(X1,
                                             X2)
    
                    # Storing the training loss.
                    lossTrain = lossFunc(output1, output2, flags)
    
    
    
                # If-statement for the face verification approach:
                # https://www.youtube.com/watch?v=0NSLgoEtdnw&t=120s
                elif lossFlag == "BCELoss":
    
                    # Collect data and forward passing the input.
                    X1, y1, X2, y2, flags = batch
                    preds = model(X1, X2)
    
                    lossTrain = lossFunc(preds, flags.unsqueeze(1))
    
    
    
                # If-statement for the triplet loss approach
                elif lossFlag == "Triplet" or lossFlag == "TripletTorch":
    
                    # Collecting triplets consisting of (anc, pos, neg), and their labels
                    XAnchor, XPos, XNeg, labelsPos, labelsNeg = batch
                    
                    
                    output1, output2, output3 = model(XAnchor,
                                                      XPos,
                                                      XNeg)
                    
                    # By the FaceNet-paper, we normalize the output embeddings
                    # such that ||f(x)||_2 = 1. We further mask these embeddings
                    # to select semihard-triplets.
                    output1, output2, output3 = \
                        torch.nn.functional.normalize(output1),\
                        torch.nn.functional.normalize(output2),\
                        torch.nn.functional.normalize(output3)
                    semihardMask = getSemihardTripletMask(output1, output2, output3, margin=0.2)
                    
                    
                    lossTrain = lossFunc(output1[semihardMask], 
                                         output2[semihardMask],
                                         output3[semihardMask])
                    
                    # lossTrain = lossFunc(output1, output2, output3)
    
                plotIterator += 1
                iterationMiniBatch += 1
                
                    
                miniBatchTrainLoss = lossTrain.item()
                epochLoss += lossTrain.item()
                
                # # We check the history of the epoch loss:
                # trainingLoss = torch.cat((trainingLoss,
                #                           torch.tensor(epochLoss).unsqueeze(0)))
                
                miniBatchTrainLossVector = torch.cat( (miniBatchTrainLossVector,
                                                       torch.tensor(miniBatchTrainLoss).unsqueeze(0) ))
                
                
    
                # if plotIterator % lossPlotIdx == 0:
                    
                #     lossXVector = torch.cat((lossXVector,
                #                             torch.tensor(iterationMiniBatch).unsqueeze(0)))
                    
                #     plotLossVector = torch.cat((plotLossVector, 
                #                                 torch.mean(miniBatchLossVector).clone().unsqueeze(0)))
                    # miniBatchLossVector = torch.tensor([])
                    
                
                # if plotIterator % gradStepPlotIdx == 0 or plotIterator == 1:
                    
                    
                    ###############################################################
                    # Here we define the subplot that plots training loss, and performances
                    ###############################################################
                    
    
                
                
                # If-statement to deal with the case where none of the triplets
                # are exclusively semi-hard.
                if math.isnan(lossTrain) == False:
                #     continue
                # else:
                    checkStepsList[epochIdx] += 1
                    
                    lossTrain.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # Calculating gradients, performing backpropagation and zeroing out
                # the gradients thereafter.

                
          
            
            
            # We store the model's performance for each lossUpdateInt.
            trainOutput = evalSiamese(model,
                                      trainPerformLoader, 
                                      device,
                                      performanceMeasures,
                                      lossFlag=lossFlag,
                                      distMetric="euclidean")
            
           
            thresholdTrainPerformance = trainOutput[0][performanceIdx]
            interTrainPerformances = torch.cat((interTrainPerformances, 
                                               thresholdTrainPerformance), dim=1)
            
            thresholdsTrain = trainOutput[1][performanceIdx]
            interThresholdsTrain = torch.cat((interThresholdsTrain,
                                              thresholdsTrain), dim=1)
            
            

            
            
            valOutput = evalSiamese(model,
                                      valLoader, 
                                      device,
                                      performanceMeasures,
                                      lossFlag=lossFlag,
                                      distMetric="euclidean",
                                      evalRuns=100)
            
            thresholdValPerformance = valOutput[0][performanceIdx]
            interValPerformances = torch.cat((interValPerformances,
                                             thresholdValPerformance), dim=1)   
            
            thresholdsVal = valOutput[1][performanceIdx]
            interThresholdsVal = torch.cat((interThresholdsVal,
                                            thresholdsVal), dim=1)
 


            
            
            
    

            
                    
    

            # NOTE: BECAUSE miniBatchValLossVector WAS ADDED AFTER RESULTS,
            # THE RUN-SEED MUST BE FROZEN AND RESET
            # Store the current RNG states
            rng_states = torch.get_rng_state()
            np_rng_state = np.random.get_state()
            random_state = random.getstate()        

            miniBatchValLossVector = eval_loss_func(valLoader, model)


            # NOTE: BECAUSE miniBatchValLossVector WAS ADDED AFTER RESULTS,
            # THE RUN-SEED MUST BE FROZEN AND RESET
            torch.set_rng_state(rng_states)
            np.random.set_state(np_rng_state)
            random.setstate(random_state)

            





            plotXVector = torch.cat((plotXVector,
                                      torch.tensor(epoch).unsqueeze(0)))
           
            
            plotTrainLossVector = torch.cat((plotTrainLossVector,
                                             miniBatchTrainLossVector.unsqueeze(1)), dim=1)
            plotValLossVector   = torch.cat((plotValLossVector,
                                             miniBatchValLossVector.unsqueeze(1)), dim=1)
            
            
            performanceXVector = torch.cat((performanceXVector, 
                                            torch.tensor(epochIdx).unsqueeze(0)))
            plotTrainPerformance = torch.cat((plotTrainPerformance, thresholdTrainPerformance), dim=1)
            plotValPerformance = torch.cat((plotValPerformance, thresholdValPerformance), dim=1)
            
            
            
            
            performance = thresholdTrainPerformance, thresholdValPerformance
            plotPerformance = plotTrainPerformance, plotValPerformance
            
            ###############################################################
            # Here we define the subplot that plots training loss, and performances
            ###############################################################
            
            # Misc. plotting options
            
            lossPerformanceFig, ax1, ax2 = lossPlot(lossFlag=lossFlag, batchSize=batchSize, 
                                                    performance=performance,
                                                    plotPerformance=plotPerformance,
                                                    performanceXVector=performanceXVector,
                                                    plotXVector=plotXVector,
                                                    plotTrainLossVector=plotTrainLossVector, 
                                                    plotValLossVector=plotValLossVector,
                                                    performanceMeasures=performanceMeasures, 
                                                    datasetCardinality=datasetCardinality, 
                                                    epoch=epoch, 
                                                    firstPoint=firstPoint,
                                                    plotInter=plotInter,
                                                    fig=lossPerformanceFig, ax1=ax1, ax2=ax2,
                                                    FARThreshold=FARThreshold)
            
            
            firstPoint=False
            

            # We save the model for each epoch just in case some bug or error
            # crashes the program
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'interTrainPerformances': interTrainPerformances,
                        'interValPerformances': interValPerformances,
                        'plotTrainLossVector': plotTrainLossVector,
                        'plotValLossVector': plotValLossVector,
                        'checkStepsList': checkStepsList}, 
                        savePATH + "/saved.pt")
    
    
            # If TAR is in the performance measures to be output, we supply the 
            # output with the chosen FARThreshold value (default at 0.01).
            outputPerformanceMeasures = performanceMeasures.copy()
            if "TAR" in outputPerformanceMeasures:
                TARIndex = outputPerformanceMeasures.index("TAR")
                outputPerformanceMeasures[TARIndex] = "TAR @ FAR(" + str(FARThreshold) + ")"
                
                
            # Outputs made user-friendly.
            outputPerformancesTrain = {performanceMeasure + " (mean, std)": (round(torch.mean(performance).item(), 5), 
                                                            round(torch.std(performance).item(), 5)) for \
                                       performance, performanceMeasure in \
                                       zip(thresholdTrainPerformance, outputPerformanceMeasures)}
                
            outputPerformancesVal   = {performanceMeasure + " (mean, std)": (round(torch.mean(performance).item(), 5), 
                                                            round(torch.std(performance).item(), 5)) for \
                                       performance, performanceMeasure in \
                                       zip(thresholdValPerformance, outputPerformanceMeasures)}
                
                
            

            # Motivational text to keep the user up to date on the model training.
            # if epoch in epochsXAxis:
            if epoch % 1 == 0:
                print('\n{}  |  Epoch {}  |  Training loss {:.3f}  |  Validation loss {:.3f}'.format(
                    datetime.now().time(), epoch, torch.nan_to_num(miniBatchTrainLossVector, nan=0).mean(dim=0),
                    torch.nan_to_num(miniBatchValLossVector, nan=0).mean(dim=0)))
                print('\nCurrent time: {}  \n|  Training Performance:    {}  |\n|  Validation Performance:  {}  |\n'.format(
                    datetime.now().time(), 
                    outputPerformancesTrain, 
                    outputPerformancesVal))
    
            if learningVisualizationEpoch:
                plt.show()
            

            
            
            # bestNumEpochs is defined for each performance measure.
            for performanceMeasure, interValPerformanceMean, interValPerformanceSTD in zip(performanceMeasures, 
                                                               torch.mean(interValPerformances, dim=2),
                                                               torch.std(interValPerformances, dim=2)):
                
                bestModelIdx = torch.argmax(interValPerformanceMean)
                bestPerformances[performanceMeasure] = (str(int(bestModelIdx + 1)),
                                                            interValPerformanceMean[bestModelIdx].item(),
                                                            interValPerformanceSTD)
                    
                    
            # bestNumEpochs = int(torch.argmax(interValPerformances)) + 1
        
        
        
        
        
        
        
        
         
            # If we manage to classify everything correctly, there is no need
            # to continue training.
            if sum(torch.mean(interValPerformances[:, -1]) == \
                   torch.ones((len(performanceMeasures)), dtype=float)) == len(performanceMeasures):
                
                # We write the number of epochs that gave the best performances
                runParameters["bestPerformances"] = bestPerformances
                
                # The models are stored in a dictionary.
                bestModels = {performanceMeasure: model for \
                              performanceMeasure, performanceStats in bestPerformances.items()}
                
                modelTuple = (interValPerformances, 
                              interTrainPerformances,
                              runParameters, 
                              bestModels,
                              bestPerformances,
                              lossPerformanceFig,
                              interThresholdsVal,
                              interThresholdsTrain)
                
                
                
                return modelTuple
            
            # We can also choose to save the model for each epoch. 
            if saveInterEpoch and epoch in epochsXAxis:
                
                modelsDictionary["NumEpochs: "+str(epoch)] = model
            
            # If we chose to save models between epochs, we pick out the model that 
            # scored the highest validation performance. We also need to overwrite
            # intermediate values that are to be added to the returned tuple.
            if saveInterEpoch:
                
                # We write the number of epochs that gave the best performances
                runParameters["bestPerformances"] = bestPerformances
                
                bestModels = {performanceMeasure: modelsDictionary["NumEpochs: " + \
                              str(performanceStats[0])] for performanceMeasure, performanceStats in bestPerformances.items()}
               
                    
                modelTuple = (interValPerformances,
                              interTrainPerformances,
                              runParameters,
                              bestModels,
                              bestPerformances,
                              lossPerformanceFig,
                              interThresholdsVal,
                              interThresholdsTrain)
        
        
            # If we chose NOT to save models between epochs, the most current model
            # is selected, and we don't have to overwrite anything as the most
            # current model is seleted.
            else:
                bestModels = {performanceMeasure: model for \
                              performanceMeasure in performanceMeasures}
                    
                modelTuple = (interValPerformances,
                              interTrainPerformances,
                              runParameters,
                              bestModels,
                              bestPerformances,
                              lossPerformanceFig,
                              interThresholdsVal,
                              interThresholdsTrain)
    
    

        
    except KeyboardInterrupt:
        print("\n\n\nTraining interrupted\n\n\nPlease wait... saving experiment files...\n\n")
        
        # We write the number of epochs that gave the best performances
        runParameters["bestPerformances"] = bestPerformances
        
        
        
        bestModels = {performanceMeasure: model for \
                      performanceMeasure, performanceStats in bestPerformances.items()}
        
        modelTuple = (interValPerformances, 
                      interTrainPerformances,
                      runParameters, 
                      bestModels,
                      bestPerformances,
                      lossPerformanceFig,
                      interThresholdsVal,
                      interThresholdsTrain)
        
        return modelTuple
    return modelTuple




def evalSiamese(model, evalLoader, device, performanceMeasure,
                embeddingVisualization=False, distMetric="euclidean", dataFlag=None,
                lossFlag="Triplet", nnDecisionBoundaries=False, numClassesPlot=None,
                modelType=None, PCATransform=True, FARThreshold=10e-3, datasetTransform="tSNE",
                evalRuns=10, pickThreshold=(False, None)):

    # Letting the CNN know that it is in evaluation-mode
    model.eval()
    
   


    with torch.no_grad():
        
        # TODO 
        # ---Incorporate batching of evaluation scores---
        
        
        # Instantiating performances and performance lists.
        accuraciesThresholds = []
        precisionsThresholds = []
        recallsThresholds = []
        F1ScoresThresholds = []
        TARsThresholds = []
        AUCsList = []
        
        accuracyOutput = []
        precisionOutput = []
        recallOutput = []
        F1ScoreOutput = []
        TAROutput = []
        AUCOutput = []

        
        
        augment = evalLoader.dataset._augmentImages
        
        # We extract the whole dataset, and do a forward pass on the whole set.
        evalDataset = evalLoader.dataset.dataset
        
        # The different IDs in the evaluation set is stored.
        evalIDs = list(evalDataset.keys())
        
        

        # The images' IDs are made such that evalImagesTensor[idx, :, :]
        # corresponds to the label evalLabels[idx]
        evalLabels = []
        for ID in evalIDs:

            numUniqueIDs = evalDataset[ID].shape[0]

            for uniquePic in range(0, numUniqueIDs):
                evalLabels.append(ID)


        
        Y = torch.zeros((len(evalLabels), len(evalLabels)))

        

        # We iteratively make the similarity matrix
        for i, label in enumerate(evalLabels):
            Y[i, :] = torch.tensor([int(labelIteration
                                        != label) for
                                    labelIteration in
                                    evalLabels])

            # If we choose to evaluate the network by the verification approach,
            # the pairwiseSimilarity variable is an input matrix of N samples of
            # D-dimensional vectors (NxD). The i-th row (from the enumeration
            # index of the for loop) will always consist of zeros, as the i-th
            # embedding similarity-checked with the i-th embedding will reveal
            # full similarity, as it should.
            # if modelType == "Verification":
            #     pairwiseSimilarity = nn.Sigmoid()(
            #         model.similarityNet(pairwiseSimilarity)).view(1, -1)        
        
        evalFlags = Y
        triuIndices = torch.triu_indices(evalFlags.shape[0],
                                         evalFlags.shape[1], offset=1)
        
        # We flatten the upper triangular matrix
        evalFlagsFlat = evalFlags[triuIndices[0], triuIndices[1]]
        
        
        for evalRun in range(evalRuns):
        
            # Stacking the evaluation data
            evalImagesTensor = torch.cat(list(evalDataset.values()), dim=0)

            if evalLoader.dataset.imageFlag == True:
                evalImagesTensor = evalImagesTensor.unsqueeze(1)

            # Augmenting the evaluation data
            for augIdx in range(evalImagesTensor.shape[0]):
                evalImagesTensor[augIdx] = torch.tensor(augment(np.array(evalImagesTensor[augIdx][0]))).unsqueeze(0)
            
            # We run the model on the images, creating embeddings.
            embeddedOutput = model(evalImagesTensor.to(device))
            embeddedOutput = torch.nn.functional.normalize(embeddedOutput)
            
            
            YPred = torch.tensor(distSim(embeddedOutput.cpu(), metric=distMetric))
            
            # NOTE, pickThreshold IS ONLY TRUE FOR EVALUATING TESTING DATASETS.
            if pickThreshold[0] == True:
                thresholds = torch.tensor([pickThreshold[1]])
            else:
                # The similarity is thresholded, and the best threshold is picked.
                thresholds = torch.linspace(torch.min(YPred), torch.max(YPred), 500)
            
            
    
            thresholdAccuracyList = []
            thresholdPrecisionList = []
            thresholdRecallList = []
            thresholdF1List = []
            thresholdTARList = []
            thresholdFARList = []
            
            
            
            numSame = torch.sum(evalFlagsFlat == 0)
            numDiff = torch.sum(evalFlagsFlat == 1)
            
            iterator = 0
            for threshold in thresholds:
                iterator += 1
                # We threshold the distances as FaceNet does.
                predEvalFlags = torch.where(
                    YPred > threshold, torch.tensor(1), torch.tensor(0))
                predEvalFlagsFlat = predEvalFlags[triuIndices[0], triuIndices[1]]
    
                # Here we create the confusion matrix
                # Remember, flag=0 means same, flag=1 means different
                confMatrix = confusion_matrix(predEvalFlagsFlat, evalFlagsFlat,
                                              labels=[torch.tensor(0), torch.tensor(1)]).T
    
                
                # ACCURACY
                thresholdAccuracy = torch.sum(
                    predEvalFlagsFlat == evalFlagsFlat) / len(evalFlagsFlat)
    
    
                TA = confMatrix[0, 0]
                FA = confMatrix[1, 0]
                FR = confMatrix[0, 1]
                TR = confMatrix[1, 1]
                
                # By definition, we avoid division by zero by setting precision/
                # recall/F1-score to be 0 appropriately.
                # PRECISION
                if TA + FA == 0:
                    thresholdPrecision = 0
                else:
                    thresholdPrecision = TA / \
                        (TA + FA)
    
                # RECALL
                if TA + FR == 0:
                    thresholdRecall = 0
                else:
                    thresholdRecall =  TA / \
                        (TA + FR)
    
                # F1-SCORE
                if thresholdPrecision + thresholdRecall == 0:
                    thresholdF1 = 0
                else:
                    thresholdF1 = (2 * thresholdPrecision * thresholdRecall) / \
                        (thresholdPrecision + thresholdRecall)
    
                # TRUE ACCEPT RATE
                # TAR = TA / numSame
                TAR = TA / (TA + FR)
    
                # FALSE ACCEPT RATE
                # FAR = FA / numDiff
                FAR = FA / (FA + TR)
    
    
                thresholdAccuracyList += list([thresholdAccuracy])
                thresholdPrecisionList += list([thresholdPrecision])
                thresholdRecallList += list([thresholdRecall])
                thresholdF1List += list([thresholdF1])
                thresholdTARList += list([TAR])
                thresholdFARList += list([FAR])
                
                
            # Adding all the performance measures. This way of adding them
            # will also serve to be compatible with batching the evaluations.
            accuraciesThresholds += list([thresholds[np.argmax(thresholdAccuracyList)]])
            accuracyOutput += list([np.max(thresholdAccuracyList)])
    
    
            precisionsThresholds += list([thresholds[np.argmax(thresholdPrecisionList)]])
            precisionOutput += list([np.max(thresholdPrecisionList)])
    
    
            recallsThresholds += list([thresholds[np.argmax(thresholdRecallList)]])
            recallOutput += list([np.max(thresholdRecallList)])
    
    
            F1ScoresThresholds += list([thresholds[np.argmax(thresholdF1List)]])
            F1ScoreOutput += list([np.max(thresholdF1List)])
    
    
            # Finding the index closest to the FAR threshold
            closestToThreshold = min(thresholdFARList, key=lambda x: abs(x - FARThreshold))
            thresholdFARIndex = thresholdFARList.index(closestToThreshold)
            
            # NOTE: THIS WAS WRONG, AND WAS FIXED TWO DAYS PRIOR TO HAND-IN
            # TARsThresholds += list([thresholds[np.argmax(thresholdTARList)]])
            
            # THIS IS THE CORRECTED TARsThresholds 
            TARsThresholds += list([thresholds[thresholdFARIndex]])
            TAROutput += list([thresholdTARList[thresholdFARIndex]])
    
            # AUCsList += list([auc(thresholdFARList, thresholdTARList)])
            # AUCsList += list([auc(thresholdFARList, thresholdTARList)])
            # AUCOutput += list([AUCsList[0] / evalRuns])
            
            

        # Visualization code
        if embeddingVisualization == True:
                
            # Returning the figures for saving.
            singValFig, PCAScatterFig, tSNEScatterFig = evaluationPlot(evalLoader, 
                                                                       numClassesPlot, 
                                                                       model, 
                                                                       device, 
                                                                       dataFlag)
            AUCPlotFig, AUCPlotAx = plt.subplots()
            AUCPlotAx.plot(thresholdFARList, thresholdTARList)
            AUCPlotAx.set_title(label="Receiver Operating Characteristics curve (ROC)",
                                fontweight="bold", loc='center', pad=None)
            AUCPlotAx.set_xlabel("False Accept Rate", style='italic')
            AUCPlotAx.set_ylabel("True Accept Rate", style='italic')
            
            
        
            return (torch.tensor([[accuracyOutput], 
                                 [precisionOutput], 
                                 [recallOutput], 
                                 [F1ScoreOutput],
                                 [TAROutput],
                                 [AUCOutput]]),
                    torch.tensor([[accuraciesThresholds], 
                                 [precisionsThresholds], 
                                 [recallsThresholds], 
                                 [F1ScoresThresholds],
                                 [TARsThresholds]])), singValFig, PCAScatterFig, tSNEScatterFig, AUCPlotFig


    # We reset the model to be in train-mode again
    model.train()

    # Returning the performance measure
    return (torch.tensor([[accuracyOutput], 
                         [precisionOutput], 
                         [recallOutput], 
                         [F1ScoreOutput],
                         [TAROutput]]),
                         # [AUCOutput]]),
                         
            torch.tensor([[accuraciesThresholds], 
                         [precisionsThresholds], 
                         [recallsThresholds], 
                         [F1ScoresThresholds],
                         [TARsThresholds]]))