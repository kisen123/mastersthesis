# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:46:56 2023

@author: kisen
"""

# This function saves experiments
def saveExperimentInfo():
    # Because the hyperparameter "epochs" could be variable depending
    # on if saveInterEpoch=True, modelsEpochs[2] is the training-run's
    # hyperparameters.
    valPerformanceList.append(modelTrainTuple[0])
    trainPerformanceList.append(modelTrainTuple[1])
    RUN_HP = modelTrainTuple[2]
    systemID = "Run ID: " + date + " " + time
    systemParameters[systemID] = [DATASET_HP, RUN_HP, SYSTEM_HP]
    
    
    # We save the HP dictionaries and model options for logging experiments
    parametersToCSV(systemParameters[systemID], timePath)
    modelToTXT(model, timePath, 129, 129)
    
    
    PCATransform = True
    
    
    _, PCAInfoFigTrain, scatterFigTrain = evalSiamese(model, trainPerformLoader, device,
                                        "F1-score", embeddingVisualization=True,
                                        dataFlag="Training data",
                                        numClassesPlot=int(DATASET_HP['numIDs'] * \
                                                           DATASET_HP['trainValTestRatio'][0]/ 2),
                                        lossFlag="BCELoss", distMetric="p-norm",
                                        modelType="Veration", PCATransform=PCATransform)
    
    _, PCAInfoFigVal, scatterFigVal = evalSiamese(model, valLoader, device,
                                    "F1-score", embeddingVisualization=True,
                                    dataFlag="Validation data",
                                    numClassesPlot=int(DATASET_HP['numIDs'] * \
                                                       DATASET_HP['trainValTestRatio'][1]/ 2),
                                        lossFlag="Triplet", distMetric="p-norm",
                                    modelType="Verifation", PCATransform=PCATransform)
    
    PCAInfoFigTrain.savefig(fname=timePath + "/PCAInfoTrain.png")
    scatterFigTrain.savefig(fname=timePath + "/scatterEmbeddingSpaceTrain.png")
    PCAInfoFigVal.savefig(fname=timePath + "/PCAInfoVal.png")
    scatterFigVal.savefig(fname=timePath + "/scatterEmbeddingSpaceVal.png")
    plt.close('all')