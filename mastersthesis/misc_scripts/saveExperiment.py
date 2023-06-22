# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:34:59 2023

@author: kisen
"""

import pandas as pd
from torchinfo import summary



def parametersToCSV(systemParameters, timePath):
    maxLen = max(len(parameters) for parameters in systemParameters)
    
    parametersDict = {'DATASET_HP': list(systemParameters[0].items()), 
                      'MODEL_HP': list(systemParameters[1].items()),
                      'SYSTEM_HP': list(systemParameters[2].items())}
    
    for parameterDict in parametersDict:
        parametersDict[parameterDict] += [" "] * (maxLen - len(parametersDict[parameterDict]))
                
    dfToSave = pd.DataFrame.from_dict(parametersDict)
    
    dfToSave.to_csv(timePath + '/parameters.csv', sep=';', index=False)
    
def modelToTXT(model, timePath, dimX, dimY):
    
    modelOutputs = str(summary(model, input_size=(1, 1, dimX, dimY)))
    modelArchitecture = str(model.state_dict)
    
    with open(timePath + '/model_outputs.txt', 'w', encoding='utf-8') as file:
        file.write(modelOutputs)
        file.close()
        
    with open(timePath + '/model_architecture.txt', 'w', encoding='utf-8') as file:
        file.write(modelArchitecture)
        file.close()
        
def dictParametersToTXT(dictParameters, experimentPath, fileName):
    
    dictParameters = "\n".join([f"{k}: {v}" for k, v in dictParameters.items()])
    
    with open(experimentPath + "/{}.txt".format(fileName), 'w', encoding='utf-8') as file:
        file.write(dictParameters)
        file.close()