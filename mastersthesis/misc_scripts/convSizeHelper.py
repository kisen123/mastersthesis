# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:10:02 2023

@author: kisen
"""

"""

Section for designing CNN layers, and check that convolutions are valid.


def convSizeHelper(inputSize=None, padding=None, filterSize=None, stride=None, 
                   layerType=None):
    print(str(layerType))
    print("Input: " + str(int(inputSize)) + "x" + str(int(inputSize)))
    
    outputSize = np.floor((inputSize + 2 * padding - filterSize) / (stride) + 1)
    print("Output: " + str(int(outputSize)) + "x" + str(int(outputSize)) + '\n')
    return outputSize

# Conv
convSizeHelper(inputSize=64, padding=0, filterSize=11, stride=3, layerType="Conv")

#Pool
convSizeHelper(inputSize=18, padding=0, filterSize=3, stride=2, layerType="Pool")

# Conv
convSizeHelper(inputSize=8, padding=0, filterSize=3, stride=1, layerType="Conv")

# Pool
convSizeHelper(inputSize=6, padding=0, filterSize=2, stride=2, layerType="Pool")

# Conv
convSizeHelper(inputSize=3, padding=0, filterSize=3, stride=1, layerType="Conv")

# Pool
convSizeHelper(inputSize=4, padding=0, filterSize=2, stride=2)

# Conv
convSizeHelper(inputSize=2, padding=0, filterSize=2, stride=1)

# Flattening layer
convSizeHelper(inputSize=4, padding=0, filterSize=4, stride=1)




#%%
# Conv
convSizeHelper(inputSize=100, padding=0, filterSize=11, stride=4)

#Pool
convSizeHelper(inputSize=23, padding=0, filterSize=3, stride=2)

# Conv
convSizeHelper(inputSize=11, padding=0, filterSize=5, stride=1)

# Pool
convSizeHelper(inputSize=7, padding=0, filterSize=2, stride=2)

# Conv
convSizeHelper(inputSize=3, padding=0, filterSize=3, stride=1)
"""