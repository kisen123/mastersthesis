# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:01:55 2023

@author: kisen
"""

import torch
import numpy as np
import random
import math

# Needed for rotation augmentation
import cv2
from dataset_transform.rotationScripts import rotationMatrix, polarIm, polarTransform

class SiameseDataset(torch.utils.data.Dataset):
    
    # Constructor function, to set dataset, randomization seeds and other
    # standard variables. 
    def __init__(self, device, dataset, datasetTypeFlag, mode, lossFunction,
                 imageFlag=True, percentOfPairs=1.0, imageRotationTuple=(0,0),
                 polarBool=False, blurMaxSize=0, visualize=False, unknownDictionary={}):
        
        
        self.visualize = visualize
        
        # The image rotation hyperparameters are set
        self.imageRotationTuple = imageRotationTuple
        self.augmentInterval = self.imageRotationTuple[1] - self.imageRotationTuple[0]
        self.clockRotate = -self.augmentInterval
        self.cClockRotate = self.augmentInterval
        
        
        self.rotateBool = (self.clockRotate != 0) == True or \
                          (self.cClockRotate != 0) == True
        
        # Decide if we polar transform the rotated image, and the mask size
        # of the potential blurring of the image before polar transformation.
        # This combats aliasing.
        self.polarBool = polarBool
        self.blurMaxSize = blurMaxSize
        
        # Dataset as a dictionary -> key=str(ID), value=(n, x, y)
        # We ONLY polar-transform the dataset in the __init__-method
        # with this if statement. During training, we need to augment, 
        # and followingly polar-transform on the fly as they are 
        # non-commutative operations.
        
        self.dataset = dataset
        
        # TODO REMOVE POLARTRANSFORM AFTER FIX OF EVALUATION CODE IS DONE
        # if self.polarBool and mode == "evaluation":
        #     self.dataset = polarTransform(dataset)
        # else:
        #     self.dataset = dataset
        
        self.unknownDictionary = unknownDictionary
        
        
        
        # We keep on searching for an ID that has at least one image.
        # This code is much more relevant for the validation and test datasets, 
        # as they are very scarce.
        self.randomImage = random.choice(list(self.dataset.values()))
        while self.randomImage.shape[0] == 0:
            self.randomImage = random.choice(list(self.dataset.values()))
            
        self.resolutionTuple = self.randomImage[0].shape
        self.dimX, self.dimY = self.resolutionTuple[0], self.resolutionTuple[1]
        
        # Border value used for padding rotated images
        self.borderValue = self.randomImage[0][0,0].item()
        
        
        # List of the str(ID)-keys in self.dataset
        self.IDs = list(dataset.keys())
        
        # Speaks to if the user uses this Dataset construction object
        # for "training" or "evaluation".
        self.mode = mode
        
        # Boolean to determine if input is 2D or regular 1D
        self.imageFlag = imageFlag
        
        
        self.lossFunction = lossFunction
        self.numCollections = len(self.IDs)
        self.percentOfPairs = percentOfPairs
        self.device = device
        
        # Summing together the number of lonely IDs/unknown IDs
        self.numUnknowns = sum([len(unknownDictionary[key]) for key in list(unknownDictionary.keys())])
        self.numLonelies = sum([1 for key in list(dataset.keys()) if len(dataset[key]) == 1])
        
        # If we are to evaluate a dataset, we want to be able to determine how
        # many genuine/impostor pairs to evaluate.
        if self.mode == "evaluation":
            
            numGenPairs = sum(len(tensorTuple) for tensorTuple in self.dataset.values())
            

        
        # For each dataset, we set a different RNG.
        if datasetTypeFlag == "train":
            self.rnd = np.random.RandomState(0)
        elif datasetTypeFlag == "val":
            self.rnd = np.random.RandomState(1)
        
        # The trainPerform dataset is used to evaluate the training data as the
        # model trains.
        elif datasetTypeFlag == "trainPerform":
            self.rnd = np.random.RandomState(2)
        elif datasetTypeFlag == "test":
            self.rnd = np.random.RandomState(3)
            
            # TODO I don't think I need this
        elif datasetTypeFlag == "unknown":
            self.rnd = np.random.RandomState(3)    
        
        
        
        
        
        # We add the unknownDictionary to the datasetDictionary.
        # NOTE: This is only during training, and is only meant
        # to let lonelies/unknowns be a potential part of negative images.
        if unknownDictionary != {}:
            self.dataset = {**dataset, **unknownDictionary}
            self.IDs = list(self.dataset.keys())
        
        
    # Magic method to define the length of a Siamese Dataset.
    def __len__(self):
        

        numIms = np.sum([int(len(self.dataset[ID])) for ID in self.IDs])
        
        # If the unknownDictionary dataset is not empty, we exclude the
        # that get stacked up in the last list element.
        if self.unknownDictionary != {}:
            numSamplesID = [len(sampleID) for sampleID in list(self.dataset.values())][:-1]
        else:
            numSamplesID = [len(sampleID) for sampleID in list(self.dataset.values())]
        
        
        # Returns the pair cardinality of the dataset
        if self.lossFunction == "Contrastive" or self.mode == "evaluation":

            cardinality = int(( numIms * (numIms-1) ) / 2)
            
            
            
        # Returns the triplet cardinality of the dataset
        elif self.lossFunction == "Triplet" or self.lossFunction == "TripletTorch":
            
            # TODO HERE!!!!
            cardinality = sum(math.factorial(numSamples) * (numIms - numSamples + self.numLonelies + self.numUnknowns) for numSamples in numSamplesID)
            
        
        return int(self.percentOfPairs * cardinality)
        
    
    # Method to augment images. Called on during training.
    def _augmentImages(self, *args):
        augmentedImages = ()
        for i, image in enumerate(args):
            # Random angles and their rotation matrices are set here
            randomAngle = self.rnd.uniform(self.clockRotate, 
                                            self.cClockRotate + 1)
            
            rotationMatrixRand = rotationMatrix(self.dimX, self.dimY, randomAngle)
            
            
            # Random blurring mask sizes are set here
            if self.blurMaxSize == 1:
                randomBlurSize = 1
                
            else:
                randomBlurSize = self.rnd.choice(list(range(1, self.blurMaxSize, 2)))
            
            # This code blurs the image prior to 
            # eventually polar-transforming it.
            if self.blurMaxSize != 0:
    
                image = cv2.GaussianBlur(image, 
                                       (randomBlurSize, 
                                       randomBlurSize), 0)
            
            if self.rotateBool:
                # TODO Training trick: We rotate the negative image
                # with the same angle as the anchor. NOT DONE HERE,
                # CAN BE DONE IF THE DATASET IS MADE WITHOUT ROTATIONS,
                # AND IF WE DO ROTATIONS ONLY IN THE __getitem__ METHOD HERE.
                
                
                
                image = cv2.warpAffine(np.array(image),
                                          rotationMatrixRand,
                                          (self.dimX, self.dimY),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=self.borderValue)
                
            
            
            if self.polarBool:
                # This code polar-transforms the images
                image = polarIm(image)
                
                
                
                
            
            # TODO CAN BE REMOVED, WAS JUST TO MAKE AN ILLUSTRATION IN THE THESIS.
            # if self.polarBool:
            #     # This code polar-transforms the images
            #     polarImage = polarIm(image)
                
                
            # import matplotlib.pyplot as plt
            
            
            # fig, ax = plt.subplots(1, 2)
            
            # fig.set_dpi(200)
            # ax[0].imshow(image, cmap='gray')
            # ax[0].axis("off")
            # ax[0].set_title("Salmon louse image\n(Regular)", size=10)
            
            # ax[1].imshow(polarImage, cmap='gray')
            # ax[1].axis("off")
            # ax[1].set_title("Salmon louse image\n(Polar-transformed)", size=10)
            
            # plt.show()
            
                
                
                
                
                
            augmentedImages += (image, )
            
        if i == 0:
            augmentedImages = augmentedImages[0]
        return augmentedImages
    
    # Magic-method retrieved by the dataloader and for loop.
    def __getitem__(self, datasetIdx):
        """
        datasetIdx is set by how many __getitem__ is implicitly called. It
        will go from 0 to __len__(self)
        
        __getitem__ returns a tuple:
        self.lossFunction = "Triplet" -> (anchors, positives, negatives, labelsPos, labelsNeg)
        self.lossFunction = "Contrastive" -> (ims1, labels1, ims2, labels2, flags)
        """
        # flag    ->    0 = same class, 1 = different classes
        
        
        # The Siamese Dataset under training depends on the loss-function
        # if self.mode == "training":
            
        # The data point retrieval logic for Triplet Loss
        if self.lossFunction == "Triplet" or \
           self.lossFunction == "TripletTorch" or \
           self.visualize:
                
            
            # We extract the anchor label
            labelAnc = self.rnd.choice(self.IDs)
            
            # The number of anchor label images is set
            numAnchorCollection = len(self.dataset[labelAnc])
            
            # The index from the anchor label images is set, and the anchor
            # image extracted.
            anchorIdx = self.rnd.randint(0, numAnchorCollection)
            anchorIm = np.array(self.dataset[labelAnc][anchorIdx])
            
            # If the anchor is a lonely/unknown label, we set the positive
            # image to be the same image. When augmented, the image is not
            # trivial to be the same as the anchor.
            if numAnchorCollection == 1:
                positiveIdx = anchorIdx
                
            else:
                # We add a while-statement because we don't want the positive
                # to be the anchor image if there are other positives than
                # the trivial one, i.e. the anchor image
                positiveIdx = self.rnd.randint(0, numAnchorCollection)
                while positiveIdx == anchorIdx:
                    positiveIdx = self.rnd.randint(0, numAnchorCollection)
                    
            positiveIm = np.array(self.dataset[labelAnc][positiveIdx])
            
            
            # Choose a random negative label with a different label
            # than the anchor data point. Note also that negative images
            # might come from the unknownDictionary dataset too.
            labelNeg = labelAnc
            while labelNeg == labelAnc:
                if self.unknownDictionary != {}:
                    negativeIDIdx = self.rnd.randint(0, self.numCollections) + 1
                else:
                    negativeIDIdx = self.rnd.randint(0, self.numCollections)
                labelNeg = self.IDs[negativeIDIdx]
                

            # We choose a random data point from the negative label
            numNegCollection = len(self.dataset[labelNeg])
            negativeCollectionIdx = self.rnd.randint(0, numNegCollection)
            
            # We now have the negative label, and a fixed random choice for which 
            # of the images in the collection of the negative sample to choose.
            negativeIm = np.array(self.dataset[labelNeg][negativeCollectionIdx])
            
            
            # TODO 
            # This logic makes the unknown IDs be very rarely picked as a 
            # negative although they are rich with samples.
            # TODO
            
            
            # We call upon the _augmentImages method to augment the
            # images as described in the thesis.
            anchorIm, positiveIm, negativeIm = self._augmentImages(anchorIm,
                                                                   positiveIm,
                                                                   negativeIm)
            
                
            return (torch.tensor(anchorIm).unsqueeze(0), 
                    torch.tensor(positiveIm).unsqueeze(0),
                    torch.tensor(negativeIm).unsqueeze(0), 
                    
                    labelAnc, labelNeg)
            
            
        # The data point retrieval logic for Contrastive loss is similar to 
        # how the Siamese Neural Network evaluation methods collect data points
        # if self.mode == "evaluation" or self.lossFunction == "Contrastive" or \
        #     self.lossFunction == "BCELoss":
            
        #     # We choose genuine/impostor pairs from a customizable binary
        #     # distribution in Contrastive training
            
        #     if self.mode == "training":
        #         flags = self.rnd.choice([0, 1], p=[0.5, 0.5])
        #     else:
        #         flags = self.rnd.choice([0, 1], p=[0.5, 0.5])
                
                
        #     label = self.rnd.choice(self.IDs)
            
            
        #     simIdx = self.rnd.randint(0, self.numCollections-1)
            
        #     numLabelCollection = len(self.dataset[label])
            
        #     # If we come across an ID that only has one image, we need to 
        #     # manually set the similarity flag to be 1.
        #     if numLabelCollection == 1 and flags == 0:
        #         flags = 1
            
        #     # This if-statement sets the state for two images with same label
        #     if flags == 0: 
                
        #         simLabel = label
            
        #         numLabelCollectionSim = numLabelCollection
                
        #         # We choose two random images in a collection
        #         siameseDupeFlag1 = self.rnd.randint(0, numLabelCollection)
        #         siameseDupeFlag2 = self.rnd.randint(0, numLabelCollectionSim)
                
        #         # The while-loop ensures the images from the collection to be 
        #         # different.
        #         while siameseDupeFlag1 == siameseDupeFlag2:
        #             siameseDupeFlag2 = self.rnd.randint(0, numLabelCollection)
                
        #     # This else-statement sets the state for two images with different labels.
            
        #     # TODO Here we might be able to infer some tricks for learning, by
        #     # for example deliberately choosing points that are close together.
        #     else:
                
                
        #         # The while-loop ensures the images are of a different label.
        #         while self.IDs[simIdx] == label:
        #             simIdx += 1
                    
                    
        #             # Resets the iteration through the IDs if it were to come 
        #             # to that.
        #             if simIdx == self.numCollections: simIdx = 0
            
        #         simLabel = self.IDs[simIdx]
                
        #         numLabelCollectionSim = len(self.dataset[simLabel])
        #         # Here it doesn't matter which images in the collection are 
        #         # chosen.
        #         siameseDupeFlag1 = self.rnd.randint(0, numLabelCollection)
        #         siameseDupeFlag2 = self.rnd.randint(0, numLabelCollectionSim)
            
        #     ims1 = self.dataset[label][siameseDupeFlag1]
        #     label1 = label
            
            
        #     ims2 = self.dataset[simLabel][siameseDupeFlag2]
        #     label2 = simLabel
            
        #     # Must NOT be done in evaluation 
        #     # TODO
        #     ims1, ims2 = self._augmentImages(ims1, ims2)
                
        #     flags = torch.tensor(flags, dtype=torch.float32).to(device=self.device)
            

            
        #     return (torch.tensor(ims1).unsqueeze(0), label1,
        #             torch.tensor(ims2).unsqueeze(0), label2,
                    
        #             flags)    







# # Random angles and their rotation matrices are set here
# randomAngle1 = self.rnd.uniform(self.clockRotate, 
#                                 self.cClockRotate + 1)
# randomAngle2 = self.rnd.uniform(self.clockRotate, 
#                                 self.cClockRotate + 1)
# randomAngle3 = self.rnd.uniform(self.clockRotate, 
#                                 self.cClockRotate + 1)

# rotationMatrix1 = rotationMatrix(self.dimX, self.dimY, randomAngle1)
# rotationMatrix2 = rotationMatrix(self.dimX, self.dimY, randomAngle2)
# rotationMatrix3 = rotationMatrix(self.dimX, self.dimY, randomAngle3)



# # Random blurring mask sizes are set here
# if self.blurMaxSize == 1:
#     randomBlurSize1 = 1; randomBlurSize2 = 1; randomBlurSize3 = 1
    
# else:
#     randomBlurSize1 = self.rnd.choice(list(range(1, self.blurMaxSize, 2)))
#     randomBlurSize2 = self.rnd.choice(list(range(1, self.blurMaxSize, 2)))
#     randomBlurSize3 = self.rnd.choice(list(range(1, self.blurMaxSize, 2)))

# if self.rotateBool:
#     # TODO Training trick: We rotate the negative image
#     # with the same angle as the anchor. NOT DONE HERE,
#     # CAN BE DONE IF THE DATASET IS MADE WITHOUT ROTATIONS,
#     # AND IF WE DO ROTATIONS ONLY IN THE __getitem__ METHOD HERE.
    
    
    
#     anchorIm = cv2.warpAffine(np.array(anchorIm),
#                               rotationMatrix1,
#                               (self.dimX, self.dimY),
#                               borderMode=cv2.BORDER_CONSTANT,
#                               borderValue=self.borderValue)
    
#     positiveIm = cv2.warpAffine(np.array(positiveIm),
#                                 rotationMatrix2,
#                                 (self.dimX, self.dimY),
#                                 borderMode=cv2.BORDER_CONSTANT,
#                                 borderValue=self.borderValue)
    
#     negativeIm = cv2.warpAffine(np.array(negativeIm),
#                                 rotationMatrix3,
#                                 (self.dimX, self.dimY),
#                                 borderMode=cv2.BORDER_CONSTANT,
#                                 borderValue=self.borderValue)
    
# # This code blurs the image prior to 
# # eventually polar-transforming it.
# if self.blurMaxSize != 0:

#     anchorIm = cv2.GaussianBlur(anchorIm, 
#                                 (randomBlurSize1, 
#                                  randomBlurSize1), 0)
#     positiveIm = cv2.GaussianBlur(positiveIm, 
#                                  (randomBlurSize2, 
#                                   randomBlurSize2), 0)
#     negativeIm = cv2.GaussianBlur(negativeIm, 
#                                  (randomBlurSize3, 
#                                   randomBlurSize3), 0)

# if self.polarBool:
#     # This code polar-transforms the images
#     anchorIm = polarIm(anchorIm)
#     positiveIm = polarIm(positiveIm)
#     negativeIm = polarIm(negativeIm)
    
# return anchorIm, positiveIm, negativeIm