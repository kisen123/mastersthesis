# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:20:54 2023

@author: kisen
"""
# Importing pretty standard libraries
from matplotlib import pyplot as plt

import random
import torch


# Imports of PCA-scripts for visualization purposes
from visualize_scripts.PCAScripts import PCALearn, PCATransformData

# Importing tSNE
from sklearn.manifold import TSNE

# Importing script to show the images as one hovers over its plotted point
from misc_scripts.scatter_plot_with_hover import scatter_plot_with_hover

# Imports to plot a canvas
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')




def evaluationPlot(evalLoader, numClassesPlot, model, device, dataFlag,
                   datasetTransform="tSNE"):
    
    
    # Make the plot interactive
    get_ipython().run_line_magic('matplotlib', 'qt5')            
    
    # We collect all the images from the dictionary, such that the PCA
    # space is learned on this
    evalDataset = evalLoader.dataset.dataset
    
    # To keep the plot nice and simple, we only plot a number of unique IDs.
    if numClassesPlot > len(evalDataset):
    
        print("There is only a number of " + str(len(evalDataset)) +
              " classes. This is therefore the selected number of classes to plot")
        numClassesPlot = len(evalDataset)
    
    indicesToPlot = random.sample(
        range(len(evalDataset)), numClassesPlot)
    
    keysToBePlotted = \
        [list(evalDataset.keys())[randomIdx]
         for randomIdx in indicesToPlot]
    
    imagesPlot = \
        torch.cat([evalDataset[key] for key in keysToBePlotted])
    
    listOfIDs = []
    for key in keysToBePlotted:
    
        numUniqueIDs = evalDataset[key].shape[0]
    
        for uniquePic in range(0, numUniqueIDs):
            listOfIDs.append(key)
    
    # The randomly chosen keys are the data that is to be plotted.
    imagesTensor = torch.cat(list(evalDataset.values()), dim=0)
    
    if evalLoader.dataset.imageFlag == True:
        imagesTensor = imagesTensor.unsqueeze(1)
        imagesPlot = imagesPlot.unsqueeze(1)
    
    
    
    
    # We run the model on the images, creating embeddings.
    embeddedOutput = model.embeddingNet(imagesTensor.to(device))
    
    
    
    
    
    # The select few points that are to plotted are forward passed
    # again, I am no Python wizard.
    plotEmbeddedOutput = model.embeddingNet(imagesPlot.to(device))
    
    transformedData, dataBar, V, S = PCALearn(embeddedOutput,
                                              device=device)
    
    # The transformedEmbeddings variable now contains the PCA-transformed
    # data, ready for plotting.
    transformedEmbeddings = PCATransformData(plotEmbeddedOutput, dataBar,
                                             V, device=device)
    
    
    
    
    
    
    
    
    
    # -----------------------------------------------------------------
    # PLOTS TO SHOW INFORMATION ABOUT THE PCA TRANSFORMATION
    # We also show how the singular values, and how much of the dataset
    # variance they capture.
    numDims = min(embeddedOutput.shape[-1], embeddedOutput.shape[-2])
    dimsVector = torch.linspace(1, numDims, numDims)
    
    singValFig, (ax1, ax2) = plt.subplots(1,2)
    ax1.semilogy(dimsVector, S.cpu(), '-o', color='k')
    ax1.grid(True)
    ax1.set_title("Singular values per PCA dimension",
                  fontweight="bold", size=8)
    
    ax2.plot(dimsVector, torch.cumsum(S.cpu(), dim=0) /
             torch.sum(S.cpu()), '-o', color='k')
    ax2.grid(True)
    ax2.set_title(
        "Percentage of variance \ncaptured by the singular values", fontweight="bold", size=8)
    
    
    # PLOTS TO SHOW INFORMATION ABOUT THE PCA TRANSFORMATION
    # -----------------------------------------------------------------
    
    
    
    



    #### PCA ####
    # Generate data x, y for scatter and an array of images.
    PCAx, PCAy = transformedEmbeddings[:, 0].cpu().numpy(), \
                 transformedEmbeddings[:, 1].cpu().numpy()
           
    PCAAxis1Name, PCAAxis2Name = "PC1", "PC2"
    #### PCA ####






    #### t-SNE ####
    # Instantiating a TSNE object, by default it gives two dimensions for
    # visualization.
    # tsne = TSNE(perplexity=min(len(transformedEmbeddings), 30))
    
    tsne = TSNE(perplexity=min(len(transformedEmbeddings), 10))
    
    # Note that the PCA-transformed embeddings are tSNE-fitted. As PCA
    # is a linear transformation, it should not affect the overall 
    # performance of the tSNE-mapping.
    X = transformedEmbeddings.cpu().numpy()
    tSNE_X = tsne.fit_transform(X)
    
    # x, y are overwritten.
    tSNEx, tSNEy = tSNE_X[:, 0], tSNE_X[:, 1]
    
    tSNEAxis1Name, tSNEAxis2Name = "tSNE-1", "tSNE-2"
    #### t-SNE ####
    
    
    
    
    
    
    # Making the colors for the different IDs
    customColorsPointwise = []
    customColors = []
    currentID = "placeholder"
    for idx, ID in enumerate(listOfIDs):
    
        if ID == "UNKNOWN":
            random.seed(0)
        else:
            # For one ID, we set the unique seed
            random.seed(int(ID))
        r, g, b = round(random.uniform(0, 1), 3),\
            round(random.uniform(0, 1), 3),\
            round(random.uniform(0, 1), 3)
    
        customColorsPointwise.append((r, g, b))
    
        # We also need the set of colors in order.
        if currentID != ID:
            customColors.append((r, g, b))
            currentID = ID
            
    
    # Instantiating the embedding plot figures.
    PCAScatterFig, PCAScatterAx = scatter_plot_with_hover(PCAx, PCAy, imagesTensor, customColorsPointwise)
    tSNEScatterFig, tSNEScatterAx = scatter_plot_with_hover(tSNEx, tSNEy, imagesTensor, customColorsPointwise)
    
    
    
    # Misc. plotting options.
    PCALabel = "PCA-transformed embeddings of input images (" + dataFlag + ")"
    PCAScatterAx.set_title(label=PCALabel, fontweight="bold", loc='center', pad=None)
    
    tSNELabel = "tSNE-transformed embeddings of input images (" + dataFlag + ")"
    tSNEScatterAx.set_title(label=tSNELabel, fontweight="bold", loc='center', pad=None)
    
    PCAScatterAx.set_xlabel(PCAAxis1Name, style="italic"), PCAScatterAx.set_ylabel(PCAAxis2Name, style="italic")
    tSNEScatterAx.set_xlabel(tSNEAxis1Name, style="italic"), tSNEScatterAx.set_ylabel(tSNEAxis2Name, style="italic")
    
    PCAScatterAx.grid(True)
    tSNEScatterAx.grid(True)
    
    # We reset the model to be in train-mode after evaluations.
    model.train()
    

    return singValFig, PCAScatterFig, tSNEScatterFig
