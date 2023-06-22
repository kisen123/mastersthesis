# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:28:44 2023

@author: kisen
"""
# Code that MIGHT be useful
#%%
# Creating some helper functions
def imshow(img, text1=None, text2=None):
    npimg = img.numpy()
    plt.axis("off")
    if text1:
        plt.text(90, 70, text1, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    if text2:
        plt.text(0, 70, text2, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
        
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

trainPerformDataset = SiameseDataset(valDictionary, 
                                     "trainPerform", mode="evaluation",
                                     lossFunction="Contrastive")


# Grab one image that we are going to test
dataiter = iter(trainPerformDataset)
for i in range(3):
    
    for im1, label1, im2, label2, flag in trainPerformDataset:
    


        if flag == 0:
            text2 = "Genuine"
        else:
            text2 = "Impostor"
        
        # Concatenate the two images together
        concatenated = torch.cat((im1.unsqueeze(0), im2.unsqueeze(0)), 0)
        
        output1, output2 = modelTest(im1.cuda().unsqueeze(1), im2.cuda().unsqueeze(1))
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated),
               f'Dissimilarity: {euclidean_distance.item():.2f}',
               f'ID: {text2}')
#%% SECTION FOR SIAMESE TRAINING OF COLUMN SUMMED FEATURES


# Device and model is set
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

model = FourLayerNet(input_size=32, hidden_size1=256,
                   hidden_size2=512, hidden_size3=128,
                   output_size=8).to(device)
setSeed(123)


# The datasets are set
trainDataset = SiameseDataset(trainColData, 
                              "train", mode="training", 
                              lossFlag="Triplet", imageFlag=False)

trainPerformDataset = SiameseDataset(valColData, 
                                     "trainPerform", mode="evaluation",
                                     lossFlag="Triplet", imageFlag=False)

valDataset = SiameseDataset(testColData, 
                            "val", mode="evaluation",
                            lossFlag="Triplet", imageFlag=False)

# NB! AS OF NOW, mode=evaluation WON'T DO ANYTHING GOOD, THE CORRECT EVALUATION
# SCHEME IS DONE INSIDE THE evalSiamese FUNCTION!

# The train loader is set
trainLoader = DataLoader(trainDataset, batch_size=hyperParametersSiamese[0]["batchSize"], shuffle=True)

# The validation loader is set
valLoader = DataLoader(valDataset, batch_size=1,
                       shuffle=False)

# A train performance loader is set to evaluate the model in between
# parameter updates and epochs.
trainPerformLoader = DataLoader(trainPerformDataset, 
                                batch_size=1,
                                shuffle=False)


# The optimizer is defined
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.01)
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# KOM TILBAKE ADAM OPTIMIZER

modelsEpochs = trainSiamese(model, "Triplet", hyperParametersSiamese[0], 
                            hyperParametersSiameseTicks,
                            trainLoader, trainPerformLoader, valLoader, 
                            optimizer, device, saveInterEpoch=False, 
                            performanceShadow=True, learningVisualizationEpoch=False,
                            learningVisualizationStep=False)


#%% Simple K-means
trainImgsFlattened = torch.cat(list(trainDictionary.values()), dim=0).reshape(28,-1)
listOfIDs = list(trainDictionary.keys())


labelsList = []

labels = [label for i, label in enumerate(listOfIDs) for _ in range(int(len(list(trainDictionary.values())[i])))]
numClusters = len(listOfIDs)

from sklearn.cluster import KMeans

kmeans= KMeans(n_clusters=numClusters, random_state=0)

kmeans.fit(trainImgsFlattened)
cluster_assignments = kmeans.labels_

cluster_centers = kmeans.cluster_centers_


# PCA Transform the images
transformedData, dataBar, V, _ = PCALearn(trainImgsFlattened, 'cpu')

plot2DPCA(transformedData, listOfIDs, labels)
#%% This section is for saving models
# load pickle module
import pickle



# create a binary pickle file 
f = open("modelFile.pkl","wb")

# write the python object (dict) to pickle file
pickle.dump(modelTest, f)

# close file
f.close()

#%% This section is for loading models
# LOAD HERE



#%% This section plots up the PCA 2D space for the trained embeddings

# This function sets up the PCA-space, and returns variables needed to rebuild it
def PCALearn(trainingData):
    
    # Restructuring the data tensor
    # imgs = torch.permute(trainingData, (2,0,1))
    # imgsFlatten = imgs.flatten(1)
    imgsFlatten = trainingData
    
    # Useful variables
    nPoints, _ = imgsFlatten.shape
    onesVector = torch.ones(nPoints, 1)
    
    # Preprocessing the dataset by removing the mean observation
    imgsFlattenAvg = torch.mean(imgsFlatten, dim=0).unsqueeze(0)
    dataBar = matmul(onesVector, imgsFlattenAvg)
    B = imgsFlatten - dataBar
    
    # Doing the Singular Value Decomposition on the dataset
    _, _, Vt = torch.linalg.svd(B, full_matrices=False)
    
    # Transposing the Vtranspose matrix
    # V = Vt.mH
    V = torch.transpose(Vt, 0, 1)
    
    principalComponents = matmul(B, V)
    
    return principalComponents, dataBar[0,:].unsqueeze(0), V

# To predict on validation data, we must first transform the validation data to PCA space
# To do that, we need to transform the data in the same way 
def PCATransformData(evaluationData, dataBarTrained, V):
    
    # Restructuring the data tensor
    imgs = torch.permute(evaluationData, (2,0,1))
    imgsFlatten = imgs.flatten(1)
    
    # Useful variables
    nPoints, _ = imgsFlatten.shape
    onesVector = torch.ones(nPoints, 1)
    
    # Broadcasting the mean row
    dataBarTrained = matmul(onesVector, dataBarTrained)
    
    # Preprocessing the dataset by removing the mean observation from the 
    # training data of the PCA transformation.
    
    B = imgsFlatten - dataBarTrained
    
    principalComponents = matmul(B, V)
    
    # The rows are the samples, and the columns are the principal component features
    return principalComponents

#%% This small section runs the classifiers and the PCA-transformation and learning functions
# modelTest = modelsEpochs["NumEpochs: 20"][2]
modelTestCPU = modelTest.cpu()
X1 = trainingDataset[:][0][:, :, :, 0]
X2 = trainingDataset[:][0][:, :, :, 1]
output1, output2 = modelTestCPU(X1.unsqueeze(1).cpu(), X2.unsqueeze(1).cpu())
#%%
# trainingData is the tensor containing all the training data images (69,69,nTrainSamples)
transformedData, dataBar, V = PCALearn(torch.cat((output1, output2), dim=0))
#transformedEvalData = PCATransformData(validationData, dataBar, V)

#%% This section plots a dataset in PCA-space


def plot2DPCA(evalData, listOfIDs, labels):
    
    transformedData, dataBar, V, S = PCALearn(evalData, 
                                           device='cpu')
    
    # The transformedEmbeddings variable now contains the PCA-transformed
    # data, ready for plotting.
    # transformedEmbeddings = PCATransformData(evalData, dataBar,
    #                                          V, device='cpu')
    
    
    
    
    customColorsPointwise = []
    customColors = []
    currentID = "placeholder"
    for idx, ID in enumerate(labels):
        
        
        # For one ID, we set the unique seed
        random.seed(int(ID))
        r, g, b = round(random.uniform(0, 1), 3),\
                  round(random.uniform(0, 1), 3),\
                  round(random.uniform(0, 1), 3)
                  
                  
        customColorsPointwise.append((r,g,b))
        
        # We also need the set of colors in order.
        if currentID != ID:
            customColors.append((r,g,b))
            currentID = ID
        
    
    # We define the colors in a Patch-object to pass it to a legend
    # method.
    patches = [Patch(color=color, label=f"ID: {ID}") for color, ID \
               in zip(customColors, listOfIDs)]
    
    
    plt.scatter(transformedData[:, 0].cpu(), 
                transformedData[:, 1].cpu(), c=customColorsPointwise)
    
    plt.legend(handles=patches, bbox_to_anchor=(1, 1), fontsize=12, edgecolor='black')
    
    # Misc. plotting options.
    plt.title(label="PCA-transformed embeddings of input images", 
              fontweight="bold", loc='center', pad=None,)
    
    plt.xlabel("PC1", style="italic"); plt.ylabel("PC2", style="italic")
    plt.grid(True)
    # Show the plot
    plt.show()
    
    
    # We also show how the singular values, and how much of the dataset
    # variance they capture.
    numDims = min(transformedData.shape[-1], transformedData.shape[-2])
    dimsVector = torch.linspace(1, numDims, numDims)
    
    
    singValFig = plt.figure()
    ax1 = singValFig.add_subplot(121)
    ax1.semilogy(dimsVector, S.cpu(), '-o', color='k')
    ax1.grid(True)
    ax1.set_title("Singular values per PCA dimension", fontweight="bold", size=8)
    
    
    ax2 = singValFig.add_subplot(122)
    ax2.plot(dimsVector, torch.cumsum(S.cpu(), dim=0)/torch.sum(S.cpu()), '-o', color='k')
    ax2.grid(True)
    ax2.set_title("Percentage of variance \ncaptured by the singular values", fontweight="bold", size=8)
    
    plt.show()
    # PCAPlot()
#%%
evalData = torch.cat(list(trainDictionary.values()))
listOfIDs = list(trainDictionary)
plot2DPCA(evalData, listOfIDs)
#%%
plot2DPCA()
#%%
trainedModels = trainSiamese(model, hyperParametersSiamese, 
                             hyperParametersSiameseTicks, trainDataset,
                             optimizer, device)
             
#%%
# Next, we evaluate the model with accuracy
valDataset = SiameseDataset(validationDataset)
valLoader = DataLoader(valDataset, batch_size=len(valDataset), shuffle=False)

valPerformance = evalSiamese(model, valLoader, device, "Accuracy")

#%% Train accuracy
trainDataset = SiameseDataset(trainingDataset[0:600])
trainLoader = DataLoader(trainDataset, batch_size=600, shuffle=False)

trainAccuracy = evalSiamese(model, trainLoader, device, "Accuracy")


#%% This section searches over the specified hyperparameters, and picks the best one in model selection


# Setting the device to train on
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

# Setting the validation loader
valDataset = SiameseDataset(validationDataset)
valLoader = DataLoader(valDataset, batch_size=len(valDataset), shuffle=False)

# Making an empty dictionary for the models to be contained in.
modelsDictionary = {}

# The hyperparameters are searched...
for i, parameters in enumerate(hyperParametersSiamese):
    print("\nProcessing hyperparameters:", parameters, '...')
    
    # SET A SEED FOR THE SIAMESEDATASETCLASS KOM TILBAKE, ALSO SET LOSSFUNC
    # TO BE A HYPERPARAMETER
    lossFunc = ContrastiveLoss().to(device=device)
    
    
    # Setting up the training data, because batch size is a hyperparameter to 
    # be tuned for the Siamese network approach, it has to be defined 
    # inside the for loop.
    trainDataset = SiameseDataset(trainingDataset)
    trainLoader = DataLoader(trainDataset,
                              batch_size=parameters["batchSize"],
                              shuffle=False)
    
    
    # Controlling the randomization by setting fixed seeds. Among things, this
    # fixes initialization of weights for neural networks, and cluster centers 
    # for k-means, for independent code runs.
    torch.manual_seed(i)
    
    # Instantiating the model
    model = SiameseNetResnet18().to(device=device)
    
    # Setting the optimizer along with its hyperparameters
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=0.001)
    
    # The instantiated model is trained
    trainSiamese(model, parameters["epochs"], trainLoader,
                 optimizer, device, parameters)
    
    
    # Evaluating the model on the validation data. Note that a range of thresholds
    # have been tested to yield the best validation accuracy. KOM TILBAKE, DETAILS
    # MUST BE FLESHED OUT.
    # NEEDS DISSIMILARITY FUNCTION AS HYPERPARAMETERINPUT KOM TILBAKE
    thresholdAccuracy = evalSiamese(model, valLoader, device) #, (dissimFunc)
    
    
    # Retrieveing the validation accuracy of the model, and storing the model
    # alongside the accuracy. Also, the ticks are stored for plotting purposes,
    # and the classifierType is needed for testing purposes.
    modelsDictionary[str(parameters)] = (thresholdAccuracy, model,
                                         hyperParametersSiameseTicks[i])
    
    print("Model search progression: "  + str(i+1) + " / " + str(len(hyperParametersSiamese)))
    
#%%
def plotClassifiers(modelsDictionary, chooseOwnIds=False, allModels=False):

    numClassifiers = len(modelsDictionary)
    
    # Sorting the models based on their validation accuracies
    sortedModels = sorted(modelsDictionary.items(), key=lambda x: x[1][0])
    
    # This if-statement plots all the models 
    if allModels == True:
        
        # Defining the data to be plotted.
        allModelAccuracies = [modelTuple[1][0] for modelTuple in sortedModels]
        modelTicks = [str(i) for i in range(0,numClassifiers)]
        
        # Make a user-defined colormap.
        customCM = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])
        
        # Making a normalizer to scale the model colors.
        cNorm = mcol.Normalize(vmin=0,
                               vmax=numClassifiers)
        
        # Scaling the colors based on how many models there are.
        cPick = cm.ScalarMappable(norm=cNorm, cmap=customCM)
        cPick.set_array([])
        
        # Plotting each bar with its own unique gradient-like color, and
        # setting the x-tick appearance frequency
        fig, ax = plt.subplots(figsize=(10,6))
        
        myLocator = mticker.MultipleLocator(20)
        ax.xaxis.set_major_locator(myLocator)
        for i, modelAccuracy in enumerate(allModelAccuracies):
            plt.bar(x=modelTicks[i], 
                    height=modelAccuracy,
                    color=cPick.to_rgba(i))
        
        # Setting a color gradient bar.
        plt.colorbar(cPick, label="Model index by color")
        
        # Misc. plotting options
        plt.ylim(allModelAccuracies[0]-0.005, allModelAccuracies[-1]+0.005)
        plt.grid(True, axis='y')
        plt.xlabel("Model index by performance on the validation data", 
                   fontweight='bold', fontsize=15)
        plt.ylabel("Accuracy on validation data", fontstyle='italic', fontsize=15)
        
        plt.suptitle("All model validation accuracies",
                     fontweight='extra bold', fontsize=30)
        
        plt.show()
        
    else:
        # Default chooses the 4 best and 4 worst classifiers
        if chooseOwnIds==False:
            listOfBests = list(np.linspace(numClassifiers-4, numClassifiers-1, 
                                      4, dtype=int))
            listOfWorsts = list(np.linspace(0, 3, 4, dtype=int))
            
            
            
        # Lets the user choose the 4 best and 4 worst classifiers
        elif chooseOwnIds==True:
            
            print("To avoid cluttering in the plot, a maximum of 4 classifiers")
            print("is a good rule of thumb.")
            print(" ")
            print(" ")
            print("The best classifier is index: " + str(numClassifiers-1))
            print("The worst classifier is index: " + str(0))
            print(" ")
            print("With that in mind, please choose: ")
            
            listOfBests = [int(item) for item in \
                           input("Best model indices (example: 90  92  93  97): ")\
                               .split()]
                
            listOfWorsts = [int(item) for item in \
                            input("Worst model indices (example: 0  3  4  6): ")\
                                .split()]
    
            
        # Slicing the best/worst models
        bestModelsList = [sortedModels[idxModel] for idxModel in listOfBests]
        worstModelsList = [sortedModels[idxModel] for idxModel in listOfWorsts]
        
        # Retrieving the parameter-ticks and validation accuracies.
        bestModelsParameters = [modelTuple[1][2] for modelTuple in bestModelsList]
        bestModelsAccuracies = [modelTuple[1][0] for modelTuple in bestModelsList]
        worstModelsParameters = [modelTuple[1][2] for modelTuple in worstModelsList]
        worstModelsAccuracies = [modelTuple[1][0] for modelTuple in worstModelsList]
        
        # The bar chart expects the x-ticks to be i the form of dictionaries
        bestModelsDict = {}
        for i in range(0, len(listOfBests)):
            bestModelsDict[str(bestModelsParameters[i])] = bestModelsAccuracies[i]
            
        worstModelsDict = {}
        for i in range(0, len(listOfWorsts)):
            worstModelsDict[str(worstModelsParameters[i])] = worstModelsAccuracies[i]
        
        
        # This section plots the models with the best/worst validation accuracies.
        
        # Setting the x-ticks for the best/worst models
        bestModels = bestModelsDict
        xTicksBest = list(bestModels.keys())
        bestValAccuracies = list(bestModels.values())
        
        worstModels = worstModelsDict
        xTicksWorst = list(worstModels.keys())
        worstValAccuracies = list(worstModels.values())
        
        # Instantiating a figure.
        fig, ax = plt.subplots(figsize = (20, 10))
        
        # Creating the best/worst validation accuracy bar plots.
        plt.bar(xTicksWorst, worstValAccuracies, color='red',
                width=0.4) 
        plt.bar(xTicksBest, bestValAccuracies, color='blue',
                width=0.4)
        
        
        # Scaling the shown accuracies, and the x-ticks.
        plt.ylim((worstValAccuracies[0]-0.005, bestValAccuracies[-1]+0.005))
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        # Misc. plotting options
        plt.grid(True, axis='y')
        plt.xlabel("Hyperparameter options", 
                   fontweight='bold', fontsize=25)
        plt.ylabel("Accuracy on validation data", fontstyle='italic', fontsize=20)
        
        if chooseOwnIds==False:
            plt.suptitle("Best " + str(len(listOfBests)) + \
                     " validation accuracies (blue) & worst " + \
                         str(len(listOfWorsts)) + " validation accuracies (red)",
                     fontweight='extra bold', fontsize=30)
                
        elif chooseOwnIds==True:
            plt.suptitle("The best models by rank: " + str(abs(np.array(listOfBests)\
                                                               -numClassifiers)) + \
                     " validation accuracies (blue) & worst models by rank: " + \
                         str(abs(np.array(listOfWorsts)-numClassifiers)) + \
                         " validation accuracies (red) out of " + str(numClassifiers) \
                             + " classifiers.", fontweight='extra bold', fontsize=23)
        
        plt.show()
        
        bestTypeModel = bestModelsList[-1]
        return bestTypeModel

#%%
plotClassifiers(modelsEpochs)

#%%
plotClassifiers(modelsDictionary, allModels=True)
#%%
def trainFunction(CNN, numEpochs, trainLoader, optimizer, device):
    
    # Defining a couple of standard variables
    lossesTrain = []
    sizeBatch = len(trainLoader)
    
    # Letting the CNN know that it is in train-mode
    CNN.train()
    
    optimizer.zero_grad(set_to_none=True)
    print("\nPlease wait... the model is now being trained...\n")
    for epoch in range(1, numEpochs + 1):
        lossTrain = 0.0
        for trainImgs, trainLabels in trainLoader:
            
            # Making full use of my newly acquired GPU :)
            trainImgs = trainImgs.to(device=device, dtype=torch.float)
            trainLabels = trainLabels.to(device=device, dtype=torch.long)
            
            # Doing a forward pass
            batchOutputs = CNN(trainImgs.unsqueeze(1))
            
            # LOSS FUNCTION CALCULATIONS
            #batchLoss = ceLoss(batchOutputs[:,:,0,0], trainLabels)
            batchLoss = ceLoss(batchOutputs, trainLabels)
            
            batchLoss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            lossTrain += batchLoss.item()
        lossesTrain.append(lossTrain / sizeBatch)
        print('{}  |  Epoch {}  |  Training loss {:.3f}'.format(
            datetime.now().time(), epoch, lossTrain / sizeBatch))
    return lossesTrain


#%% This section has the computeAccuracy function
def computeAccuracy(CNN, evalLoader, device):
    
    # Setting the number of batches in the evalLoader dataset
    numBatches = len(evalLoader)
    
    # Setting the model to evaluation mode
    CNN.eval()
    batchPerformances = 0.0
    wrongs = []
    with torch.no_grad():
        sumClassificationAccuracy = 0
        
        # 
        for evalImgs, evalLabels in evalLoader:
            batchSize = evalImgs.size(0)
            
            evalImgs = evalImgs.to(device=device, dtype=torch.float)
            evalLabels = evalLabels.to(device=device)
            
            # Forward-prop of the evaluation images.
            batchOutputs = CNN(evalImgs.unsqueeze(1))
            
            # Retrieving the label as indices.
            predictedLabels = torch.argmax(batchOutputs, dim=1)
            #batchTrues = predictedLabels[:,0,0] == evalLabels
            batchTrues = predictedLabels == evalLabels
            
            wrongs.append(list(np.asarray(batchTrues.cpu())))
            
            batchPerformances += torch.sum(batchTrues) / batchSize
            
        sumClassificationAccuracy = batchPerformances.item() / numBatches
        wrongs = [item for batchWrongs in wrongs for item in batchWrongs]
        wrongsIdx = [i for i, booleanCheck in enumerate(wrongs) if not booleanCheck]
        
        return wrongsIdx, sumClassificationAccuracy

#%% This section fits the model
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")
        
#testModel = LN5_var1.to(device=device)
testModel2 = SiameseNet().to(device=device)

trainLoader = DataLoader(trainingDataset, batch_size=4, shuffle=False)

# The seed is set
torch.manual_seed(123)

# Instantiating loss functions and activation functions
ceLoss = nn.CrossEntropyLoss().to(device=device)

# The optimizer is defined
optimizer = optim.SGD(testModel2.parameters(), lr=0.01, momentum=0.4, weight_decay=0.001)
        
trainSiamese(testModel2, 30, trainLoader, optimizer, device)
        
        
#%% This section evaluates the model
model = testModel2
valLoader = DataLoader(validationDataset, batch_size=4, shuffle=False)

# Checking the validation accuracy of the trained model
wrongsIdx, valAccuracy = computeAccuracy(model, valLoader, device)



#%% This section shows the softmax activation output of a user-given datapoint
# It also shows an example image of where the classification went wrong.

logSoftmax = nn.LogSoftmax(dim=1).to(device=device)
sigmoid = nn.Sigmoid().to(device=device)
softmax = nn.Softmax(dim=1).to(device=device)

print("The wrong classification indices from the validation data are: " + str(wrongsIdx))
userIndex = int(input("Please choose a datapoint to see the softmax output of: "))
dataPoint = standValData[userIndex,:,:].unsqueeze(0).unsqueeze(0)

# Doing a forward pass
modelPass = model(dataPoint.to(device=device, dtype=torch.float)).cpu()
histoGram = np.asarray(softmax(modelPass)[0,:,0,0].detach())

inClasses = np.linspace(0,9,10)

fig1, ax1 = plt.subplots()
bar1 = ax1.bar(inClasses, histoGram)
ax1.set_title("Softmax of forward pass of image in database")

# We show a figure of what the model predicted, giving an image example of the
# predicted, wrong class.
predictedClass = torch.argmax(softmax(modelPass)[0,:,0,0].detach())
examplePredictIdx = np.where(validationLabels == predictedClass.item())[0][0]
plt.figure()
plt.imshow(unrotatedValidation[:, :, examplePredictIdx], cmap='gray', vmin=0, vmax=1)
plt.title("The model predicted this to be class: " + str(predictedClass.item()) + ". This class looks like this:")

# We show a figure of what the model should have predicted, giving an image 
# example of how the correct class looks like.
actualClass = validationLabels[userIndex]
exampleActualIdx = np.where(validationLabels == actualClass)[0][0]
plt.figure()
plt.imshow(unrotatedValidation[:, :, exampleActualIdx], cmap='gray', vmin=0, vmax=1)
plt.title("The actual class of the datapoint is: " + str(actualClass) + ". This class looks like this:")


#%% This small section defines a dropout function for validating the model
def enableDropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
#%% This section defines the Monte Carlo dropout in evaluation mode
def MCPredictions(dataPoint, numSamples, model, device, plots=False, distPlot=True):
    
    # evalLoader = DataLoader(evalDataset, 
    #                         batch_size=evalDataset.tensors[0].shape[0],
    #                         shuffle=False)
    inClasses = np.linspace(0,9,10)
    softmax = nn.Softmax(dim=1).to(device=device)
    
    sumHistograms = np.zeros(10)
    for sample in range(numSamples):
        
        # Enabling evaluation and dropout on evaluation
        model.eval()
        enableDropout(model)
        
        dataPoint = dataPoint.to(device=device)
        output = model(dataPoint)
        histoGram = np.asarray(softmax(output)[0,:,0,0].detach().cpu())
        
        sumHistograms += histoGram
        if plots == True:
            fig1, ax1 = plt.subplots()
            bar1 = ax1.bar(inClasses, histoGram)
            ax1.set_title("Softmax of forward pass of image in database")
    
    distribution = sumHistograms / numSamples
    
    if distPlot == True:
        fig1, ax1 = plt.subplots()
        bar1 = ax1.bar(inClasses, distribution)
        ax1.set_title(f"Softmax of {numSamples} forward pass of image in database")

for numSampling in [100]:
    MCPredictions(dataPoint, numSampling, model, device)
#%% This section sets up the output layer activation function histogram thingamajig




# NOTE, leave out a class from the training set to see the histogram plots 
"""
logSoftmax = nn.LogSoftmax(dim=1).to(device=device)
sigmoid = nn.Sigmoid().to(device=device)
softmax = nn.Softmax(dim=1).to(device=device)

imNotDatabase = Image.open("9-1.tif")
imNotDatabase = convertToTensor(imNotDatabase)[0,:,:]
imNotDatabase = imNotDatabase.unsqueeze(0).unsqueeze(0)

imInDatabase = validationData[:,:,4].unsqueeze(0).unsqueeze(0)

inDatabasePass = model(imInDatabase)
notDatabasePass = model(imNotDatabase)

inHistogram = np.asarray(softmax(inDatabasePass)[0,:,0,0].detach())
outHistogram = np.asarray(softmax(notDatabasePass)[0,:,0,0].detach())

inClasses = np.linspace(0,8,9)

fig1, ax1 = plt.subplots()
bar1 = ax1.bar(inClasses, inHistogram)

fig2, ax2 = plt.subplots()
bar2 = ax2.bar(inClasses, outHistogram)

ax1.set_title("Softmax of forward pass of image in database")
ax2.set_title("Softmax of forward pass of image NOT in database")
"""
#%% This section show images as they are passed into the network

# We save the weights and the conv layers in these lists
modelWeights = []
networkLayers = []

# Extracting each layer in the model in a list
modelChildren = list(model.children())

# This for loop appends all the conv layers and the 
# respective weights
counter = 0

for layer in range(len(modelChildren)):
    
    counter += 1
        
    if type(modelChildren[layer]) == nn.Conv2d:
        modelWeights.append(modelChildren[layer].weight)
        
    networkLayers.append(modelChildren[layer])

# Printing useful information
print(f"Total network layers: {counter}")
print(networkLayers)


# Next, we do a forward pass of the network, piece by piece


# Defining some important variables and lists.
image = dataPoint.to(device=device)
outputs = []
names = []

convOutputs =[]

for layer in networkLayers:
    
    # Doing forward passes one layer at a time.
    image = layer(image)
    outputs.append(image)
    
    # We are especially interested in what happens in the 
    # convolution steps, so we store that in its own list.
    if type(layer) == nn.Conv2d:
        convOutputs.append(image)
        # MAYBE INFER THE ACTIVATION FUNCTION AS WELL HERE???
    
    names.append(str(layer))

#%%
for featureMaps in convOutputs:
    maxValue = torch.max(featureMaps)
    counter = 0
    
    # We calculate the number of images in the rows and columns of the 
    # figure.
    numSubplots = featureMaps.shape[1]
    numColumns = np.floor(np.sqrt(numSubplots))
    
    numRows = numSubplots // numColumns
    
    figSizeX = int(numColumns * 5)
    figSizeY = int(numRows * 5)
    
    fig = plt.figure(figsize=(figSizeX, figSizeY))
    
    if numSubplots % numColumns != 0:
        numRows += 1
    
    for featureMap in featureMaps[0,:,:,:]:
        
        # We add the plots of the feature maps as subplots.
        a = fig.add_subplot(int(numRows), int(numColumns), counter+1)
        
        imgPlot = plt.imshow(relu(featureMap).cpu().detach().numpy(), 
                             cmap='gray', vmin=0, vmax=maxValue)
        counter += 1
    fig.suptitle("This layer's max value: " + \
                 str(round(float(maxValue), 1)), 
                 fontsize=figSizeX*2)
#%%
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print(conv_layers)

image = dataPoint.to(device=device)
outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
print(len(outputs))#print feature_maps
for feature_map in outputs:
    print(feature_map.shape)
#%%   
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy()) 
    for fm in processed:
        print(fm.shape)

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
    
#%% TEST CODE


class SiameseDatasetTest(T.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        
        dictKey = list(self.data.keys())
        
        
        return self.data[dictKey[index]]
    
    def __len__(self):
        return len(self.data)
    
from torch.utils.data import DataLoader

# Create an instance of the SiameseDataset class
dataset = SiameseDatasetTest(trainDictionary)

# Create a data loader
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Iterate through the data points in the dataset
for data in data_loader:
    
    print(data)
    pass


#%%
def polar_transform(images, transform_type='linearpolar'):
    """
    This function takes multiple images, and apply polar coordinate conversion to it.
    """
    
    (N, C, H, W) = images.shape

    for i in range(images.shape[0]):

        img = images[i].numpy()  # [C,H,W]
        img = np.transpose(img, (1, 2, 0))  # [H,W,C]

        if transform_type == 'logpolar':
            img = cv2.logPolar(img, (H // 2, W // 2), W / math.log(W / 2), cv2.WARP_FILL_OUTLIERS).reshape(H, W, C)
        elif transform_type == 'linearpolar':
            img = cv2.linearPolar(img, (H // 2, W // 2), W / 2, cv2.WARP_FILL_OUTLIERS).reshape(H, W, C)
        img = np.transpose(img, (2, 0, 1))

        images[i] = torch.from_numpy(img)

    return images


convertToTensor = transforms.ToTensor()
testImage = convertToTensor( Image.open("0-1.tif") )[0].unsqueeze(0).unsqueeze(0)

polar_transform(testImage)


#%%
evalDataset = valLoader.dataset.dataset

listOfIDs = []
for key in evalDataset:
    
    numUniqueIDs = evalDataset[key].shape[0]
    
    for uniquePic in range(0, numUniqueIDs):
        listOfIDs.append(key)

#%%
tensor1 = torch.randn(2, 224, 224)
tensor2 = torch.randn(3, 224, 224)
tensor3 = torch.randn(5, 224, 224)

stacked_tensor = torch.cat([tensor1, tensor2, tensor3], dim=0)

print(stacked_tensor.shape)  # Output: (10, 224, 224)



#%% Section to plot column summation
testImage = trainStandDictionary["6"][0]

# For every colResolution-th column, we sum those columns together
colRes = 4


if testImage.shape[1] % colRes != 0:
    raise ValueError("The resolution must be divisible by the column dimension")

numColumns = testImage.shape[1] // colRes
reshapedImage = testImage.reshape(-1, colRes, numColumns)
colCollapsed = torch.sum(reshapedImage, axis=1)

fig, ax = plt.subplots(1, 2)

ax[0].imshow(testImage, cmap='gray', vmin=0, vmax=1)
ax[0].set_title("Collapsing columns together")

# Plotting the vertical column bars that are to be summed over
for rectangle in range(numColumns-1):
    ax[0].add_patch(patches.Rectangle(( colRes * rectangle-0.5, -1), colRes,
                                   testImage.shape[0]+1, fill=None, alpha=0.5,
                                   color='r'))
    
    ax[0].add_patch(patches.Arrow(colRes * rectangle, testImage.shape[0] // 2, colRes-1, 0, color='r'))


ax[0].add_patch(patches.Arrow(colRes * (rectangle+1), testImage.shape[0] // 2, colRes-1, 0, color='r'))
    
# The red grids marks where the columns are summed, and the arrows is what it
# conceptually looks like.
ax[1].imshow(colCollapsed, cmap='gray')
ax[1].set_title("Column-summed image")

#%% Collapsing the row dimension, revealing the pixel histogram with a number of numColumns bins
collapsedPixelHistogram = torch.sum(colCollapsed, axis=0)

plt.plot(list(range(numColumns)), collapsedPixelHistogram, '.', markersize=10)
plt.title("Column resolution: " + str(colRes) + ". Number of columns: " + str(numColumns))
plt.xlabel("Column no.")
plt.ylabel("Summed pixel intensity")
plt.grid(True)



#%% Code for feature extraction (features are summed columns!)


trainData = {}
valData = {}
testData = {}



#%%
for col in range(0, numSame):
    
    # Getting the image to plot
    imgToPlot = trainDictionary[louseID][col]
    
    # Misc. plotting options.
    ax[col].imshow(imgToPlot, cmap='gray', vmin=0, vmax=1)
    
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


summedRows = torch.sum(testImage, dim=0)
summedColumns = torch.sum()