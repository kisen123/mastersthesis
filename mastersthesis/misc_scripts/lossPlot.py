# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:13:10 2023

@author: kisen
"""

import torch
import numpy as np

from matplotlib import pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')



def lossPlot(lossFlag=None, batchSize=None, plotPerformance=None, 
             performanceXVector=None, plotXVector=None, plotTrainLossVector=None, 
             plotValLossVector=None, performanceMeasures=None, 
             datasetCardinality=None, epoch=None, performance=None,
             initialize=False, firstPoint=False, plotInter=False,
             fig=None, ax1=None, ax2=None, FARThreshold=10-3):
    
    # Instancing marker options for the different performance measures
    markers = ['$A$', '$P$', '$R$', '$F1$', '$TAR$', '.']
    performanceMeasuresListInit = ["Accuracy", "Precision", "Recall", "F1-score", 
                              "TAR", "AUC"]
    performanceMarkers = {performanceMeasure: marker for performanceMeasure, marker in zip(performanceMeasuresListInit, markers)}
    
    
    # Adjusting the TAR to have its corresponding FARThreshold for plotting
    # purposes
    performanceMeasuresPlot = performanceMeasures.copy()
    if 'TAR' in performanceMeasuresPlot:
        TARIndex = performanceMeasuresPlot.index("TAR")
        performanceMeasuresPlot[TARIndex] = "TAR @ FAR(" + str(FARThreshold) + ")"
    
    if initialize:
        
        numSameTrain, numDiffTrain = datasetCardinality[0]
        numSameVal, numDiffVal = datasetCardinality[1]
        
        # get_ipython().run_line_magic('matplotlib', 'qt5')
        
        if plotInter == False:
            plt.ioff()
        
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
                                       sharex=True, figsize=(12,8))
        
        
        fig.suptitle("Loss function: " + lossFlag, 
                     fontsize=16, fontweight='bold')
        
        ax1.grid(True)
        ax2.grid(True)
        
        ax1.set_title("Loss over a minibatch (Batch-size: "+str(batchSize)+")")
        ax1.set_ylabel("Averaged minibatch loss")
        
        ax2.set_title("Data " +\
                      str(performanceMeasuresPlot) +\
                      " of varying numbers of epochs." +\
                          
                      # Training cardinality    
                      "\n-    Training cardinality: |" +\
                              r"$\mathcal{P}_{same}$" +\
                                      "| = " + str(numSameTrain) +\
                                          "    -    |" +\
                                              r"$\mathcal{P}_{diff}$" +\
                                          "| = " +\
                                              str(numDiffTrain) +\
                                              "    -" +\
                      # Validation cardinality
                      "\n-    Validation cardinality: |" +\
                          r"$\mathcal{P}_{same}$" +\
                                  "| = " + str(numSameVal) +\
                                      "    -    |" +\
                                          r"$\mathcal{P}_{diff}$" +\
                                      "| = " +\
                                          str(numDiffVal) +\
                                          "    -" )
        ax2.set_ylabel(str(performanceMeasuresPlot))
        
        # Defining legends
        ax1.legend(["Training loss per gradient step",
                    "Validation loss per gradient step"])
        
        ax2.legend(["Training " + str(performanceMeasuresPlot),
                    "Validation " + str(performanceMeasuresPlot)])
        
        fig.text(0.5, 0.04, "Number of completed epochs", ha='center',
                 va='center')
        
        fig.subplots_adjust(hspace=0.4)
        
        return fig, ax1, ax2
        
    else:
        
        
        thresholdTrainPerformance, thresholdValPerformance = performance
        plotTrainPerformance, plotValPerformance = plotPerformance
    


        lowerTrainPerformanceBounds = torch.min(plotTrainPerformance, dim=2)[0]
        upperTrainPerformanceBounds = torch.max(plotTrainPerformance, dim=2)[0]
        lowerValPerformanceBounds   = torch.min(plotValPerformance,   dim=2)[0]
        upperValPerformanceBounds   = torch.max(plotValPerformance,   dim=2)[0]
        
        plotTrainPerformanceErrors = torch.cat((lowerTrainPerformanceBounds.unsqueeze(1),
                                               upperTrainPerformanceBounds.unsqueeze(1)), dim=1)
        plotValPerformanceErrors   = torch.cat((lowerValPerformanceBounds.unsqueeze(1), 
                                               upperValPerformanceBounds.unsqueeze(1)), dim=1)
        
        
        lowerTrainLossBounds, _ = torch.nan_to_num(plotTrainLossVector, nan=0).min(dim=0)
        upperTrainLossBounds, _ = torch.nan_to_num(plotTrainLossVector, nan=0).max(dim=0)
        lowerValLossBounds, _   = torch.nan_to_num(plotValLossVector, nan=0).min(dim=0)
        upperValLossBounds, _   = torch.nan_to_num(plotValLossVector, nan=0).max(dim=0)
        
        plotTrainLossError = torch.cat((lowerTrainLossBounds.unsqueeze(0), upperTrainLossBounds.unsqueeze(0)))
        plotValLossError   = torch.cat((lowerValLossBounds.unsqueeze(0), upperValLossBounds.unsqueeze(0)))
        
        
        
        
        
        ax1.set_xlim([0, epoch+1])
        ax2.set_xlim([0, epoch+1])
        
        if firstPoint:
            
            # Training loss
            ax1.plot(plotXVector, torch.nan_to_num(plotTrainLossVector, nan=0).mean(dim=0),
                         marker='.', linestyle="--", color='tab:blue')
            
            
            ax1.plot(plotXVector, torch.nan_to_num(plotValLossVector, nan=0).mean(dim=0),
                     marker='.', linestyle="--", color='tab:orange')
            
            
            # Looping over the different performance measures
            for trainPerformance, valPerformance, performanceMeasure, plotTrainPerformanceError, plotValPerformanceError \
            in zip(plotTrainPerformance, plotValPerformance, performanceMeasures,
                   plotTrainPerformanceErrors, plotValPerformanceErrors):
                
                # Training accuracy
                ax2.plot(plotXVector, torch.mean(trainPerformance),
                             marker=performanceMarkers[performanceMeasure],
                             linestyle="-", color='tab:blue', markersize=5)
                
                
            
                # Validation accuracy
                ax2.plot(plotXVector, torch.mean(valPerformance),
                             marker=performanceMarkers[performanceMeasure],
                             linestyle="-", color='tab:orange', markersize=5)
                
            
            if plotInter:
                plt.show(block=False)
                plt.pause(0.001)
            
            
    
        else:
            
            # Training loss
            ax1.plot(plotXVector, torch.nan_to_num(plotTrainLossVector, nan=0).mean(dim=0),
                     marker='.', linestyle="--", color='tab:blue')
            
            ax1.fill_between(plotXVector[epoch-2:epoch], plotTrainLossError[0, epoch-2:epoch], plotTrainLossError[1, epoch-2:epoch],
                             color='tab:blue', alpha=0.4)
            
            
            ax1.plot(plotXVector, torch.nan_to_num(plotValLossVector, nan=0).mean(dim=0),
                     marker='.', linestyle="--", color='tab:orange')
            ax1.fill_between(plotXVector[epoch-2:epoch], plotValLossError[0, epoch-2:epoch], plotValLossError[1, epoch-2:epoch],
                             color='tab:orange', alpha=0.4)
            
            # Looping over the different performance measures
            for trainPerformance, valPerformance, performanceMeasure, plotTrainPerformanceError, plotValPerformanceError \
            in zip(plotTrainPerformance, plotValPerformance, performanceMeasures,
                   plotTrainPerformanceErrors, plotValPerformanceErrors):
                
                # Training accuracy
                ax2.plot(plotXVector, torch.nanmean(trainPerformance, dim=1),
                         marker=performanceMarkers[performanceMeasure],
                         linestyle="-", color='tab:blue', markersize=5)
                ax2.fill_between(plotXVector[epoch-2:epoch], plotTrainPerformanceError[0, epoch-2:epoch], plotTrainPerformanceError[1, epoch-2:epoch],
                                 color='tab:blue', alpha=0.4)
                
                # Validation accuracy
                ax2.plot(plotXVector, torch.nanmean(valPerformance, dim=1),
                         marker=performanceMarkers[performanceMeasure],
                         linestyle="-", color='tab:orange', markersize=5)
                ax2.fill_between(plotXVector[epoch-2:epoch], plotValPerformanceError[0, epoch-2:epoch], plotValPerformanceError[1, epoch-2:epoch],
                                 color='tab:orange', alpha=0.4)
            
            if plotInter:
                plt.draw()
                plt.pause(0.001)
                
            # Defining legends
            ax1.legend(["Training loss per epoch",
                        "Validation loss per epoch"])
            
            ax2.legend(["Training " + str(performanceMeasuresPlot),
                        "Validation " + str(performanceMeasuresPlot)])

        return fig, ax1, ax2
    
            
            
            
            