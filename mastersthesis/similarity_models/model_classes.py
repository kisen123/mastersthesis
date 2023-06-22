# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:23:55 2023

@author: kisen
"""

import torch
from torch import nn
import torchvision.models as PyTorchModels
from torchinfo import summary


"""
Deformable convolution based of the paper:
    
    Deformable Convolutional Networks
    
    URL to paper:
    https://arxiv.org/abs/1703.06211
    
"""
# from DeformableConvs.deconv import DeformableConv2d



"""
Cylindrical convolution based of the paper: 
    
    CyCNN: A Rotation Invariant CNN using 
    Polar Mapping and Cylindrical Convolution Layers
    
    URL to paper:
    https://arxiv.org/pdf/2007.10588.pdf
"""
# from models.cyconvlayer import CyConv2d


# class Siamese_LeNet5_var(nn.Module):
#     def __init__(self, similarityFlag=False):
#         super(Siamese_LeNet5_var, self).__init__()
        
#         self.similarityFlag = similarityFlag
        
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
#                                stride=1, padding=2)
#         self.act1 = nn.Tanh()
#         self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)
#         self.dropout1 = nn.Dropout(p=0.2)
        
        
        
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,
#                                stride=1, padding=2)
#         self.act2 = nn.Tanh()
#         self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)
#         self.dropout2 = nn.Dropout(p=0.2)
        
        
        
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=5,
#                                stride=1, padding=2)
#         self.act3 = nn.Tanh()
#         self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.dropout3 = nn.Dropout(p=0.2)
        
        
        
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
#                                stride=1)
#         self.act4 = nn.Tanh()
        
#         self.adaptivePool = nn.AdaptiveAvgPool2d((2, 2))
#         self.dropout4 = nn.Dropout(p=0.2)
        
#         self.fc = nn.Sequential(nn.Linear(1024, 128))
        
        
        
#         self.layer1 = nn.Sequential(self.conv1, self.act1, self.pool1, self.dropout1)
#         self.layer2 = nn.Sequential(self.conv2, self.act2, self.pool2, self.dropout2)
#         self.layer3 = nn.Sequential(self.conv3, self.act3, self.pool3, self.dropout3)
#         self.layer4 = nn.Sequential(self.conv4, self.act4, self.adaptivePool)
        
#         self.embeddingNet = nn.Sequential(self.layer1, self.layer2, self.layer3,
#                                           self.layer4, nn.Flatten(), self.dropout4, self.fc)
        
        
#     def forward(self, x1, x2=None, x3=None, device='cuda'):
        
#         # One forward pass is guaranteed for all experiments
#         output1 = self.embeddingNet(x1.to(device)) 
        
#         # If both x2 and x3 is filled, we return three outputs
#         if x3 != None and x2 != None:
#             output2 = self.embeddingNet(x2.to(device))
#             output3 = self.embeddingNet(x3.to(device))
#             return output1, output2, output3
        
#         # If x2 is filled, we return two outputs
#         elif x2 != None and x3 == None:
#             output2 = self.embeddingNet(x2.to(device))
            
#             # If we want to include the similarity network after the embedding
#             # network, we output only a vector of sigmoid activations
#             if self.similarityFlag:
                
#                 # TODO
#                 # torch.abs MUST BE GENERALIZED TO OTHER SIMILARITY FUNCTIONS
#                 embeddingSim = torch.abs(output1 - output2)
#                 outputSigmoid = nn.Sigmoid()(self.similarityNet(embeddingSim))
#                 return outputSigmoid
            
#             # Otherwise, we return the two embeddings    
#             else:
#                 return output1, output2
            
            
class Siamese_LeNet5_var(nn.Module):
    def __init__(self, similarityFlag=False):
        super(Siamese_LeNet5_var, self).__init__()
        
        self.similarityFlag = similarityFlag
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,
                               stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.dropout1 = nn.Dropout(p=0.2)
        
        
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,
                               stride=1, padding=2)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)
        self.dropout2 = nn.Dropout(p=0.2)
        
        
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=5,
                               stride=1, padding=2)
        self.act3 = nn.Tanh()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.2)
        
        
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                               stride=1)
        self.act4 = nn.Tanh()
        
        self.adaptivePool = nn.AdaptiveAvgPool2d((2, 2))
        self.dropout4 = nn.Dropout(p=0.2)
        
        self.fc = nn.Sequential(nn.Linear(1024, 128))
        
        
        
        self.layer1 = nn.Sequential(self.conv1, self.act1, self.pool1, self.dropout1)
        self.layer2 = nn.Sequential(self.conv2, self.act2, self.pool2, self.dropout2)
        self.layer3 = nn.Sequential(self.conv3, self.act3, self.pool3, self.dropout3)
        self.layer4 = nn.Sequential(self.conv4, self.act4, self.adaptivePool)
        
        self.embeddingNet = nn.Sequential(self.layer1, self.layer2, self.layer3,
                                          self.layer4, nn.Flatten(), self.dropout4, self.fc)
        
    # This effectively does the same as __call__(), but forward() from 
    # nn.Module has some needed properties
    
    def forward(self, *args, device='cuda'):
        
        outputTensors = ()
        for i, arg in enumerate(args):
            output = self.embeddingNet(arg.to(device))
            
            outputTensors += (output, )
            
        if i == 0:
            outputTensors = outputTensors[0]
            
        return outputTensors
        


class Siamese_MobileNetV3_var(nn.Module):
    def __init__(self, similarityFlag=False):
        super(Siamese_MobileNetV3_var, self).__init__()
        
        
        
        self.similarityFlag = similarityFlag
        
        # Collecting the MobileNetV3 architecture and customizing it as 
        # explained in the thesis.
        self.embeddingNet = PyTorchModels.mobilenet_v3_small(weights=None)
        self.embeddingNet.classifier[-1] = nn.Linear(in_features=1024,
                                                     out_features=128,
                                                     bias=True)
        
        # We only have one color-channel, so this must also be manually stated.
        self.embeddingNet.features[0][0] = nn.Conv2d(in_channels=1, 
                                                     out_channels=16, 
                                                     kernel_size=(3, 3), 
                                                     stride=(2, 2), 
                                                     padding=(1, 1), 
                                                     bias=False)


        # replace_conv2d(self.embeddingNet)
        
        if similarityFlag:
            
            # This part of the architecture takes as input an energy output of the
            # embeddings ( for example absolute value -> torch.abs(F1 - F2) = E_(1,2) ),
            # forward-propagates this energy output into a network part that spits
            # out a label between 0 and 1.
            self.similarityNet = nn.Linear(self.embeddingNet.fc.out_features, 1)
        
        
        # Here we add nn.Dropout layers after each InvertedResidual-block
        # self._modify_inverted_residual_blocks_with_dropout(p=0.2)
        
        
        
    # def _modify_inverted_residual_blocks_with_dropout(self, p):
        
    #     # Looping over the high-lever layers
    #     for i, layer, in enumerate(self.embeddingNet.features):
            
    #         # Checking if we are in an InvertedResidual-block
    #         if isinstance(layer, PyTorchModels.mobilenetv3.InvertedResidual):
                
    #             # Adding dropout after the InvertedResidual-Block
    #             self.embeddingNet.features[i] = nn.Sequential(layer,
    #                                                           nn.Dropout(p=p))
        
        
        
    def forward(self, *args, device='cuda'):
        
        outputTensors = ()
        for i, arg in enumerate(args):
            output = self.embeddingNet(arg.to(device))
            
            outputTensors += (output, )
            
        if i == 0:
            outputTensors = outputTensors[0]
            
        return outputTensors
            
            
# mobilenetv3 = Siamese_MobileNetV3_var().to('cuda')
# print(summary(mobilenetv3, input_size=(1,1,127,127)))
            
            
            
            
            
            
            
            
            
            
            
            
            

# This network is used for visualizing the training of the network.
class Siamese_ResNet18_var(nn.Module):
    def __init__(self, similarityFlag=False):
        super(Siamese_ResNet18_var, self).__init__()
        
        self.similarityFlag = similarityFlag
        
        # Collecting the resnet-architecture and customizing the input and output
        self.embeddingNet = PyTorchModels.resnet18(weights=None)
        
        num_in_channels = 1
        num_out_channels = self.embeddingNet.conv1.out_channels
        size_kernel = self.embeddingNet.conv1.kernel_size
        num_in_features = self.embeddingNet.fc.in_features
        
        
        replaceLayer = nn.Linear(in_features=num_in_features, out_features=128)
        self.embeddingNet.fc = nn.Sequential(nn.Dropout(p=0.2),
                                             replaceLayer)
        
        
        self.embeddingNet.conv1 = nn.Conv2d(in_channels=num_in_channels,
                                     out_channels=num_out_channels,
                                     kernel_size=size_kernel,
                                     stride=(2, 2), padding=(3, 3))
        # replace_conv2d(self.embeddingNet)
        
        if similarityFlag:
            
            # This part of the architecture takes as input an energy output of the
            # embeddings ( for example absolute value -> torch.abs(F1 - F2) = E_(1,2) ),
            # forward-propagates this energy output into a network part that spits
            # out a label between 0 and 1.
            self.similarityNet = nn.Linear(self.embeddingNet.fc.out_features, 1)
        
        
        
        
        
        
    def forward(self, *args, device='cuda'):
        
        outputTensors = ()
        for i, arg in enumerate(args):
            output = self.embeddingNet(arg.to(device))
            
            outputTensors += (output, )
            
        if i == 0:
            outputTensors = outputTensors[0]
            
        return outputTensors