# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 12:23:55 2023

@author: kisen
"""

from torch import nn


"""
Cylindrical convolution based of the paper: 
    
    CyCNN: A Rotation Invariant CNN using 
    Polar Mapping and Cylindrical Convolution Layers
    
    URL to paper:
    https://arxiv.org/pdf/2007.10588.pdf
"""
from models.cyconvlayer import CyConv2d



def replaceConvWithCyConv(module):
    
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, CyConv2d(in_channels=child.in_channels, 
                                           out_channels=child.out_channels, 
                                           kernel_size=child.kernel_size[0], 
                                           stride=child.stride[0], 
                                           padding=child.padding[0], 
                                           dilation=child.dilation[0],
                                           groups=child.groups))
        
        else:
            replaceConvWithCyConv(child)
            
    return module
            

# Cy_Siamese_LeNet5_var = replaceConvWithCyConv(Siamese_LeNet5_var())
# Cy_Siamese_MobileNetV3_var = replaceConvWithCyConv(Siamese_MobileNetV3_var())
# Cy_Siamese_ResNet18_var = replaceConvWithCyConv(Siamese_ResNet18_var())

#%%
# class Cy_Siamese_LeNet5_var(nn.Module):
#     def __init__(self):
#         super(Cy_Siamese_LeNet5_var, self).__init__()
        
        
        
#         self.conv1 = CyConv2d(in_channels=1, out_channels=6, kernel_size=5,
#                                stride=1, padding=2)
#         self.act1 = nn.Tanh()
#         self.pool1 = nn.AvgPool2d(kernel_size=3, stride=3)
        
        
        
#         self.conv2 = CyConv2d(in_channels=6, out_channels=32, kernel_size=5,
#                                stride=1, padding=2)
#         self.act2 = nn.Tanh()
#         self.pool2 = nn.AvgPool2d(kernel_size=3, stride=3)
        
        
        
        
#         self.conv3 = CyConv2d(in_channels=32, out_channels=128, kernel_size=5,
#                                stride=1, padding=2)
#         self.act3 = nn.Tanh()
#         self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        
        
        
#         self.conv4 = CyConv2d(in_channels=128, out_channels=512, kernel_size=3,
#                                stride=1)
#         self.act4 = nn.Tanh()
        
#         self.adaptivePool = nn.AdaptiveAvgPool2d((1, 1))
        
        
#         self.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(512, 128))
        
        
        
#         self.layer1 = nn.Sequential(self.conv1, self.act1, self.pool1)
#         self.layer2 = nn.Sequential(self.conv2, self.act2, self.pool2)
#         self.layer3 = nn.Sequential(self.conv3, self.act3, self.pool3)
#         self.layer4 = nn.Sequential(self.conv4, self.act4, self.adaptivePool)
        
#         self.embeddingNet = nn.Sequential(self.layer1, self.layer2, self.layer3,
#                                           self.layer4, nn.Flatten(), self.fc)
        
        
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















# class Cy_Siamese_MobileNetV3_var(nn.Module):
#     def __init__(self, similarityFlag=False):
#         super(Cy_Siamese_MobileNetV3_var, self).__init__()
        
        
        
#         self.similarityFlag = similarityFlag
        
#         # Collecting the MobileNetV3 architecture and customizing it as 
#         # explained in the thesis.
#         self.embeddingNet = PyTorchModels.mobilenet_v3_small(weights=None)
#         self.embeddingNet.classifier[-1] = nn.Linear(in_features=1024,
#                                                      out_features=128,
#                                                      bias=True)
        
#         # We only have one color-channel, so this must also be manually stated.
#         self.embeddingNet.features[0][0] = CyConv2d(in_channels=1, 
#                                                      out_channels=16, 
#                                                      kernel_size=(3, 3), 
#                                                      stride=(2, 2), 
#                                                      padding=(1, 1), 
#                                                      bias=False)


#         # replace_conv2d(self.embeddingNet)
        
#         if similarityFlag:
            
#             # This part of the architecture takes as input an energy output of the
#             # embeddings ( for example absolute value -> torch.abs(F1 - F2) = E_(1,2) ),
#             # forward-propagates this energy output into a network part that spits
#             # out a label between 0 and 1.
#             self.similarityNet = nn.Linear(self.embeddingNet.fc.out_features, 1)
        
        
        
        
#     def forward(self, x1, x2=None, x3=None):
        
#         # One forward pass is guaranteed for all experiments
#         output1 = self.embeddingNet(x1) 
        
#         # If both x2 and x3 is filled, we return three outputs
#         if x3 != None and x2 != None:
#             output2 = self.embeddingNet(x2)
#             output3 = self.embeddingNet(x3)
#             return output1, output2, output3
        
#         # If x2 is filled, we return two outputs
#         elif x2 != None and x3 == None:
#             output2 = self.embeddingNet(x2)
            
#             # If we want to include the similarity network after the embedding
#             # network, we output only a vector of sigmoid activations
#             if self.similarityFlag:
                
#                 # torch.abs MUST BE GENERALIZED TO OTHER SIMILARITY FUNCTIONS
#                 embeddingSim = torch.abs(output1 - output2)
#                 outputSigmoid = nn.Sigmoid()(self.similarityNet(embeddingSim))
#                 return outputSigmoid
            
#             # Otherwise, we return the two embeddings    
#             else:
#                 return output1, output2
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

# # This network is used for visualizing the training of the network.
# class Cy_Siamese_ResNet18_var(nn.Module):
#     def __init__(self, similarityFlag=False):
#         super(Cy_Siamese_ResNet18_var, self).__init__()
        
#         self.similarityFlag = similarityFlag
        
#         # Collecting the resnet-architecture and customizing the input and output
#         self.embeddingNet = PyTorchModels.resnet18()
        
#         num_in_channels = 1
#         num_out_channels = self.embeddingNet.conv1.out_channels
#         size_kernel = self.embeddingNet.conv1.kernel_size
#         num_in_features = self.embeddingNet.fc.in_features
        
        
#         replaceLayer = nn.Linear(in_features=num_in_features, out_features=128)
#         self.embeddingNet.fc = nn.Sequential(nn.Dropout(p=0.2),
#                                              replaceLayer)
        
        
#         self.embeddingNet.conv1 = nn.Conv2d(in_channels=num_in_channels,
#                                      out_channels=num_out_channels,
#                                      kernel_size=size_kernel,
#                                      stride=(2, 2), padding=(3, 3))
#         # replace_conv2d(self.embeddingNet)
        
#         if similarityFlag:
            
#             # This part of the architecture takes as input an energy output of the
#             # embeddings ( for example absolute value -> torch.abs(F1 - F2) = E_(1,2) ),
#             # forward-propagates this energy output into a network part that spits
#             # out a label between 0 and 1.
#             self.similarityNet = nn.Linear(self.embeddingNet.fc.out_features, 1)
        
        
        
        
        
        
#     def forward(self, x1, x2=None, x3=None):
        
#         # One forward pass is guaranteed for all experiments
#         output1 = self.embeddingNet(x1) 
        
#         # If both x2 and x3 is filled, we return three outputs
#         if x3 != None and x2 != None:
#             output2 = self.embeddingNet(x2)
#             output3 = self.embeddingNet(x3)
#             return output1, output2, output3
        
#         # If x2 is filled, we return two outputs
#         elif x2 != None and x3 == None:
#             output2 = self.embeddingNet(x2)
            
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
            

