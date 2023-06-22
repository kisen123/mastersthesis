# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:06:44 2023

@author: kisen
"""

import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

def polar_transform(img):
    rows, cols = img.shape[:2]
    cx, cy = cols//2, rows//2
    r_max = int(np.ceil(np.sqrt(cols**2 + rows**2)))

    out = np.zeros((rows, r_max), dtype=np.uint8)

    for x in range(cols):
        for y in range(rows):
            r = math.sqrt((x-cx)**2 + (y-cy)**2)
            theta = math.atan2(y-cy, x-cx)
            theta_degrees = (theta * 180) / math.pi
            if theta_degrees < 0:
                theta_degrees += 360
            y_out = int((theta_degrees * rows) / 360)
            x_out = int(r)
            if x_out < r_max:
                out[y_out,x_out] = img[y,x]

    return out

imgDir = "./images/0-0.tif";
img = np.array(Image.open(imgDir))[:,:,0]
polar_transform(img)


#%%
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import math

import cv2

def linear_polar(img, center, maxRadius, flags=cv2.WARP_FILL_OUTLIERS, interpolation=cv2.INTER_LINEAR):
    # Check input shape and type

    
    # Compute output shape and type
    h, w = img.shape[:]
    out_shape = (int(maxRadius), 180)
    out_type = img.dtype
    
    # Allocate output tensor on device
    out = torch.zeros((1,) + out_shape, dtype=torch.float32, device=img.device)
    
    # Compute grid coordinates for output tensor
    r, theta = torch.meshgrid(torch.arange(0, out_shape[0]).float(), torch.arange(0, out_shape[1])*math.pi/180)
    x = center[0] + r*torch.cos(theta)
    y = center[1] + r*torch.sin(theta)
    
    # Convert grid coordinates to pixel coordinates
    x_norm = 2*x/w - 1
    y_norm = 2*y/h - 1
    grid = torch.stack((x_norm, y_norm), dim=-1)
    
    # Call PyTorch grid_sample function on input tensor
    out = F.grid_sample(img, grid.unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True)
    
    # Return output tensor
    return out.type(out_type)

imgDir = "./syntheticFolder/images/0-0.tif";
img = Image.open(imgDir)
convert_tensor = transforms.ToTensor()
img = convert_tensor(img)
linear_polar(img[0,:,:], (64.5, 64.5), 64.5)
