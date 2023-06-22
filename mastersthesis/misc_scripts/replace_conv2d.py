# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:26:17 2023

@author: kisen
"""

from models.cyconvlayer import CyConv2d
from torch import nn

def replace_conv2d(module):
    # Recursively replace nn.Conv2d with CustomConv2d
    new_module = module.__class__.__new__(module.__class__)
    new_module.__dict__ = module.__dict__.copy()
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(new_module, name, CyConv2d(child.in_channels,
                                                child.out_channels,
                                                child.kernel_size[0],
                                                child.stride[0],
                                                child.padding,
                                                child.dilation))
        elif isinstance(child, nn.Sequential):
            new_seq = nn.Sequential(*[replace_conv2d(sub_child) for sub_child in child])
            setattr(new_module, name, new_seq)
        else:
            setattr(new_module, name, replace_conv2d(child))
    return new_module