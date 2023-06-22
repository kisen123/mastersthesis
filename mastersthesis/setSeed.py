# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:45:17 2023

@author: kisen
"""

import torch
import numpy as np
import random

def setSeed(seed, cuda=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False