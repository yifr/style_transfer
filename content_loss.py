from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

'''
Content loss is a function representing a weighted version of the content distance for an individual layer.
The function takes feature maps F_xl of layer L in a network processing X and returns weighted
content distance W_cl * D^l_c(X, C) betwenn image X and content image C.

This function is implemented as a torch module with a constructor that takes F_cl as input.
The distance ||F_xl - F_cl|| ^ 2 is the MSE between the two feature maps.
'''

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
