'''
Based Neural-Style algorithm developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
Tutorial code: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import content_loss
import copy

#configure network to run on GPU or CPU depending on hardware:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

imsize = 512 if torch.cuda.is_available() else 128 #smaller image size if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style = input('Please provide the path to the photo to use as the style picture: ')
style_img = image_loader(style)

content = input('Please provide the path to the photo to use as the content picture: ')
content_img = image_loader(content)

assert style_img.size() == content_img.size(), \
        'Please make sure that the style and content images are the same size'

unloader = transforms.ToPILImage()

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone() #clone the tensor to avoid altering it
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(3)

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')



