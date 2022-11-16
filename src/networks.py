import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import pandas as pd
import os
import io
from PIL import Image
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

# import ResNet18_Weights 


# local imports
from dataloader import SatDataset


def get_resnet50(outputs, pretrained=True):
    if pretrained:
        pretrained_weights = ResNet50_Weights.DEFAULT
    else:
        pretrained_weights = None

    net = models.resnet50(weights=pretrained_weights) # try with both pre-trained and not pre-trained ResNet model!
    num_ftrs = net.fc.in_features
    print("num_ftrs", num_ftrs)
    net.fc = nn.Linear(in_features=num_ftrs, out_features=outputs)

    return net,  pretrained_weights.transforms() if pretrained else None

def get_resnet18(outputs, pretrained=True):
    if pretrained:
        pretrained_weights = ResNet18_Weights.DEFAULT
    else:
        pretrained_weights = None

    net = models.resnet18(weights=pretrained_weights) # try with both pre-trained and not pre-trained ResNet model!
    num_ftrs = net.fc.in_features
    print("num_ftrs", num_ftrs)
    net.fc = nn.Linear(in_features=num_ftrs, out_features=outputs)

    return net,  pretrained_weights.transforms() if pretrained else None

class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net(nn.Module):
    ''' Models a simple Convolutional Neural Network'''
	
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5) 
        self.fc1 = nn.Linear(37210, 120)# 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        # 1 real output
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # print("forward called with")
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 37210)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # print("returning")
        # print(x.shape)
        return x
