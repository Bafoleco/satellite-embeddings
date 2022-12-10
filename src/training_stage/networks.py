import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import sys
sys.path.append('../')

import torch # PyTorch package
import torchvision # load datasets
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import pandas as pd
from PIL import Image
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_32_Weights, Wide_ResNet101_2_Weights, VGG13_Weights, ViT_L_16_Weights

# local imports
from dataset.dataloader import SatDataset


name_to_weights = {
    "pretrained_resnet18": ResNet18_Weights.DEFAULT,
    "pretrained_resnet50": ResNet50_Weights.DEFAULT,
    "pretrained_visiontransformer": ViT_B_16_Weights.DEFAULT,
    "pretrained_vit_b_32": ViT_B_32_Weights.DEFAULT,
    "pretrained_vit_l_32": ViT_L_32_Weights.DEFAULT,
    "pretrained_wide_resnet101_2": Wide_ResNet101_2_Weights.DEFAULT,
    "pretrained_vgg13": VGG13_Weights.DEFAULT,
    "baseline_model": ViT_B_16_Weights.DEFAULT
}

def get_weights(name):
    name = name[0: name.rfind("_")]

    if name.startswith("pretrained_visiontransformer"):
        name = "pretrained_visiontransformer"

    if name not in name_to_weights:
        raise None

    return name_to_weights[name]

def get_vgg13(outputs, pretrained=True):
    if pretrained:
        pretrained_weights = VGG13_Weights.DEFAULT
    else:
        pretrained_weights = None

    net = models.vgg13(weights=pretrained_weights) # try with both pre-trained and not pre-trained ResNet model!

    print(net)

    # num_ftrs = net.heads.head.in_features

    embed_dim = 8192

    embedder = nn.Sequential(
        nn.Linear(in_features=25088, out_features=embed_dim),
        nn.ReLU(),
        nn.Linear(in_features=embed_dim, out_features=outputs)
    )

    print(embed_dim)

    net.classifier = embedder

    # print("num_ftrs", num_ftrs)
    # net.heads.head = nn.Linear(in_features=num_ftrs, out_features=outputs)

    model_name = "pretrained_vgg13" if pretrained else "vgg13"

    return net, model_name, pretrained_weights.transforms() if pretrained else None


def get_vit_l_16(outputs, pretrained=True):
    if pretrained:
        pretrained_weights = ViT_L_16_Weights.DEFAULT
    else:
        pretrained_weights = None

    net = models.vit_l_16(weights=pretrained_weights) # try with both pre-trained and not pre-trained ResNet model!

    # print(net)

    num_ftrs = net.heads.head.in_features
    print("num_ftrs", num_ftrs)
    net.heads.head = nn.Linear(in_features=num_ftrs, out_features=outputs)

    model_name = "pretrained_vit_l_16" if pretrained else "vit_l_16"

    return net, model_name, pretrained_weights.transforms() if pretrained else None

def get_visiontransformer(outputs, embedding_dim, pretrained=True):
    if pretrained:
        pretrained_weights = ViT_B_16_Weights.DEFAULT
    else:
        pretrained_weights = None

    net = models.vit_b_16(weights=pretrained_weights)
    
    num_in_ftrs = net.heads.head.in_features
    net.heads.head = nn.Sequential(
        nn.Linear(in_features=num_in_ftrs, out_features=embedding_dim),
        nn.ReLU(),
        nn.Linear(in_features=embedding_dim, out_features=outputs)
    )

    model_name = "pretrained_visiontransformer" if pretrained else "visiontransfromer"
    model_name += f"_{embedding_dim}"

    return net, model_name, pretrained_weights.transforms() if pretrained else None

def get_baseline():
    net = models.vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
    model_name = "baseline_model"
    return net, model_name, ViT_L_16_Weights.DEFAULT.transforms()

def get_resnet50(outputs, pretrained=True):
    if pretrained:
        pretrained_weights = ResNet50_Weights.DEFAULT
    else:
        pretrained_weights = None

    net = models.resnet50(weights=pretrained_weights) # try with both pre-trained and not pre-trained ResNet model!
    num_ftrs = net.fc.in_features
    print("num_ftrs", num_ftrs)
    net.fc = nn.Linear(in_features=num_ftrs, out_features=outputs)

    model_name = "pretrained_resnet50" if pretrained else "resnet50"

    return net, model_name, pretrained_weights.transforms() if pretrained else None

def get_resnet18(outputs, pretrained=True):
    if pretrained:
        pretrained_weights = ResNet18_Weights.DEFAULT
    else:
        pretrained_weights = None

    net = models.resnet18(weights=pretrained_weights) # try with both pre-trained and not pre-trained ResNet model!
    num_ftrs = net.fc.in_features
    print("num_ftrs", num_ftrs)
    net.fc = nn.Linear(in_features=num_ftrs, out_features=outputs)

    model_name = "pretrained_resnet18" if pretrained else "resnet18"

    return net, model_name, pretrained_weights.transforms() if pretrained else None



net = get_visiontransformer(10, 1024, pretrained=True)

