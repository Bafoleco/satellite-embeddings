import matplotlib.pyplot as plt # for plotting
import numpy as np
from util import get_model_loss, get_percent_error, graph_performance # for transformation

import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# local imports
from dataloader import SatDataset
from networks import ConvolutionalNeuralNet, Net
import util, networks, tasks
import train_multitask_model

# set batch_size and number of workers
batch_size = 64

# initialize network
net, model_name, transfrom = networks.get_visiontransformer(6, pretrained=True)

print("Num Params - Total, Trainable")

pytorch_total_params = sum(p.numel() for p in net.parameters())
print(pytorch_total_params)

pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(pytorch_trainable_params)

# load train data
dataset = tasks.create_dataset_all(transfrom)

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [8000, 1000, 1000])
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# train model
train_model(net, trainloader, valloader, testloader, 10, 0.001, model_name, batch_size)