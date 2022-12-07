import matplotlib.pyplot as plt # for plotting
import numpy as np
from util import get_model_loss, get_percent_error, graph_performance # for transformation
import os
from sklearn.model_selection import RandomizedSearchCV

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

from ray import tune

#####
#####

# set batch_size and number of workers
batch_size = 64
num_workers = 2

#     config={"lr": tune.grid_search([0.001, 0.01, 0.1]), "batch_size": tune.randint(1, 100),
#    "dataset": dataset_elevation,
 #   "net": net,
 #   "trainloader": trainloader,
 #   "testloader": testloader,
 #   "num_epochs": 10,
 #   "model_name": model_name,
 #   "plot_dir": plot_dir,
 #   "model_dir": model_dir,
 #   save_model = True})

def train_model(config, checkpoint_dir = None):
    net = config['net']
    save_model = config['save_model']
    dataset = config['dataset']
    trainloader = config['trainloader']
    testloader = config['testloader']
    num_epochs = config['num_epochs']
    model_name = config['model_name']
    plot_dir = config['plot_dir']
    model_dir = config['model_dir']

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=0.001)

    # setup for GPU training
    print("CUDA status: ", torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    print("STARTING TRAINING LOOP")

    x, val_losses, train_losses = [], [], []
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        avg = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 1:
                print("epoch " + str(epoch) + ", i=" + str(i) + " loss=" + str(running_loss / 10))
                avg = running_loss / 10
                running_loss = 0.0
                
        val_loss = get_model_loss(valloader, dataset, net, criterion)
        tune.report(val_loss=val_loss)
        print("Validation loss: " + str(val_loss))
        x.append(epoch)
        val_losses.append(val_loss)
        train_losses.append(avg)

        plt.close()
        plt.scatter(x, val_losses, label="Validation Loss")
        plt.scatter(x, train_losses, label="Training Loss")
        plt.legend()

        if (not os.path.exists(plot_dir)):
            os.mkdir(plot_dir)

        plt.savefig(plot_dir + model_name + "loss.png")

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Finished Training')
    net.eval()

    # graphing performance
    graph_performance(plot_dir, valloader, dataset, net)

    # save model
    name = model_name + "_" + dataset.get_task_code()

    if (not os.path.exists(model_dir)):
        os.makedirs(model_dir)
        
    torch.save(net, model_dir + name + ".pth")
