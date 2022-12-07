import matplotlib.pyplot as plt # for plotting
import numpy as np
from pathlib import Path
import os

import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

# local imports
from util.util import get_model_loss, get_percent_error, graph_performance

source_path = Path(__file__).resolve()
source_dir = source_path.parent

def train_model(dataset, net, trainloader, valloader, testloader, num_epochs, learning_rate, model_name, save_model=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.003)

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

        print("Validation loss: " + str(val_loss))
        x.append(epoch)
        val_losses.append(val_loss)
        train_losses.append(avg)

        plt.close()
        plt.scatter(x, val_losses, label="Validation Loss")
        plt.scatter(x, train_losses, label="Training Loss")
        plt.legend()
        plt.savefig("loss.png")

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('Finished Training')
    net.eval()

    # graphing performance
    graph_performance("./plots/", valloader, dataset, net)

    # save model
    name = model_name + "_" + dataset.get_task_code()
    torch.save(net, os.path.join(source_dir.parent, "out", "models", name + ".pth"))
