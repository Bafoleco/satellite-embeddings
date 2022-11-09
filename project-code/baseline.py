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
import util, networks

#####
#####

# set batch_size
batch_size = 64
# set number of workers
num_workers = 4

# initialize network
net, transfrom = networks.get_resnet(1, pretrained=True)

print("Num Params - Total, Trainable")

pytorch_total_params = sum(p.numel() for p in net.parameters())
print(pytorch_total_params)

pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(pytorch_trainable_params)

# load train data
housing_data = "../data/int/applications/housing/outcomes_sampled_housing_CONTUS_16_640_POP_100000_0.csv"
image_root = "../data/raw/mosaiks_images"
dataset = util.create_dataset_treecover(transfrom)

print("length of dataset ", len(dataset))

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [8000, 1000, 1000])
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


model_name = "pretrained_resnet"

# print(net)

criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.01)


# setup for GPU training
print("CUDA status: ", torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

print("STARTING TRAINING LOOP")


x, val_losses, train_losses = [], [], []
for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    avg = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
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


# whatever you are timing goes here
# end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print('Finished Training')
print('Measuring Test Accuracy')

net.eval()

# with torch.no_grad():
#         correct = 0
#         total = 0
#         total_losss =0

#         for data, target in testloader:
#             images = data
#             labels = target
#             outputs = net(images)
#             print("Outputs ", outputs)
#             loss = criterion(outputs, labels[:, None])
#             total += labels.size(0)
#             print("label ", labels)

# #            correct += (outputs == labels).sum().item()
#             total_losss += loss.item()

#             accuracy = correct / total

#         print('Test Accuracy of the model: {} %'.format(100 * correct / total))
#         print('Test log loss ', total_losss/total)
#         print('Test accuracy ', ((correct / total) * 100))

print('Measuring Validation Accuracy')

net.eval()

# graphing 

x = []
y = []

graph_performance(valloader, dataset, net)




torch.save(net.state_dict(), "./models/" + model_name + ".pth")