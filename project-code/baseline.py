import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torchvision.models as models
import numpy as np

# local imports
from dataloader import SatDataset
from networks import ConvolutionalNeuralNet, Net
import matplotlib.pyplot as plt

#####
#####

# set batch_size
batch_size = 64

# set number of workers
num_workers = 4

# load train data
housing_data = "data/int/applications/housing/outcomes_sampled_housing_CONTUS_16_640_POP_100000_0.csv"
image_root = "data/raw/mosaiks_images"
dataset = SatDataset(housing_data, image_root)

print("length of dataset ", len(dataset))

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [1800, 225, 225])
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


net = models.resnet18(pretrained=True) # try with both pre-trained and not pre-trained ResNet model!
net.fc = nn.Linear(in_features=512, out_features=1)
model_name = "pretrained_resnet"

print(net)

criterion = nn.MSELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

print("STARTING TRAINING LOOP")

def get_percent_error(outputs, labels):
    # print(outputs.shape)
    # print(labels.shape)

    return np.average(np.abs(outputs - labels) / labels) * 100

for epoch in range(6):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, pred = torch.max(outputs, 1)

        pred = pred.unsqueeze(1)
        loss = criterion(outputs, labels[:, None])

        loss.backward()
        optimizer.step()

        # print statistics
        # print('loss item ', loss.item())

        # print(get_percent_error(outputs, labels))

        # TODO check if this is common technique
        if (np.isinf(loss.item())):
            running_loss += 10000000000
        else:
            running_loss += loss.item()

        print("epoch " + str(epoch) + ", i=" + str(i) + " loss=" + str(loss.item()))


        # if i % 100 == 0:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

# whatever you are timing goes here
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print('Finished Training')
print('Measuring Test Accuracy')

net.eval()

with torch.no_grad():
        correct = 0
        total = 0
        total_losss =0

        for data, target in testloader:
            images = data
            labels = target
            outputs = net(images)
            print("Outputs ", outputs)
            loss = criterion(outputs, labels[:, None])
            total += labels.size(0)
            print("label ", labels)

#            correct += (outputs == labels).sum().item()
            total_losss += loss.item()

            accuracy = correct / total

        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
        print('Test log loss ', total_losss/total)
        print('Test accuracy ', ((correct / total) * 100))

print('Measuring Validation Accuracy')

net.eval()

# graphing 

x = []
y = []

with torch.no_grad():
        correct = 0
        num_predictions = 0
        total_loss = 0

        percent_error = 0
        
        batchs = 0
        for data, target in valloader:
            images = data
            labels = target
            outputs = net(images)
            batchs += 1

            print(dataset.transform_output(outputs.detach()))

            #  print("Outputs ", outputs)
            loss = criterion(outputs, labels[:, None])
            num_predictions += labels.size(0)
            # print("label ", labels)

            # correct += (outputs == labels).sum().item()
            total_loss += loss.item()
            
            print(labels.shape)
            print(dataset.transform_output(outputs.detach()).shape)
            print(outputs.shape)


            price_predictions = dataset.transform_output(outputs.detach())
            true_prices = dataset.transform_output(labels)
            percent_error += get_percent_error(price_predictions, true_prices)

            for i in range(len(true_prices)):
                x.append(true_prices[i])
                y.append(price_predictions[i])

        print(num_predictions)
        print('Validation loss ', total_loss/num_predictions)
        print('Validation percent error ', percent_error / batchs)

        plt.scatter(x, y)
        plt.xlabel("True Prices")
        plt.ylabel("Predicted Prices")
        plt.savefig("predictions.png")



torch.save(net.state_dict(), "./models/" + model_name + ".pt")