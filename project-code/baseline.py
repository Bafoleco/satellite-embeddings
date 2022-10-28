import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
from torch.utils.data import Dataset
import pandas as pd
import os
import io
from PIL import Image

# python image library of range [0, 1] 
# transform them to tensors of normalized range[-1, 1]

transform = transforms.Compose( # composing several transforms together
    [torchvision.transforms.Resize(256),
     transforms.ToTensor(), # to tensor object
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # mean = 0.5, std = 0.5


#####
# DataLoader
#####

def image_present(root_dir, indices):

    # print(indices)

    ij = indices.split(",")
    i = ij[0]
    j = ij[1]

    file_name = str(i) + "_" + str(j) + ".png"
    return os.path.exists(os.path.join(root_dir, file_name))

class SatDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        all_labels = pd.read_csv(csv_file)
        
        # filter missing

        # self.labels = self.labels.apply(lambda row: row[image_present(root_dir, row[0])], axis=1)


        mask = all_labels.apply(lambda row: image_present(root_dir, row[0]), axis=1)
        print(all_labels[mask])

        self.labels = all_labels[mask]

        print("filtered missing")


        self.root_dir = root_dir
        self.transform = transform

        print(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # print("get item")
        # print(idx)


        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # else:
        #     idx = [idx]
        
        # images = []
        # prices = []

        i = idx

        # for i in idx:
        file_name = self.labels.iloc[i, 0].replace(",", "_") + ".png"

        # print(file_name)

        img_name = os.path.join(self.root_dir,
                                file_name)
        image = Image.open(img_name).convert('RGB')


        # print(image)


        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'price': self.labels.iloc[idx, 1]}

        # if self.transform:
        #     sample = self.transform(sample)
    
        price = self.labels.iloc[idx, 1]
        # prices.append(price)
        # images.append(self.transform(image))
        return [self.transform(image), np.log(price.astype(np.float32))]



print(torch.version.cuda)

#####
#####

# set batch_size
batch_size = 4

# set number of workers
num_workers = 2

# load train data
housing_data = "data/int/applications/housing/outcomes_sampled_housing_CONTUS_16_640_POP_100000_0.csv"
image_root = "data/raw/mosaiks_images"
trainset = SatDataset(housing_data, image_root, transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

# load test data
testset = trainset
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

# # put 10 classes into a set
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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

net = Net()
print(net)


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

print("STARTING TRAINING LOOP")

for epoch in range(6):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # print('iter')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels[:, None])

        # print("predictions:")
        # print(inputs.shape)
        # print(labels.shape)
        # print(type(labels))
        # print(outputs.shape)
        # print(outputs)
        # print(labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print(running_loss)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# whatever you are timing goes here
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print('Finished Training')
print(start.elapsed_time(end))  # milliseconds