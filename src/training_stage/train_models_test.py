import sys
sys.path.append('../')

import torch

# local imports
import training_stage.networks as networks, dataset.tasks as tasks
import training_stage.train_multitask_model as train_multitask_model

# set batch_size and number of workers
batch_size = 64
num_workers = 2

# initialize network
net, model_name, transfrom = networks.get_visiontransformer(6, pretrained=True)

# load train data
dataset = tasks.create_dataset_all(transfrom)

# get length of train and valid
train_len = int(0.8 * len(dataset))
valid_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - valid_len

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, valid_len, test_len])
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# train model
train_multitask_model.train_model(dataset, net, trainloader, valloader, testloader, 1, 0.00005, model_name, batch_size)