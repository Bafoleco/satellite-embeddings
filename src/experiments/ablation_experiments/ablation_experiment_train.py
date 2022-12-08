# Experiment 2: Hold One Out Experiment

# NOTE: Run all experiments from within the ablation_experiments folder.

import sys
sys.path.insert(1, '../../')
from random import randint

import training_stage.train_multitask_model as train_multitask_model
import util.util as util, training_stage.networks as networks, dataset.tasks as tasks
import torch
from scipy.stats import truncnorm
# print winning set of hyperparameters
from ray import tune

batch_size = 64
num_workers = 2

# initialize network

num_tasks = 3

net, model_name, transfrom = networks.get_visiontransformer(num_tasks, pretrained=True)

print("Num Params - Total, Trainable")

pytorch_total_params = sum(p.numel() for p in net.parameters())
print(pytorch_total_params)

pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(pytorch_trainable_params)

# load train data
dataset_allfour = tasks.create_dataset_ablation(transfrom)
dataset_elevation = tasks.create_dataset_ablation(transfrom, 'elevation')
dataset_income = tasks.create_dataset_ablation(transfrom, 'income')
dataset_roads = tasks.create_dataset_ablation(transfrom, 'roads')
dataset_treecover = tasks.create_dataset_ablation(transfrom, 'treecover')

#dataset_four = tasks.create_dataset_four(transfrom, '../../data/raw/mosaiks_images')
torch.manual_seed(0)

dataset = dataset_allfour

train_len = int(0.8 * len(dataset))
valid_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - valid_len

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, valid_len, test_len])
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

batch_size = 64
learning_rate = 0.00001
num_epochs = 5
weight_decay = 0.01

model_name += "num_epochs=" + str(num_epochs)

train_multitask_model.train_model(dataset, net, trainloader, valloader, testloader, num_epochs, learning_rate, weight_decay, model_name, batch_size)