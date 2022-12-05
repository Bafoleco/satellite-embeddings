# Experiment 2: Hold One Out Experiment

import train_multitask_model

import util, networks, tasks

import torch

batch_size = 64 # TODO: Reason through batch size
num_workers = 2

# initialize network
net, model_name, transfrom = networks.get_visiontransformer(6, pretrained=True)

print("Num Params - Total, Trainable")

pytorch_total_params = sum(p.numel() for p in net.parameters())
print(pytorch_total_params)

pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(pytorch_trainable_params)

# load train data
dataset_elevation = tasks.create_dataset_ablation(transfrom, 'elevation')
dataset_income = tasks.create_dataset_ablation(transfrom, 'income')
dataset_nightlights = tasks.create_dataset_ablation(transfrom, 'nightlights')
dataset_population = tasks.create_dataset_ablation(transfrom, 'population')
dataset_roads = tasks.create_dataset_ablation(transfrom, 'roads')
dataset_treecover = tasks.create_dataset_ablation(transfrom, 'treecover')

# Train model on hold-out elevation dataset

train_set, val_set, test_set = torch.utils.data.random_split(dataset_elevation, [8000, 1000, 1000])
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

train_model(net, trainloader, valloader, testloader, 10, 0.001, model_name, save_model=True)