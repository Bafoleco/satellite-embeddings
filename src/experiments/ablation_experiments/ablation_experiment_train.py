# Experiment 2: Hold One Out Experiment

# NOTE: Run all experiments from within the ablation_experiments folder.

import sys
sys.path.insert(1, '../')
from random import randint

import training_stage.train_multitask_model_tune as train_multitask_model_tune
import util, networks, tasks
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

model_params = {
    # randomly sample numbers from 4 to 204 estimators
    'batch_size': randint(4,200),
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'weight_decay': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'min_samples_split': truncnorm(a=0, b=1, loc=0.25, scale=0.1)
}

# load train data
dataset_elevation = tasks.create_dataset_ablation(transfrom, 'elevation', '../../data/raw/mosaiks_images')
dataset_income = tasks.create_dataset_ablation(transfrom, 'income', '../../data/raw/mosaiks_images')
dataset_roads = tasks.create_dataset_ablation(transfrom, 'roads', '../../data/raw/mosaiks_images')
dataset_treecover = tasks.create_dataset_ablation(transfrom, 'treecover', '../../data/raw/mosaiks_images')

#dataset_four = tasks.create_dataset_four(transfrom, '../../data/raw/mosaiks_images')

# four tasks chosen: treecover, elevation, roads, nightlights
# task number 3: train other models with taking out the tasks

# Train model on hold-out elevation dataset
torch.manual_seed(0)

print("Length of elevation dataset: ", len(dataset_elevation))

dataset_all = tasks.create_dataset_all(transfrom, '../../data/raw/mosaiks_images')
print("Length of all dataset: ", len(dataset_all))

train_set, val_set, test_set = torch.utils.data.random_split(dataset_elevation, [8000, 1000, 1000])
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

batch_size = 64
learning_rate = 0.0001
num_epochs = 10

# train_model(dataset, net, trainloader, valloader, testloader, num_epochs, learning_rate, model_name, batchsize, save_model=True)

plot_dir = "plots/"
model_dir = "models/"

analysis = tune.run(
    train_multitask_model_tune.train_model,
    config={"lr": tune.grid_search([0.001, 0.01, 0.1]), "batch_size": tune.randint(1, 100),
    "dataset": dataset_elevation,
    "net": net,
    "trainloader": trainloader,
    "testloader": testloader,
    "num_epochs": 10,
    "model_name": model_name,
    "plot_dir": plot_dir,
    "model_dir": model_dir,
    "save_model": True})

print("Best config: ", analysis.get_best_config(metric="val_loss"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()

# arguments:
#  dataset, net, trainloader, valloader, testloader, num_epochs, learning_rate, model_name, batchsize, plot_dir, model_dir = './models/', save_model=True
# Earlier line: train_multitask_model.train_model(dataset_elevation, net, trainloader, valloader, testloader, num_epochs, learning_rate, model_name, batch_size, plot_dir, model_dir, save_model=True)