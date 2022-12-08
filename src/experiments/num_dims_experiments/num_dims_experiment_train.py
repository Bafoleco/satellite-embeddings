# Experiment 2: Hold One Out Experiment
# NOTE: Run all experiments from within the ablation_experiments folder.

import sys
sys.path.insert(1, '../../')

import training_stage.train_multitask_model as train_multitask_model
import util.util as util, training_stage.networks as networks, dataset.tasks as tasks
import torch
from scipy.stats import truncnorm
# print winning set of hyperparameters
from ray import tune

batch_size = 64
num_workers = 4

# acquire transform
_, _, transfrom = networks.get_visiontransformer(1, 1, pretrained=True)

# load train data
dataset = tasks.create_dataset_ablation(transfrom)

dim_list = [4096, 2048, 512]

for dim in dim_list:
    net, model_name, transfrom = networks.get_visiontransformer(len(dataset.tasks), dim, pretrained=True)
    print(model_name)

    train_len = int(0.8 * len(dataset))
    valid_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - valid_len

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, valid_len, test_len], torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    batch_size = 64
    learning_rate = 0.00001
    num_epochs = 7
    weight_decay = 0.01

    train_multitask_model.train_model(dataset, net, trainloader, valloader, testloader, num_epochs, learning_rate, weight_decay, model_name, batch_size)