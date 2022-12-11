import sys
sys.path.insert(1, '../../')
import os 
import training_stage.train_multitask_model as train_multitask_model
import util.util as util, training_stage.networks as networks, dataset.tasks as tasks
import torch
from scipy.stats import truncnorm
# print winning set of hyperparameters
from ray import tune
from pathlib import Path

batch_size = 64
num_workers = 4

net, model_name, transfrom = networks.get_visiontransformer(4, 1024, pretrained=True)
net, model_name, transfrom = networks.get_baseline()

print(net)

source_path = Path(__file__).resolve()
source_dir = source_path.parent.parent

torch.save(net, os.path.join(util.get_models_path(), model_name + ".pth"))