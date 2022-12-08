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

net, model_name, transfrom = networks.get_visiontransformer(1, 1024, pretrained=True)

name = "baseline_model"
source_path = Path(__file__).resolve()
source_dir = source_path.parent.parent

torch.save(net, os.path.join(source_dir.parent, "out", "models", name + ".pth"))

# Baseline:
#  File "/home/nnaik39/satellite-embeddings/src/embed_stage/../embed_stage/embed.py", line 73, in save_embeddings
#    transform = networks.get_weights(model_name).transforms()
#  File "/home/nnaik39/satellite-embeddings/src/embed_stage/../training_stage/networks.py", line 38, in get_weights
#    raise None
#TypeError: exceptions must derive from BaseException