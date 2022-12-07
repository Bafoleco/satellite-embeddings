import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights
import os

import training_stage.networks as networks, dataset.tasks as tasks
import utils.util as util

# feature extraction is adapted from https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html

FEATURE_KEY = 'feats'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_input_features(temp_features):
    def hook(model, input, output):
        print(input[0].shape)
        temp_features[FEATURE_KEY] = input[0].detach()
    return hook

def embed_images(model):
    temp_features = {}
    embeddings = {}

    for child in model.children():
        last_layer = child

    # we must always apply hook to last hidden layer
    last_layer.register_forward_hook(get_input_features(temp_features))

    # TODO use right transform - this is now for vision - change for resnet!
    transform = ViT_B_16_Weights.DEFAULT.transforms()
    dataset = tasks.create_dataset_treecover(transform, image_root="../data/raw/eval_images")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    with torch.no_grad():
        for i, (inputs, _, ids) in enumerate(data_loader):
            print("Batch " + str(i))
 
            inputs = inputs.to(device)
            model(inputs)

            # get features from last hidden layer
            for i, file_name in enumerate(ids):
                embeddings[file_name] = temp_features[FEATURE_KEY][i].cpu().numpy()            

    return embeddings

def save_embeddings(model_name, path):
    model = torch.load("./models/" + model_name + ".pth")

    embeddings = embed_images(model)
    embeddings_name = model_name + "_embeddings.pkl"

    with open(os.path(path, embeddings_name), 'wb') as f:
        pickle.dump(embeddings, f)

model_name = "pretrained_visiontransformer_ElInPoRdTrNl"
