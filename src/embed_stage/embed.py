import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights, VGG13_Weights
import os

import training_stage.networks as networks, dataset.tasks as tasks
import util.util as util

# feature extraction is adapted from https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/05/27/extracting-features.html

FEATURE_KEY = 'feats'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_input_features(temp_features):
    def hook(model, input, output):
        # print(model)
        print(input)
        # print(input[0].shape)
        temp_features[FEATURE_KEY] = input[0].detach()
    return hook

def get_last_layer(model):
    has_children = True
    last_layer = model
    while has_children:
        for child in last_layer.children():
            last_layer = child

        if len(list(last_layer.children())) > 0:
            has_children = True
        else:
            has_children = False
    
    return last_layer

def embed_images(model, transform):
    temp_features = {}
    embeddings = {}

    final_layer = get_last_layer(model)

    print(final_layer)

    # print("Weights: " + str(final_layer.weight.shape))
    # # print("Bias: " + str(bias.shape)) for bias in final_layer.bias

    # print(final_layer.weight)


    # we must always apply hook to last hidden layer
    final_layer.register_forward_hook(get_input_features(temp_features))

    dataset = tasks.create_dataset_treecover(transform, image_root=util.get_eval_images_path())
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

def save_embeddings(model_name):
    model = torch.load(os.path.join(util.get_models_path(), model_name + ".pth"))

    transform = networks.get_weights(model_name).transforms()

    embeddings = embed_images(model, transform)
    embeddings_name = model_name + "_embeddings.pkl"

    with open(os.path.join(util.get_embeddings_path(), embeddings_name), 'wb') as f:
        pickle.dump(embeddings, f)

model_name = "pretrained_visiontransformer_ElInPoRdTrNl"
